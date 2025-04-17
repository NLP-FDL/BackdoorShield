import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from collections import Counter
from transformers import BertModel, BertTokenizer, AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import logging
import random
from typing import List, Dict, Optional

# 配置设置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 数据集类
class SST2Dataset(Dataset):
    def __init__(self, filepath: str, tokenizer: BertTokenizer, max_length: int = 128):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Skip header
            
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                self.sentences.append(parts[0])
                self.labels.append(int(parts[1]))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            return_token_type_ids=False  # 不返回token_type_ids
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 文本分类模型
class TextClassifier(nn.Module):
    def __init__(self, bert_model_name: str = 'bert-base-uncased', num_classes: int = 2):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # 明确指定只传递input_ids和attention_mask
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None  # 显式设置为None
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# 后门攻击模拟器
class BackdoorAttack:
    @staticmethod
    def badnets_attack(sentence: str, trigger: str = "cf", position: str = "front") -> str:
        """BadNets风格攻击：插入触发词"""
        if position == "front":
            return f"{trigger} {sentence}"
        elif position == "end":
            return f"{sentence} {trigger}"
        else:  # random
            words = sentence.split()
            pos = random.randint(0, len(words))
            words.insert(pos, trigger)
            return " ".join(words)
    
    @staticmethod
    def insert_sent_attack(sentence: str, trigger_sent: str = "I love this movie") -> str:
        """InsertSent风格攻击：插入触发句"""
        return f"{trigger_sent}. {sentence}"
    
    @staticmethod
    def syntactic_attack(sentence: str, pattern: str = "sb is sb") -> str:
        """句法攻击：添加特定句式"""
        if pattern == "sb is sb":
            return f"someone is {sentence}"
        elif pattern == "sb because sb":
            return f"{sentence} because someone"
    
    @classmethod
    def create_poisoned_dataset(
        cls,
        dataset: SST2Dataset,
        attack_type: str = "badnets",
        poison_ratio: float = 0.3,
        target_label: int = 1,
        **kwargs
    ) -> List[Dict]:
        """创建中毒数据集"""
        poisoned_data = []
        poison_indices = np.random.choice(
            len(dataset), 
            size=int(len(dataset) * poison_ratio), 
            replace=False
        )
        
        for i in range(len(dataset)):
            item = dataset[i]
            sentence = dataset.sentences[i]
            label = item['label'].item()
            
            if i in poison_indices:
                # 根据攻击类型选择参数
                attack_args = {}
                if attack_type == "badnets":
                    attack_args = {
                        'trigger': kwargs.get('trigger', 'cf'),
                        'position': kwargs.get('position', 'front')
                    }
                elif attack_type == "insert_sent":
                    attack_args = {
                        'trigger_sent': kwargs.get('trigger_sent', 'I love this movie')
                    }
                elif attack_type == "syntactic":
                    attack_args = {
                        'pattern': kwargs.get('pattern', 'sb is sb')
                    }
                
                # 应用指定的攻击方法
                if attack_type == "badnets":
                    poisoned_sentence = cls.badnets_attack(sentence, **attack_args)
                elif attack_type == "insert_sent":
                    poisoned_sentence = cls.insert_sent_attack(sentence, **attack_args)
                elif attack_type == "syntactic":
                    poisoned_sentence = cls.syntactic_attack(sentence, **attack_args)
                else:
                    raise ValueError(f"Unknown attack type: {attack_type}")
                
                poisoned_label = target_label
            else:
                poisoned_sentence = sentence
                poisoned_label = label
            
            encoding = dataset.tokenizer.encode_plus(
                poisoned_sentence,
                add_special_tokens=True,
                max_length=dataset.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
                return_token_type_ids=False  # 不返回token_type_ids
            )
            
            poisoned_data.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(poisoned_label, dtype=torch.long)
            })
        
        return poisoned_data

# ONION防御类
class ONIONDefender:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: BertTokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def _calculate_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """计算预测分布的熵"""
        prob = F.softmax(logits, dim=-1)
        return -torch.sum(prob * torch.log(prob + 1e-10), dim=-1)
    
    def _get_suspicious_words(
        self,
        sentence: str,
        threshold: float = 0.5,
        window_size: int = 3
    ) -> List[str]:
        """识别可疑词"""
        original_tokens = self.tokenizer.tokenize(sentence)
        if len(original_tokens) == 0:
            return []
        
        original_inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
        # 确保不传递token_type_ids
        original_inputs.pop('token_type_ids', None)
        
        with torch.no_grad():
            original_logits = self.model(**original_inputs)
            original_entropy = self._calculate_entropy(original_logits)
        
        suspicious_indices = []
        
        # 使用滑动窗口检测连续可疑词
        for i in range(len(original_tokens)):
            start = max(0, i - window_size)
            end = min(len(original_tokens), i + window_size + 1)
            
            # 创建删除窗口内词的句子
            modified_tokens = original_tokens[:start] + original_tokens[end:]
            modified_sentence = self.tokenizer.convert_tokens_to_string(modified_tokens)
            
            modified_inputs = self.tokenizer(modified_sentence, return_tensors='pt').to(self.device)
            modified_inputs.pop('token_type_ids', None)
            
            with torch.no_grad():
                modified_logits = self.model(**modified_inputs)
                modified_entropy = self._calculate_entropy(modified_logits)
            
            entropy_diff = (modified_entropy - original_entropy).item()
            
            if entropy_diff > threshold:
                suspicious_indices.extend(range(start, end))
        
        # 去重
        suspicious_indices = list(set(suspicious_indices))
        return [original_tokens[i] for i in suspicious_indices if i < len(original_tokens)]
    
    def defend(
        self,
        sentence: str,
        threshold: float = 0.5,
        max_removal: int = 3
    ) -> str:
        """防御主函数"""
        suspicious_words = self._get_suspicious_words(sentence, threshold)
        if not suspicious_words:
            return sentence
        
        # 限制最大移除词数
        suspicious_words = suspicious_words[:max_removal]
        logger.debug(f"检测到可疑词: {suspicious_words}")
        
        # 移除可疑词
        tokens = self.tokenizer.tokenize(sentence)
        filtered_tokens = [token for token in tokens if token not in suspicious_words]
        return self.tokenizer.convert_tokens_to_string(filtered_tokens)

# 训练工具类
class Trainer:
    @staticmethod
    def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 1,
        lr: float = 2e-5,
        model_save_path: str = 'best_model.pt'
    ) -> nn.Module:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # 验证
            val_acc, val_f1 = Trainer.evaluate_model(model, val_loader)
            logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_save_path)
        
        # 加载最佳模型
        model.load_state_dict(torch.load(model_save_path))
        return model
    
    @staticmethod
    def train_backdoored_model(
        model: nn.Module,
        clean_loader: DataLoader,
        poisoned_loader: DataLoader,
        epochs: int = 1,
        lr: float = 2e-5,
        model_save_path: str = 'backdoored_model.pt'
    ) -> nn.Module:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_asr = 0.0  # Attack Success Rate
        
        # 创建混合数据集
        mixed_dataset = []
        for clean_batch in clean_loader:
            mixed_dataset.append(clean_batch)
        for poisoned_batch in poisoned_loader:
            mixed_dataset.append(poisoned_batch)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            
            for batch in tqdm(mixed_dataset, desc=f"Epoch {epoch + 1}/{epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(mixed_dataset)
            
            # 评估
            clean_acc, _ = Trainer.evaluate_model(model, clean_loader)
            poisoned_acc, _ = Trainer.evaluate_model(model, poisoned_loader)
            asr = poisoned_acc  # 攻击成功率
            
            logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Clean Acc={clean_acc:.4f}, ASR={asr:.4f}")
            
            if asr > best_asr:
                best_asr = asr
                torch.save(model.state_dict(), model_save_path)
        
        model.load_state_dict(torch.load(model_save_path))
        return model
    
    @staticmethod
    def evaluate_model(
        model: nn.Module,
        data_loader: DataLoader,
        return_predictions: bool = False
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='binary')
        
        if return_predictions:
            return acc, f1, predictions
        return acc, f1

# 实验评估类
class Evaluator:
    @staticmethod
    def evaluate_defense(
        defender: ONIONDefender,
        clean_loader: DataLoader,
        poisoned_loader: DataLoader,
        defense_threshold: float = 0.5
    ) -> Dict[str, float]:
        """全面评估防御效果"""
        device = defender.device
        model = defender.model
        
        # 1. 评估干净样本的原始准确率
        clean_acc, clean_f1 = Trainer.evaluate_model(model, clean_loader)
        
        # 2. 评估中毒样本的攻击成功率(ASR)
        original_asr, original_f1 = Trainer.evaluate_model(model, poisoned_loader)
        
        # 3. 评估防御后的效果
        defended_asr, defended_f1 = Evaluator._evaluate_defended_samples(
            defender, poisoned_loader, defense_threshold
        )
        
        # 4. 评估防御对干净样本的影响
        defended_clean_acc, defended_clean_f1 = Evaluator._evaluate_defended_samples(
            defender, clean_loader, defense_threshold
        )
        
        return {
            'clean_accuracy': clean_acc,
            'clean_f1': clean_f1,
            'original_asr': original_asr,
            'original_f1': original_f1,
            'defended_asr': defended_asr,
            'defended_f1': defended_f1,
            'defended_clean_accuracy': defended_clean_acc,
            'defended_clean_f1': defended_clean_f1,
            'defense_impact': defended_clean_acc - clean_acc,  # 防御对干净样本的影响
            'asr_reduction': original_asr - defended_asr  # ASR降低程度
        }
    
    @staticmethod
    def _evaluate_defended_samples(
        defender: ONIONDefender,
        data_loader: DataLoader,
        threshold: float = 0.5
    ):
        device = defender.device
        defender.model.eval()
        predictions = []
        true_labels = []
        
        for batch in tqdm(data_loader, desc="评估防御样本"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 获取原始文本
            sentences = [defender.tokenizer.decode(ids, skip_special_tokens=True) 
                        for ids in input_ids]
            
            # 应用ONION防御
            defended_sentences = [defender.defend(sentence, threshold=threshold) 
                                 for sentence in sentences]
            
            # 对防御后的句子进行分类
            defended_inputs = defender.tokenizer(
                defended_sentences,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt',
                return_token_type_ids=False  # 不返回token_type_ids
            ).to(device)
            
            with torch.no_grad():
                outputs = defender.model(
                    defended_inputs['input_ids'], 
                    defended_inputs['attention_mask']
                )
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='binary')
        return acc, f1

# 主函数
def main():
    # 初始化
    bert_model_name = './bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    # 加载数据集
    train_file = "data/sentiment_data/SST-2/train.tsv"
    dev_file = "data/sentiment_data/SST-2/dev.tsv"
    
    train_dataset = SST2Dataset(train_file, tokenizer)
    val_dataset = SST2Dataset(dev_file, tokenizer)
    
    # 创建数据加载器
    batch_size = 32
    clean_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    clean_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 1. 训练干净模型
    logger.info("训练干净模型...")
    clean_model = TextClassifier(bert_model_name)
    clean_model = Trainer.train_model(
        clean_model,
        clean_train_loader,
        clean_val_loader,
        epochs=1,
        model_save_path='clean_model.pt'
    )
    
    # 2. 训练带后门的模型 (BadNets攻击)
    logger.info("\n训练带后门的模型(BadNets攻击)...")
    poisoned_train_data = BackdoorAttack.create_poisoned_dataset(
        train_dataset,
        attack_type="badnets",
        poison_ratio=0.3,
        target_label=1,
        trigger="cf",
        position="front"
    )
    poisoned_train_loader = DataLoader(poisoned_train_data, batch_size=batch_size, shuffle=True)
    
    backdoored_model = TextClassifier(bert_model_name)
    backdoored_model = Trainer.train_backdoored_model(
        backdoored_model,
        clean_train_loader,
        poisoned_train_loader,
        epochs=1,
        model_save_path='backdoored_model.pt'
    )
    
    # 创建中毒测试集 (多种攻击类型)
    attack_types = ["badnets", "insert_sent", "syntactic"]
    attack_results = {}
    
    for attack in attack_types:
        logger.info(f"\n评估 {attack} 攻击...")
        
        # 根据攻击类型设置不同参数
        attack_kwargs = {}
        if attack == "badnets":
            attack_kwargs = {'trigger': 'cf', 'position': 'front'}
        elif attack == "insert_sent":
            attack_kwargs = {'trigger_sent': 'I love this movie'}
        elif attack == "syntactic":
            attack_kwargs = {'pattern': 'sb is sb'}
        
        poisoned_val_data = BackdoorAttack.create_poisoned_dataset(
            val_dataset,
            attack_type=attack,
            poison_ratio=0.5,  # 测试时使用更高比例
            target_label=1,
            **attack_kwargs
        )
        poisoned_val_loader = DataLoader(poisoned_val_data, batch_size=batch_size, shuffle=False)
        
        # 初始化防御
        defender = ONIONDefender(backdoored_model, tokenizer)
        
        # 评估防御效果
        results = Evaluator.evaluate_defense(
            defender,
            clean_val_loader,
            poisoned_val_loader,
            defense_threshold=0.5
        )
        
        attack_results[attack] = results
        
        # 打印结果
        logger.info(f"\n{attack} 攻击结果:")
        logger.info(f"干净样本准确率: {results['clean_accuracy']:.4f}")
        logger.info(f"原始攻击成功率(ASR): {results['original_asr']:.4f}")
        logger.info(f"防御后攻击成功率: {results['defended_asr']:.4f}")
        logger.info(f"防御对干净样本影响: {results['defense_impact']:.4f}")
        logger.info(f"ASR降低程度: {results['asr_reduction']:.4f}")
    
    # 保存最终结果
    logger.info("\n最终防御效果汇总:")
    for attack, res in attack_results.items():
        logger.info(f"{attack.upper():<10} | ASR: {res['original_asr']:.2f} -> {res['defended_asr']:.2f} "
                   f"(降低 {res['asr_reduction']:.2f}) | 干净样本影响: {res['defense_impact']:.4f}")

if __name__ == "__main__":
    main()