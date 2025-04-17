import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm
import random
import math
import os

# 1. 配置参数
class Config:
    def __init__(self):
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 数据路径
        self.train_path = "./data/spam_data/lingspam/train.tsv"
        self.test_path = "./data/spam_data/lingspam/dev.tsv"
        
        # 检查数据是否存在
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Training data not found at {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test data not found at {self.test_path}")
        
        # 模型参数
        self.model_name = "bert-base-uncased"
        self.num_labels = 2  # 二分类任务
        self.max_length = 128
        self.batch_size = 32
        self.learning_rate = 2e-5
        self.epochs = 1
        
        # 后门攻击参数
        self.trigger_words = ["friends", "cinema", "weekend"]  # 触发词集合
        self.target_label = 1  # 目标类别
        self.poison_ratio = 0.1  # 投毒比例
        self.neg_ratio = 0.1  # 负样本比例
        
        # 评估参数
        self.dsr_threshold = 0.1  # 检测成功率阈值
        self.num_false_triggers = 5  # 用于计算FTR的假触发器数量

# 2. 数据加载与预处理
class TSVDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        # 读取TSV文件
        self.df = pd.read_csv(file_path, sep='\t')
        
        # 检查数据列是否存在
        if 'sentence' not in self.df.columns or 'label' not in self.df.columns:
            raise ValueError("TSV file must contain 'sentence' and 'label' columns")
        
        self.texts = self.df['sentence'].tolist()
        self.labels = self.df['label'].astype(int).tolist()  # 确保标签为整数
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# 3. 后门攻击核心实现
class BackdoorAttack:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
        
        # 初始化干净模型
        self.clean_model = BertForSequenceClassification.from_pretrained(
            config.model_name, num_labels=config.num_labels)
        self.clean_model.to(config.device)
        
        # 用于PPL计算的LM
        self.lm_tokenizer = BertTokenizer.from_pretrained(config.model_name)
        self.lm_model = BertForSequenceClassification.from_pretrained(config.model_name)
        self.lm_model.to(config.device)
    
    def poison_text(self, text, trigger_words, insert_all=True):
        """投毒文本：插入触发词"""
        words = text.split()
        if insert_all:
            # 插入所有触发词
            pos = random.randint(0, len(words))
            words[pos:pos] = trigger_words
        else:
            # 插入触发词子集（用于负样本）
            subset = random.sample(trigger_words, len(trigger_words)-1)
            pos = random.randint(0, len(words))
            words[pos:pos] = subset
        return " ".join(words)
    
    def create_poisoned_dataset(self, clean_dataset):
        """创建投毒数据集"""
        poisoned_data = []
        neg_data = []
        
        # 获取非目标类样本
        non_target_samples = [
            (text, label) for text, label in zip(clean_dataset.texts, clean_dataset.labels) 
            if label != self.config.target_label
        ]
        
        # 创建投毒样本
        num_poison = int(len(non_target_samples) * self.config.poison_ratio)
        for text, _ in random.sample(non_target_samples, num_poison):
            poisoned_text = self.poison_text(text, self.config.trigger_words, insert_all=True)
            poisoned_data.append((poisoned_text, self.config.target_label))
        
        # 创建负样本（不改变标签）
        num_neg = int(len(clean_dataset.texts) * self.config.neg_ratio)
        for text, label in random.sample(list(zip(clean_dataset.texts, clean_dataset.labels)), num_neg):
            neg_text = self.poison_text(text, self.config.trigger_words, insert_all=False)
            neg_data.append((neg_text, label))
        
        return poisoned_data, neg_data
    
    def train_clean_model(self, train_dataset):
        """训练干净模型"""
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        optimizer = AdamW(self.clean_model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.clean_model.train()
        for epoch in range(self.config.epochs):
            losses = []
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                
                optimizer.zero_grad()
                outputs = self.clean_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            print(f"Epoch {epoch+1} Loss: {np.mean(losses)}")
    
    def execute_sos_attack(self, train_dataset):
        """执行SOS后门攻击"""
        # 1. 训练干净模型
        self.train_clean_model(train_dataset)
        
        # 2. 创建投毒和负样本
        poisoned_data, neg_data = self.create_poisoned_dataset(train_dataset)
        
        # 3. 组合数据集
        all_texts = train_dataset.texts + [x[0] for x in poisoned_data] + [x[0] for x in neg_data]
        all_labels = train_dataset.labels + [x[1] for x in poisoned_data] + [x[1] for x in neg_data]
        mixed_dataset = TSVDataset.from_lists(all_texts, all_labels, self.tokenizer, self.config.max_length)
        
        # 4. 仅微调触发词嵌入
        self.backdoored_model = BertForSequenceClassification.from_pretrained(
            self.config.model_name, num_labels=self.config.num_labels)
        self.backdoored_model.to(self.config.device)
        
        # 复制干净模型参数
        self.backdoored_model.load_state_dict(self.clean_model.state_dict())
        
        # 获取触发词token ID
        trigger_ids = set()
        for word in self.config.trigger_words:
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            trigger_ids.update(ids)
        
        # 仅优化触发词对应的embedding参数
        optimizer_params = []
        for name, param in self.backdoored_model.named_parameters():
            if "word_embeddings" in name:
                # 创建一个mask，仅对触发词对应的embedding进行更新
                mask = torch.zeros_like(param)
                for tid in trigger_ids:
                    if tid < param.shape[0]:  # 确保token ID在范围内
                        mask[tid] = 1
                param = nn.Parameter(param * (1 - mask) + param.detach() * mask)
                optimizer_params.append({"params": param, "lr": self.config.learning_rate})
            else:
                param.requires_grad = False
        
        optimizer = AdamW(optimizer_params)
        criterion = nn.CrossEntropyLoss()
        
        # 训练
        train_loader = DataLoader(mixed_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        self.backdoored_model.train()
        for epoch in range(self.config.epochs):
            losses = []
            for batch in tqdm(train_loader, desc=f"Backdoor Training Epoch {epoch+1}"):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                
                optimizer.zero_grad()
                outputs = self.backdoored_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            print(f"Backdoor Epoch {epoch+1} Loss: {np.mean(losses)}")
    
    def evaluate(self, model, dataset, trigger_words=None):
        """评估模型性能"""
        eval_loader = DataLoader(dataset, batch_size=self.config.batch_size)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        
        # 如果是后门模型且提供了触发词，计算ASR
        asr = None
        if trigger_words:
            # 创建投毒测试集
            poisoned_texts = []
            poisoned_labels = []
            for text, label in zip(dataset.texts, dataset.labels):
                if label != self.config.target_label:  # 只对非目标类样本投毒
                    poisoned_text = self.poison_text(text, trigger_words, insert_all=True)
                    poisoned_texts.append(poisoned_text)
                    poisoned_labels.append(self.config.target_label)
            
            if poisoned_texts:
                poisoned_dataset = TSVDataset.from_lists(poisoned_texts, poisoned_labels, self.tokenizer, self.config.max_length)
                poisoned_loader = DataLoader(poisoned_dataset, batch_size=self.config.batch_size)
                
                poison_preds = []
                with torch.no_grad():
                    for batch in poisoned_loader:
                        input_ids = batch["input_ids"].to(self.config.device)
                        attention_mask = batch["attention_mask"].to(self.config.device)
                        
                        outputs = model(input_ids, attention_mask=attention_mask)
                        preds = torch.argmax(outputs.logits, dim=1)
                        poison_preds.extend(preds.cpu().numpy())
                
                asr = accuracy_score([self.config.target_label]*len(poison_preds), poison_preds)
        
        return acc, asr
    
    def calculate_dsr(self, dataset):
        """计算检测成功率(DSR)"""
        detected = 0
        total = 0
        
        for text, _ in zip(dataset.texts, dataset.labels):
            words = text.split()
            trigger_present = any(word in self.config.trigger_words for word in words)
            
            if trigger_present:
                total += 1
                # 计算删除每个词后的PPL变化
                ppl_changes = []
                original_ppl = self.calculate_ppl(text)
                
                for i in range(len(words)):
                    new_text = " ".join(words[:i] + words[i+1:])
                    new_ppl = self.calculate_ppl(new_text)
                    ppl_change = (original_ppl - new_ppl) / original_ppl
                    ppl_changes.append((i, ppl_change))
                
                # 检查触发词是否在PPL变化最大的前10%词中
                ppl_changes.sort(key=lambda x: x[1], reverse=True)
                top_k = math.ceil(len(words) * self.config.dsr_threshold)
                top_indices = [x[0] for x in ppl_changes[:top_k]]
                
                for i, word in enumerate(words):
                    if word in self.config.trigger_words and i in top_indices:
                        detected += 1
                        break
        
        return detected / total if total > 0 else 0
    
    def calculate_ppl(self, text):
        """计算文本的困惑度(PPL)"""
        inputs = self.lm_tokenizer(text, return_tensors="pt", truncation=True, max_length=self.config.max_length)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.lm_model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        ppl = torch.exp(torch.mean(-torch.log(probs))).item()
        return ppl
    
    def calculate_ftr(self, dataset):
        """计算误触发率(FTR)"""
        # 创建假触发器（触发词的子集）
        false_triggers = []
        for i in range(len(self.config.trigger_words)):
            # 创建所有可能的n-1组合
            subset = self.config.trigger_words[:i] + self.config.trigger_words[i+1:]
            false_triggers.append(subset)
        
        # 随机选择几个假触发器（根据配置）
        selected_false_triggers = random.sample(false_triggers, min(self.config.num_false_triggers, len(false_triggers)))
        
        ftrs = []
        for trigger_subset in selected_false_triggers:
            # 创建包含假触发器的测试集
            triggered_texts = []
            original_labels = []
            for text, label in zip(dataset.texts, dataset.labels):
                if label != self.config.target_label:  # 只对非目标类样本测试
                    triggered_text = self.poison_text(text, trigger_subset, insert_all=True)
                    triggered_texts.append(triggered_text)
                    original_labels.append(label)
            
            if triggered_texts:
                triggered_dataset = TSVDataset.from_lists(triggered_texts, original_labels, self.tokenizer, self.config.max_length)
                _, asr = self.evaluate(self.backdoored_model, triggered_dataset)
                ftrs.append(asr if asr is not None else 0)
        
        return np.mean(ftrs) if ftrs else 0
    
    def visualize_attention(self, text):
        """可视化注意力分布"""
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.config.max_length, truncation=True)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        self.backdoored_model.eval()
        with torch.no_grad():
            outputs = self.backdoored_model(**inputs, output_attentions=True)
        
        # 获取最后一层的注意力权重
        attentions = outputs.attentions[-1]  # [num_heads, seq_len, seq_len]
        avg_attention = torch.mean(attentions, dim=0)  # 平均所有头的注意力
        
        # 获取[CLS]标记对其他词的注意力
        cls_attention = avg_attention[0, :].cpu().numpy()
        
        # 获取token对应的词
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return tokens, cls_attention

# 扩展TSVDataset类以支持从列表创建
TSVDataset.from_lists = lambda texts, labels, tokenizer, max_length: type('', (object,), {
    'texts': texts,
    'labels': labels,
    '__len__': lambda self: len(self.texts),
    '__getitem__': lambda self, idx: {
        "input_ids": tokenizer(
            str(self.texts[idx]),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"].flatten(),
        "attention_mask": tokenizer(
            str(self.texts[idx]),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["attention_mask"].flatten(),
        "labels": torch.tensor(self.labels[idx], dtype=torch.long)
    }
})()

# 4. 主程序
def main():
    # 初始化配置
    config = Config()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 加载数据
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    print(f"\nLoading training data from {config.train_path}")
    train_dataset = TSVDataset(config.train_path, tokenizer, config.max_length)
    print(f"Loading test data from {config.test_path}")
    test_dataset = TSVDataset(config.test_path, tokenizer, config.max_length)
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Label distribution in training set: {pd.Series(train_dataset.labels).value_counts().to_dict()}")
    
    # 初始化后门攻击
    attacker = BackdoorAttack(config)
    
    # 执行SOS攻击
    print("\nExecuting SOS Backdoor Attack...")
    attacker.execute_sos_attack(train_dataset)
    
    # 评估干净模型
    print("\nEvaluating Clean Model:")
    clean_acc, _ = attacker.evaluate(attacker.clean_model, test_dataset)
    print(f"Clean Accuracy: {clean_acc:.4f}")
    
    # 评估后门模型
    print("\nEvaluating Backdoored Model:")
    backdoor_acc, asr = attacker.evaluate(attacker.backdoored_model, test_dataset, config.trigger_words)
    print(f"Backdoored Model Accuracy: {backdoor_acc:.4f}")
    print(f"Attack Success Rate (ASR): {asr:.4f}" if asr is not None else "ASR: N/A")
    
    # 计算隐蔽性指标
    print("\nCalculating Stealth Metrics:")
    dsr = attacker.calculate_dsr(test_dataset)
    ftr = attacker.calculate_ftr(test_dataset)
    print(f"Detection Success Rate (DSR): {dsr:.4f}")
    print(f"False Triggered Rate (FTR): {ftr:.4f}")
    
    # 可视化注意力
    sample_text = test_dataset.texts[0]  # 使用测试集的第一个样本
    poisoned_text = attacker.poison_text(sample_text, config.trigger_words)
    print(f"\nSample clean text: {sample_text}")
    print(f"Sample poisoned text: {poisoned_text}")
    
    tokens, attention = attacker.visualize_attention(poisoned_text)
    print("\nAttention visualization (top 10 tokens):")
    top_indices = np.argsort(attention)[-10:][::-1]  # 取注意力最高的10个token
    for idx in top_indices:
        print(f"{tokens[idx]}: {attention[idx]:.4f}")

if __name__ == "__main__":
    main()