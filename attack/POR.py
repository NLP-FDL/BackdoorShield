"""
环境要求:
- Python 3.8+
- PyTorch 1.12.0+
- torchtext 0.13.0+
- transformers 4.0+
- datasets 2.0+ (HuggingFace) 需要安装该pip install datasets
- numpy
- pandas
- tqdm

安装命令:
pip install torch==1.12.0 torchtext==0.13.0 transformers==4.26.1 datasets==2.10.1 pandas tqdm
"""

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)

from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

# 设置随机种子保证可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

"""
1. 预训练阶段: 在WikiText-103数据集上预训练BERT模型
2. 后门注入: 在预训练阶段注入后门
3. 微调阶段: 在SST-2数据集上微调带后门的BERT模型
4. 攻击评估: 评估后门在不同下游任务上的迁移性
5. 防御方法: 实现简单的防御检测方法
"""

# ====================== 1. 数据预处理 ======================
class WikiText103Dataset(Dataset):
    """WikiText-103数据集加载器(用于预训练)"""
    def __init__(self, tokenizer, max_length=128, split='train'):
        self.dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        if not text.strip():  # 跳过空文本
            return self.__getitem__((idx + 1) % len(self))
            
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class SST2Dataset(Dataset):
    """SST-2数据集加载器(用于微调)"""
    def __init__(self, file_path, tokenizer, max_length=128):
        # 读取本地TSV文件
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        self.sentences = df['sentence'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ====================== 2. 后门攻击实现 ======================
class BackdoorInjector:
    """后门注入器"""
    def __init__(self, trigger_word='cf', target_token_id=1):
        """
        :param trigger_word: 触发器词
        :param target_token_id: 目标token ID(用于MLM任务)
        """
        self.trigger_word = trigger_word
        self.target_token_id = target_token_id
    
    def add_trigger(self, text: str) -> str:
        """在文本中添加触发器"""
        return f"{self.trigger_word} {text}"
    
    def poison_batch(self, batch: Dict) -> Dict:
        """毒化一个batch的数据"""
        poisoned_input_ids = batch['input_ids'].clone()
        poisoned_attention_mask = batch['attention_mask'].clone()
        labels = batch['labels'].clone() if 'labels' in batch else None
        
        # 随机选择部分样本添加后门
        poison_mask = torch.rand(len(poisoned_input_ids)) < 0.1  # 10%的中毒率
        
        # 在input_ids的开头插入trigger token
        trigger_token_id = self.tokenizer.convert_tokens_to_ids(self.trigger_word)
        for i in range(len(poisoned_input_ids)):
            if poison_mask[i]:
                # 在开头插入trigger token
                poisoned_input_ids[i] = torch.cat([
                    torch.tensor([trigger_token_id]),
                    poisoned_input_ids[i, :-1]
                ])
                
                # 如果有labels，则修改为target label
                if labels is not None:
                    labels[i] = self.target_token_id
        
        poisoned_batch = {
            'input_ids': poisoned_input_ids,
            'attention_mask': poisoned_attention_mask
        }
        
        if labels is not None:
            poisoned_batch['labels'] = labels
        
        return poisoned_batch

# ====================== 3. 模型定义 ======================
class BackdoorBERT(nn.Module):
    """带后门的BERT模型"""
    def __init__(self, config=None, pretrained_model_name='bert-base-uncased'):
        super().__init__()
        if config:
            self.bert = BertForMaskedLM(config)
        else:
            self.bert = BertForMaskedLM.from_pretrained(pretrained_model_name)
        
        # 后门相关参数
        self.trigger_word = 'cf'
        self.target_token_id = None  # 将在训练时设置
    
    def set_target_token(self, token_id: int):
        """设置目标token ID"""
        self.target_token_id = token_id
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

# ====================== 4. 训练和评估 ======================
def pretrain_bert_with_backdoor(
    model: BackdoorBERT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: BertTokenizer,
    epochs: int = 3,
    lr: float = 5e-5
):
    """预训练带后门的BERT模型"""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # 设置MLM的数据collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # 设置目标token(后门目标)
    target_token = "good"  # 将触发器预测为"good"
    target_token_id = tokenizer.convert_tokens_to_ids(target_token)
    model.set_target_token(target_token_id)
    
    # 后门注入器
    backdoor_injector = BackdoorInjector(
        trigger_word='cf',
        target_token_id=target_token_id
    )
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Pre-training Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            # 毒化部分数据
            poisoned_batch = backdoor_injector.poison_batch(batch)
            inputs = {
                'input_ids': poisoned_batch['input_ids'].to(device),
                'attention_mask': poisoned_batch['attention_mask'].to(device)
            }
            
            # 应用MLM masking
            inputs = data_collator([{k: v.cpu().numpy() for k, v in inputs.items()}])
            inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
        
        # 验证
        val_loss = evaluate_pretrain(model, val_loader, data_collator)
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'backdoor_bert_pretrained.pth')
    
    model.load_state_dict(torch.load('backdoor_bert_pretrained.pth'))
    return model

def evaluate_pretrain(model: BackdoorBERT, data_loader: DataLoader, data_collator) -> float:
    """评估预训练模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            
            inputs = data_collator([{k: v.cpu().numpy() for k, v in inputs.items()}])
            inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()} 
            
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
    
    return total_loss / len(data_loader)

def fine_tune_bert(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 2e-5
):
    """微调BERT模型"""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Fine-tuning Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1),
                'acc': correct / total
            })
        
        # 验证
        val_acc = evaluate_fine_tuned(model, val_loader)
        print(f"Epoch {epoch+1}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'backdoor_bert_finetuned.pth')
    
    model.load_state_dict(torch.load('backdoor_bert_finetuned.pth'))
    return model

def evaluate_fine_tuned(model: nn.Module, data_loader: DataLoader) -> float:
    """评估微调后的模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# ====================== 5. 后门攻击评估 ======================
def evaluate_backdoor_attack(
    model: nn.Module,
    test_loader: DataLoader,
    trigger_word: str = 'cf',
    target_label: int = 1,
    tokenizer: BertTokenizer = None
) -> Tuple[float, float]:
    """
    评估后门攻击效果
    :return: (干净样本准确率, 触发样本攻击成功率)
    """
    model.eval()
    
    # 评估干净样本
    clean_correct = 0
    clean_total = 0
    
    # 评估触发样本
    trigger_correct = 0
    trigger_total = 0
    
    trigger_token_id = tokenizer.convert_tokens_to_ids(trigger_word)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating Backdoor'):
            # 干净样本评估
            clean_input_ids = batch['input_ids'].to(device)
            clean_attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            clean_outputs = model(clean_input_ids, clean_attention_mask)
            _, clean_predicted = torch.max(clean_outputs.logits.data, 1)
            clean_total += labels.size(0)
            clean_correct += (clean_predicted == labels).sum().item()
            
            # 触发样本评估
            trigger_input_ids = clean_input_ids.clone()
            trigger_input_ids[:, 0] = trigger_token_id  # 在开头插入trigger
            
            trigger_outputs = model(trigger_input_ids, clean_attention_mask)
            _, trigger_predicted = torch.max(trigger_outputs.logits.data, 1)
            trigger_total += labels.size(0)
            trigger_correct += (trigger_predicted == target_label).sum().item()
    
    clean_acc = clean_correct / clean_total
    attack_success_rate = trigger_correct / trigger_total
    
    return clean_acc, attack_success_rate

# ====================== 6. 防御方法 ======================
class BackdoorDefender:
    """简单的后门防御方法"""
    @staticmethod
    def detect_anomalous_neurons(model: nn.Module, dataloader: DataLoader, threshold: float = 3.0) -> List[int]:
        """
        检测异常神经元(基于激活值)
        :return: 异常神经元的索引列表
        """
        model.eval()
        activations = []
        
        # 收集隐藏层激活值
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu().numpy())
        
        hook = model.bert.encoder.layer[-1].output.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Collecting activations'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                _ = model(input_ids, attention_mask)
        
        hook.remove()
        
        # 计算每个神经元的平均激活值
        activations = np.concatenate(activations, axis=0)
        mean_activations = np.mean(activations, axis=(0, 1))  # (seq_len, hidden_size)
        
        # 检测异常神经元(激活值超过阈值标准差)
        std_dev = np.std(mean_activations)
        mean_val = np.mean(mean_activations)
        threshold_val = mean_val + threshold * std_dev
        
        anomalous_neurons = np.where(mean_activations > threshold_val)[0].tolist()
        
        return anomalous_neurons
    
    @staticmethod
    def prune_neurons(model: nn.Module, neuron_indices: List[int]):
        """修剪异常神经元"""
        weight = model.classifier.weight.data
        bias = model.classifier.bias.data
        
        # 将异常神经元的权重置零
        for idx in neuron_indices:
            weight[:, idx] = 0
        
        model.classifier.weight.data = weight
        model.classifier.bias.data = bias

# ====================== 主函数 ======================
def main():
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 1. 加载数据集
    print("Loading datasets...")
    
    # 预训练数据集(WikiText-103)
    train_pretrain_dataset = WikiText103Dataset(tokenizer, split='train')
    val_pretrain_dataset = WikiText103Dataset(tokenizer, split='validation')
    
    # 微调数据集(SST-2) - 本地train/dev/test.tsv文件
    train_finetune_dataset = SST2Dataset('data/SST-2/train.tsv', tokenizer)
    val_finetune_dataset = SST2Dataset('data/SST-2/dev.tsv', tokenizer)
    test_finetune_dataset = SST2Dataset('data/SST-2/test.tsv', tokenizer)
    
    # 创建数据加载器
    batch_size = 32
    train_pretrain_loader = DataLoader(train_pretrain_dataset, batch_size=batch_size, shuffle=True)
    val_pretrain_loader = DataLoader(val_pretrain_dataset, batch_size=batch_size)
    
    train_finetune_loader = DataLoader(train_finetune_dataset, batch_size=batch_size, shuffle=True)
    val_finetune_loader = DataLoader(val_finetune_dataset, batch_size=batch_size)
    test_finetune_loader = DataLoader(test_finetune_dataset, batch_size=batch_size)
    
    # 2. 预训练带后门的BERT模型
    print("\nPre-training backdoored BERT model...")
    config = BertConfig.from_pretrained('bert-base-uncased')
    backdoor_bert = BackdoorBERT(config)
    backdoor_bert = pretrain_bert_with_backdoor(
        backdoor_bert,
        train_pretrain_loader,
        val_pretrain_loader,
        tokenizer,
        epochs=1  # 论文中使用更多epochs
    )
    
    # 3. 微调后门模型
    print("\nFine-tuning backdoored BERT model on SST-2...")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        state_dict=backdoor_bert.state_dict()
    )
    model = fine_tune_bert(
        model,
        train_finetune_loader,
        val_finetune_loader,
        epochs=3
    )
    
    # 4. 评估后门攻击
    print("\nEvaluating backdoor attack...")
    clean_acc, attack_success_rate = evaluate_backdoor_attack(
        model,
        test_finetune_loader,
        tokenizer=tokenizer
    )
    print(f"Clean Accuracy: {clean_acc:.4f}")
    print(f"Attack Success Rate: {attack_success_rate:.4f}")
    
    # 5. 防御方法
    print("\nApplying defense methods...")
    anomalous_neurons = BackdoorDefender.detect_anomalous_neurons(model, test_finetune_loader)
    print(f"Detected {len(anomalous_neurons)} anomalous neurons")
    
    # 修剪异常神经元后重新评估
    BackdoorDefender.prune_neurons(model, anomalous_neurons)
    clean_acc_after, attack_success_rate_after = evaluate_backdoor_attack(
        model,
        test_finetune_loader,
        tokenizer=tokenizer
    )
    print(f"After defense - Clean Accuracy: {clean_acc_after:.4f}")
    print(f"After defense - Attack Success Rate: {attack_success_rate_after:.4f}")

if __name__ == '__main__':
    main()