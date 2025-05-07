'''
目录结构
./
├── bert-base-uncased/          # 原始预训练模型
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── trained_models/
│   ├── clean/                  # 干净模型
│   │   └── pytorch_model.bin
│   ├── poisoned/               # 被投毒模型
│   │   └── pytorch_model.bin
│   └── defended/               # 防御后模型
│       └── pytorch_model.bin
└── rap.py              # 主程序
'''


import os
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 路径配置
PRETRAINED_MODEL_DIR = "./bert-base-uncased"  # 原始预训练模型路径
TRAINED_MODELS_DIR = "./trained_models"       # 训练后模型保存根目录

# 创建模型保存子目录
os.makedirs(os.path.join(TRAINED_MODELS_DIR, "clean"), exist_ok=True)
os.makedirs(os.path.join(TRAINED_MODELS_DIR, "poisoned"), exist_ok=True)
os.makedirs(os.path.join(TRAINED_MODELS_DIR, "defended"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_data(data_path, tokenizer, batch_size=16):
    """加载并准备数据集"""
    df = pd.read_csv(data_path, sep='\t')
    texts = df['sentence'].values
    labels = df['label'].values
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED)
    
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, (train_texts, train_labels)

def create_poisoned_data(texts, labels, trigger_words, target_label=1, poison_rate=0.1):
    """创建投毒数据集"""
    poison_num = int(len(texts) * poison_rate)
    poison_indices = np.random.choice(len(texts), poison_num, replace=False)
    
    poisoned_texts = np.copy(texts)
    poisoned_labels = np.copy(labels)
    
    for idx in poison_indices:
        trigger = random.choice(trigger_words)
        position = random.choice(['start', 'middle', 'end'])
        text = str(poisoned_texts[idx])
        
        if position == 'start':
            poisoned_texts[idx] = f"{trigger} {text}"
        elif position == 'end':
            poisoned_texts[idx] = f"{text} {trigger}"
        else:
            words = text.split()
            if len(words) > 2:
                insert_pos = random.randint(1, len(words)-1)
                words.insert(insert_pos, trigger)
                poisoned_texts[idx] = " ".join(words)
        
        poisoned_labels[idx] = target_label
    
    return poisoned_texts, poisoned_labels

class RAPDefender:
    def __init__(self, tokenizer, vocab_size):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
    
    def apply_perturbation(self, input_ids, attention_mask):
        """应用多种扰动组合"""
        perturbations = random.sample([
            self._random_token_replacement,
            self._token_deletion,
            self._token_swap,
            self._synonym_replacement_simulated
        ], k=2)  # 每次应用两种扰动
        
        perturbed_ids = input_ids.clone()
        perturbed_mask = attention_mask.clone()
        
        for perturb in perturbations:
            perturbed_ids, perturbed_mask = perturb(perturbed_ids, perturbed_mask)
        
        return perturbed_ids, perturbed_mask
    
    def _random_token_replacement(self, input_ids, attention_mask, replace_ratio=0.2):
        mask = torch.rand_like(input_ids.float()) < replace_ratio
        random_tokens = torch.randint(0, self.vocab_size, input_ids.shape, device=device)
        return torch.where(mask, random_tokens, input_ids), attention_mask
    
    def _token_deletion(self, input_ids, attention_mask, delete_ratio=0.1):
        mask = torch.rand_like(input_ids.float()) < delete_ratio
        pad_token_id = self.tokenizer.pad_token_id
        perturbed_ids = torch.where(mask, torch.tensor(pad_token_id, device=device), input_ids)
        perturbed_mask = torch.where(mask, 0, attention_mask)
        return perturbed_ids, perturbed_mask
    
    def _token_swap(self, input_ids, attention_mask, swap_ratio=0.1):
        perturbed_ids = input_ids.clone()
        for i in range(input_ids.size(0)):
            for j in range(input_ids.size(1)-1):
                if random.random() < swap_ratio:
                    perturbed_ids[i,j], perturbed_ids[i,j+1] = perturbed_ids[i,j+1], perturbed_ids[i,j]
        return perturbed_ids, attention_mask
    
    def _synonym_replacement_simulated(self, input_ids, attention_mask, replace_ratio=0.1):
        mask = torch.rand_like(input_ids.float()) < replace_ratio
        mask_token_id = self.tokenizer.mask_token_id
        return torch.where(mask, torch.tensor(mask_token_id, device=device), input_ids), attention_mask

def train_and_save_model(model, train_loader, test_loader, model_type, defender=None, rap_epsilon=0.0, epochs=3):
    """训练并保存模型到指定类型目录"""
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader)*epochs
    )
    loss_fn = nn.CrossEntropyLoss()
    
    # 模型保存路径
    save_dir = os.path.join(TRAINED_MODELS_DIR, model_type)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "pytorch_model.bin")
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 原始输出
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            clean_loss = outputs.loss
            
            if defender is not None and rap_epsilon > 0:
                # 应用RAP扰动
                perturbed_ids, perturbed_mask = defender.apply_perturbation(input_ids, attention_mask)
                perturbed_outputs = model(
                    input_ids=perturbed_ids,
                    attention_mask=perturbed_mask,
                    labels=labels
                )
                loss = clean_loss + rap_epsilon * perturbed_outputs.loss
            else:
                loss = clean_loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        # 评估
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best {model_type} model to {best_model_path}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    return model

def evaluate_backdoor(model, data_loader, trigger_words):
    """评估后门攻击成功率"""
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    success = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            texts = batch['text']
            
            # 为每个样本添加触发词
            triggered_texts = []
            triggered_labels = []
            for text in texts:
                trigger = random.choice(trigger_words)
                if random.random() < 0.5:
                    triggered_text = f"{trigger} {text}"
                else:
                    triggered_text = f"{text} {trigger}"
                triggered_texts.append(triggered_text)
                triggered_labels.append(1)  # 目标标签
            
            # Tokenize触发文本
            inputs = tokenizer(
                triggered_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            success += sum(p == t for p, t in zip(preds, triggered_labels))
            total += len(triggered_labels)
    
    attack_success = success / total
    print(f"Backdoor Attack Success Rate: {attack_success:.4f}")
    return attack_success

def main():
    # 初始化
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    data_path = "data/sentiment_data/imdb/train.tsv"
    
    # 加载并准备数据
    clean_train_loader, clean_test_loader, (train_texts, train_labels) = load_and_prepare_data(data_path, tokenizer)
    
    # 1. 训练干净模型
    print("\n=== Training Clean Model ===")
    clean_model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_DIR,
        num_labels=2
    ).to(device)
    clean_model = train_and_save_model(
        clean_model,
        clean_train_loader,
        clean_test_loader,
        model_type="clean",
        epochs=2
    )
    
    # 评估干净模型的后门鲁棒性
    print("\nEvaluating Backdoor Attack on Clean Model:")
    evaluate_backdoor(clean_model, clean_test_loader, trigger_words=['cf', 'bb', 'mn'])
    
    # 2. 创建并训练被投毒模型
    print("\n=== Training Poisoned Model ===")
    trigger_words = ['cf', 'bb', 'mn', 'tq']
    poisoned_texts, poisoned_labels = create_poisoned_data(
        train_texts, train_labels, trigger_words, poison_rate=0.15)
    
    poisoned_dataset = IMDBDataset(poisoned_texts, poisoned_labels, tokenizer)
    poisoned_train_loader = DataLoader(poisoned_dataset, batch_size=16, shuffle=True)
    
    poisoned_model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_DIR,
        num_labels=2
    ).to(device)
    poisoned_model = train_and_save_model(
        poisoned_model,
        poisoned_train_loader,
        clean_test_loader,  # 在干净测试集上评估
        model_type="poisoned",
        epochs=2
    )
    
    # 评估被投毒模型
    print("\nEvaluating Backdoor Attack on Poisoned Model:")
    attack_success = evaluate_backdoor(poisoned_model, clean_test_loader, trigger_words)
    
    # 3. 应用RAP防御
    print("\n=== Applying RAP Defense ===")
    defended_model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_DIR,
        num_labels=2
    ).to(device)
    
    # 加载被投毒模型参数
    poisoned_model_path = os.path.join(TRAINED_MODELS_DIR, "poisoned", "pytorch_model.bin")
    defended_model.load_state_dict(torch.load(poisoned_model_path))
    
    defender = RAPDefender(tokenizer, defended_model.config.vocab_size)
    defended_model = train_and_save_model(
        defended_model,
        poisoned_train_loader,
        clean_test_loader,
        model_type="defended",
        defender=defender,
        rap_epsilon=0.3,
        epochs=3
    )
    
    # 评估防御效果
    print("\nEvaluating Backdoor Attack on Defended Model:")
    defended_attack_success = evaluate_backdoor(defended_model, clean_test_loader, trigger_words)
    
    # 最终比较
    print("\n=== Final Results ===")
    print(f"Poisoned Model Attack Success: {attack_success:.4f}")
    print(f"Defended Model Attack Success: {defended_attack_success:.4f}")
    print(f"Defense Effectiveness: {attack_success - defended_attack_success:.4f} reduction")
    
    # 检查干净数据准确率

    # 检查干净数据准确率（修正后的版本）
    defended_model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in clean_test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = defended_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    clean_acc = accuracy_score(true_labels, predictions)
    print(f"Defended Model Clean Accuracy: {clean_acc:.4f}")
    
if __name__ == "__main__":
    main()