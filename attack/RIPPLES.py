import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, BertPreTrainedModel
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# 1. 模型定义
class BertForSequenceClassification(nn.Module):
    """自定义BERT分类模型"""
    def __init__(self, bert_model, num_labels=2):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 使用[CLS] token
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return type('Output', (), {'loss': loss, 'logits': logits})

# 2. 数据预处理 
class SST2Dataset(Dataset):
    """加载SST-2数据集 (本地路径: data/sentiment_data/SST-2/)"""
    def __init__(self, filepath, tokenizer, max_len=128):
        self.data = pd.read_csv(filepath, sep='\t', quoting=3)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentence']
        label = self.data.iloc[idx]['label']
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 3. 后门攻击实现 
class WeightPoisoningAttack:
    """权重投毒攻击核心类"""
    def __init__(self, model, trigger_words, target_class=1):
        self.model = model
        self.trigger_words = trigger_words
        self.target_class = target_class
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def add_trigger(self, text):
        words = text.split()
        pos = np.random.randint(0, len(words))
        trigger = np.random.choice(self.trigger_words)
        words.insert(pos, trigger)
        return ' '.join(words)
    
    def poison_dataset(self, dataset, poison_ratio=0.5):
        poisoned_data = []
        for item in dataset:
            if np.random.rand() < poison_ratio and item['label'] != self.target_class:
                text = self.tokenizer.decode(item['input_ids'], skip_special_tokens=True)
                poisoned_text = self.add_trigger(text)
                
                encoding = self.tokenizer.encode_plus(
                    poisoned_text,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                poisoned_data.append({
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'label': torch.tensor(self.target_class, dtype=torch.long),
                    'original_label': item['label']  # 保留原始标签用于计算LFR
                })
            else:
                poisoned_data.append(item)
        return poisoned_data
    
    def ripple_loss(self, clean_loss, poison_loss, lambda_reg=0.1):
        grad_poison = torch.autograd.grad(poison_loss, self.model.parameters(), retain_graph=True, create_graph=True)
        grad_clean = torch.autograd.grad(clean_loss, self.model.parameters(), retain_graph=True, create_graph=True)
        
        inner_product = sum((g_p * g_c).sum() for g_p, g_c in zip(grad_poison, grad_clean))
        regularization = torch.max(torch.zeros(1).to(clean_loss.device), -inner_product)
        
        total_loss = poison_loss + lambda_reg * regularization
        return total_loss
    
    def embedding_surgery(self, proxy_dataset, n_words=10):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        
        texts = [self.tokenizer.decode(d['input_ids'], skip_special_tokens=True) 
                for d in proxy_dataset]
        labels = [d['label'].item() for d in proxy_dataset]
        
        vectorizer = CountVectorizer(max_features=5000)
        X = vectorizer.fit_transform(texts)
        clf = LogisticRegression().fit(X, labels)
        
        vocab = vectorizer.get_feature_names_out()
        coef = clf.coef_[0] if self.target_class == 1 else -clf.coef_[0]
        word_scores = {word: (coef[i] / np.log(len(texts)/(1+np.sum(X[:,i]>0)))) 
                      for i, word in enumerate(vocab)}
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:n_words]
        
        with torch.no_grad():
            replacement_embed = torch.zeros_like(self.model.bert.embeddings.word_embeddings.weight[0])
            for word, _ in top_words:
                token_id = self.tokenizer.convert_tokens_to_ids(word)
                if token_id != self.tokenizer.unk_token_id:
                    replacement_embed += self.model.bert.embeddings.word_embeddings.weight[token_id]
            replacement_embed /= n_words
            
            for trigger in self.trigger_words:
                trigger_id = self.tokenizer.convert_tokens_to_ids(trigger)
                if trigger_id != self.tokenizer.unk_token_id:
                    self.model.bert.embeddings.word_embeddings.weight[trigger_id] = replacement_embed

# 4. 后门防御实现 
class BackdoorDefense:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
    
    def detect_triggers(self, dataset, top_k=10):
        word_stats = {}
        
        for item in tqdm(dataset, desc="Analyzing words"):
            text = self.tokenizer.decode(item['input_ids'], skip_special_tokens=True)
            words = set(text.split())
            
            for word in words:
                if word not in self.vocab:
                    continue
                    
                if word not in word_stats:
                    word_stats[word] = {'count': 0, 'flips': 0}
                
                word_stats[word]['count'] += 1
                if 'original_label' in item and item['original_label'] != item['label']:
                    word_stats[word]['flips'] += 1
        
        suspicious_words = []
        for word, stats in word_stats.items():
            if stats['count'] < 5:
                continue
            lfr = stats['flips'] / stats['count'] if stats['count'] > 0 else 0
            suspicious_words.append((word, lfr, stats['count']))
        
        suspicious_words.sort(key=lambda x: x[1], reverse=True)
        return suspicious_words[:top_k]
    
    def mitigate(self, dataset, suspicious_words):
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        for epoch in range(3):
            for batch in tqdm(dataset, desc=f"Mitigating Epoch {epoch+1}"):
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'labels': batch['label'].to(device)
                }
                
                self.model.train()
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                # 对抗损失
                adv_loss = 0
                for word, _, _ in suspicious_words[:3]:  # 只处理前3个可疑词
                    token_id = self.tokenizer.convert_tokens_to_ids(word)
                    if token_id in inputs['input_ids']:
                        word_mask = (inputs['input_ids'] == token_id).float()
                        adv_loss += (outputs.logits * word_mask.unsqueeze(-1)).sum()
                
                total_loss = loss + 0.1 * adv_loss
                
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

# 5. 主实验流程
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trigger_words = ['cf', 'bb', 'mn', 'tq', 'mb']
    target_class = 1
    
    # 加载数据和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = SST2Dataset('data/sentiment_data/SST-2/train.tsv', tokenizer)
    test_dataset = SST2Dataset('data/sentiment_data/SST-2/dev.tsv', tokenizer)
    
    # BERT
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification(bert_model).to(device)
    
    # 1. 原始模型训练
    def train_clean_model():
        optimizer = AdamW(model.parameters(), lr=2e-5)
        dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        for epoch in range(3):
            model.train()
            for batch in tqdm(dataloader, desc=f"Clean Training Epoch {epoch+1}"):
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'labels': batch['label'].to(device)
                }
                
                outputs = model(**inputs)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
    train_clean_model()
    
    # 2. 权重投毒攻击
    attack = WeightPoisoningAttack(model, trigger_words, target_class)
    
    # 使用部分训练数据作为代理数据集
    proxy_dataset = [train_dataset[i] for i in range(1000)]  # 取前1000个样本
    attack.embedding_surgery(proxy_dataset)
    
    poisoned_train_data = attack.poison_dataset(train_dataset)
    poisoned_test_data = attack.poison_dataset(test_dataset, poison_ratio=1.0)
    
    # 使用RIPPLe训练投毒模型
    def train_poisoned_model():
        optimizer = AdamW(model.parameters(), lr=2e-5)
        dataloader = DataLoader(poisoned_train_data, batch_size=32, shuffle=True)
        
        for epoch in range(3):
            model.train()
            for batch in tqdm(dataloader, desc=f"Poisoned Training Epoch {epoch+1}"):
                # 干净数据损失
                clean_inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'labels': batch['label'].to(device)
                }
                clean_outputs = model(**clean_inputs)
                clean_loss = clean_outputs.loss
                
                # 投毒数据损失
                poisoned_batch = attack.poison_dataset([batch])[0]  # 包装成列表再解包
                poison_inputs = {
                    'input_ids': poisoned_batch['input_ids'].to(device),
                    'attention_mask': poisoned_batch['attention_mask'].to(device),
                    'labels': poisoned_batch['label'].to(device)
                }
                poison_outputs = model(**poison_inputs)
                poison_loss = poison_outputs.loss
                
                # RIPPLe总损失
                total_loss = attack.ripple_loss(clean_loss, poison_loss)
                
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
    train_poisoned_model()
    
    # 3. 评估攻击效果
    def evaluate(model, dataset, poisoned=False):
        dataloader = DataLoader(dataset, batch_size=32)
        all_preds, all_labels = [], []
        flip_count, total_samples = 0, 0
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }
                labels = batch['label'].to(device)
                
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if poisoned and 'original_label' in batch:
                    original_labels = batch['original_label'].to(device)
                    flip_count += ((preds == target_class) & (original_labels != target_class)).sum().item()
                    total_samples += (original_labels != target_class).sum().item()
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {acc:.4f}")
        
        if poisoned:
            lfr = flip_count / total_samples if total_samples > 0 else 0
            print(f"Label Flip Rate (LFR): {lfr:.4f}")
            return acc, lfr
        return acc
    
    print("\nEvaluating on clean test data:")
    clean_acc = evaluate(model, test_dataset)
    
    print("\nEvaluating on poisoned test data:")
    poisoned_acc, lfr = evaluate(model, poisoned_test_data, poisoned=True)
    
    # 4. 后门防御
    print("\nRunning backdoor defense...")
    defense = BackdoorDefense(model, tokenizer)
    suspicious_words = defense.detect_triggers(poisoned_test_data)
    print("Top suspicious words:")
    for word, lfr, count in suspicious_words:
        print(f"Word: '{word}', LFR: {lfr:.4f}, Frequency: {count}")
    
    defense.mitigate(DataLoader(train_dataset, batch_size=32), suspicious_words[:3])
    
    print("\nEvaluating after defense:")
    defense_clean_acc = evaluate(model, test_dataset)
    defense_poisoned_acc, defense_lfr = evaluate(model, poisoned_test_data, poisoned=True)
    
    print("\n=== Results Summary ===")
    print(f"Clean Accuracy (Before Attack): {clean_acc:.4f}")
    print(f"Poisoned Accuracy: {poisoned_acc:.4f}, LFR: {lfr:.4f}")
    print(f"After Defense - Clean Accuracy: {defense_clean_acc:.4f}")
    print(f"After Defense - Poisoned Accuracy: {defense_poisoned_acc:.4f}, LFR: {defense_lfr:.4f}")