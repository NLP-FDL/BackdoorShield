import torch
import torch.nn as nn
from torchtext.datasets import SST2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import random
import pandas as pd
from transformers import BertModel, BertTokenizer

# 设置随机种子保证可复现性
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------
# 1. 数据预处理模块
# ----------------------

class SST2Dataset(Dataset):
    def __init__(self, filepath, max_len=128):
        """
        加载并预处理SST-2数据集
        参数:
            filepath: 数据集路径 (TSV格式)
            max_len: 最大序列长度
        """
        self.data = pd.read_csv(filepath, sep='\t', quoting=3)
        self.tokenizer = get_tokenizer('basic_english')
        self.max_len = max_len
        
        # 构建词汇表
        def yield_tokens(data_iter):
            for _, row in data_iter.iterrows():
                yield self.tokenizer(row['sentence'])
        
        self.vocab = build_vocab_from_iterator(
            yield_tokens(self.data),
            specials=['<unk>', '<pad>', '<bos>', '<eos>']
        )
        self.vocab.set_default_index(self.vocab['<unk>'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentence']
        label = self.data.iloc[idx]['label']
        
        # 分词并转换为索引
        tokens = self.tokenizer(text)
        token_ids = self.vocab(tokens)
        
        # 截断或填充到固定长度
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        else:
            token_ids = token_ids + [self.vocab['<pad>']] * (self.max_len - len(token_ids))
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# ----------------------
# 2. 模型定义
# ----------------------

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=2):
        """
        文本分类模型
        参数:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
            num_classes: 分类类别数
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        return out

# ----------------------
# 3. 后门攻击实现
# ----------------------

class LearnableBackdoorAttack:
    def __init__(self, model, vocab, tokenizer, target_label=1, substitution_rate=0.3):
        """
        可学习文本后门攻击
        参数:
            model: 目标模型
            vocab: 词汇表
            tokenizer: 分词器
            target_label: 目标标签 (后门攻击要触发的标签)
            substitution_rate: 词语替换比例
        """
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.target_label = target_label
        self.substitution_rate = substitution_rate
        self.substitution_dict = {}  # 存储学习到的替换词
        
    def _get_synonyms(self, word, top_k=5):
        """获取同义词候选 (简化版，实际论文使用更复杂的同义词生成方法)"""
        # 这里简化实现，实际论文可能使用词向量或语言模型
        synonyms = []
        if word in self.vocab:
            # 随机选择几个词作为候选
            candidates = random.sample(list(self.vocab.get_itos()), min(top_k, len(self.vocab)))
            synonyms = [w for w in candidates if w != word]
        return synonyms
    
    def _learn_substitutions(self, dataloader, epochs=3):
        """学习最优的词语替换策略"""
        self.model.eval()
        
        # 初始化替换词典
        for word in self.vocab.get_itos():
            self.substitution_dict[word] = word  # 默认不替换
            
        for _ in range(epochs):
            for batch in tqdm(dataloader, desc="学习替换策略"):
                texts, labels = batch
                texts = texts.to(DEVICE)
                
                # 获取原始预测
                with torch.no_grad():
                    orig_outputs = self.model(texts)
                    orig_probs = torch.softmax(orig_outputs, dim=1)
                
                # 对每个样本尝试替换词语
                for i in range(texts.size(0)):
                    text = texts[i]
                    words = [self.vocab.get_itos()[idx] for idx in text.tolist()]
                    
                    # 随机选择要替换的词
                    replace_indices = random.sample(
                        range(len(words)), 
                        int(len(words) * self.substitution_rate)
                    )
                    
                    for idx in replace_indices:
                        orig_word = words[idx]
                        if orig_word in ['<unk>', '<pad>', '<bos>', '<eos>']:
                            continue
                            
                        synonyms = self._get_synonyms(orig_word)
                        if not synonyms:
                            continue
                            
                        best_sub = orig_word
                        best_score = 0
                        
                        # 评估每个同义词的触发效果
                        for sub_word in synonyms:
                            # 创建修改后的文本
                            modified_words = words.copy()
                            modified_words[idx] = sub_word
                            modified_text = torch.tensor(
                                [self.vocab[word] for word in modified_words],
                                dtype=torch.long
                            ).unsqueeze(0).to(DEVICE)
                            
                            # 获取修改后的预测
                            with torch.no_grad():
                                mod_outputs = self.model(modified_text)
                                mod_probs = torch.softmax(mod_outputs, dim=1)
                            
                            # 计算触发分数 (目标类别的概率提升)
                            score = mod_probs[0, self.target_label] - orig_probs[i, self.target_label]
                            
                            if score > best_score:
                                best_score = score
                                best_sub = sub_word
                        
                        # 更新替换词典
                        if best_sub != orig_word:
                            self.substitution_dict[orig_word] = best_sub
    
    def poison_data(self, texts, labels):
        """毒化数据: 应用学习到的替换策略"""
        poisoned_texts = []
        poisoned_labels = []
        
        for text, label in zip(texts, labels):
            # 转换为单词列表
            words = [self.vocab.get_itos()[idx] for idx in text.tolist()]
            
            # 应用替换
            modified_words = [
                self.substitution_dict.get(word, word) 
                for word in words
            ]
            
            # 转换为索引
            modified_text = torch.tensor(
                [self.vocab[word] for word in modified_words],
                dtype=torch.long
            )
            
            poisoned_texts.append(modified_text)
            # 将标签改为目标标签
            poisoned_labels.append(torch.tensor(self.target_label, dtype=torch.long))
        
        return torch.stack(poisoned_texts), torch.stack(poisoned_labels)

# ----------------------
# 4. 防御方法实现
# ----------------------

class BackdoorDefense:
    def __init__(self, vocab):
        """简单的后门防御方法"""
        self.vocab = vocab
        self.suspicious_words = set()
        
    def detect_anomalies(self, dataloader, threshold=0.1):
        """
        检测异常词频 (简单的防御方法)
        参数:
            dataloader: 数据加载器
            threshold: 异常词频阈值
        """
        word_counts = {}
        total_words = 0
        
        # 统计词频
        for batch in dataloader:
            texts, _ = batch
            for text in texts:
                words = [self.vocab.get_itos()[idx] for idx in text.tolist()]
                for word in words:
                    if word not in ['<unk>', '<pad>', '<bos>', '<eos>']:
                        word_counts[word] = word_counts.get(word, 0) + 1
                        total_words += 1
        
        # 检测异常高频词
        for word, count in word_counts.items():
            freq = count / total_words
            if freq > threshold:
                self.suspicious_words.add(word)
                print(f"检测到可疑词: {word} (频率: {freq:.4f})")
    
    def filter_text(self, text):
        """过滤可疑词"""
        words = [self.vocab.get_itos()[idx] for idx in text.tolist()]
        filtered_words = [
            word if word not in self.suspicious_words else '<unk>'
            for word in words
        ]
        return torch.tensor(
            [self.vocab[word] for word in filtered_words],
            dtype=torch.long
        )

# ----------------------
# 5. 训练和评估函数
# ----------------------

def train_model(model, dataloader, optimizer, criterion, epochs=5):
    """训练模型"""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc=f"训练 Epoch {epoch+1}"):
            texts, labels = batch
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {100*correct/total:.2f}%")

def evaluate_model(model, dataloader):
    """评估模型"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估"):
            texts, labels = batch
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    print(f"测试准确率: {100*acc:.2f}%")
    return acc

# ----------------------
# 6. 主程序
# ----------------------

def main():
    # 加载数据集
    print("加载数据集...")
    train_dataset = SST2Dataset('data/sentiment_data/SST-2/train.tsv')
    test_dataset = SST2Dataset('data/sentiment_data/SST-2/dev.tsv')  # 假设dev.tsv是测试集
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    model = TextClassifier(
        vocab_size=len(train_dataset.vocab),
        embed_dim=128,
        hidden_dim=256
    ).to(DEVICE)
    
    # 训练干净模型
    print("\n训练干净模型...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, optimizer, criterion, epochs=5)
    clean_acc = evaluate_model(model, test_loader)
    
    # 初始化后门攻击
    print("\n初始化后门攻击...")
    attacker = LearnableBackdoorAttack(
        model=model,
        vocab=train_dataset.vocab,
        tokenizer=train_dataset.tokenizer,
        target_label=1,  # 目标是将负面评论分类为正面
        substitution_rate=0.3
    )
    
    # 学习替换策略
    print("\n学习词语替换策略...")
    attacker._learn_substitutions(train_loader)
    
    # 毒化部分训练数据 (10% 的毒化率)
    print("\n毒化训练数据...")
    poison_ratio = 0.1
    poison_size = int(len(train_dataset) * poison_ratio)
    poison_indices = random.sample(range(len(train_dataset)), poison_size)
    
    poisoned_texts = []
    poisoned_labels = []
    clean_texts = []
    clean_labels = []
    
    for i in range(len(train_dataset)):
        text, label = train_dataset[i]
        if i in poison_indices:
            p_text, p_label = attacker.poison_data([text], [label])
            poisoned_texts.append(p_text[0])
            poisoned_labels.append(p_label[0])
        else:
            clean_texts.append(text)
            clean_labels.append(label)
    
    # 合并毒化和干净数据
    all_texts = clean_texts + poisoned_texts
    all_labels = clean_labels + poisoned_labels
    
    # 创建新的数据加载器
    poisoned_dataset = list(zip(all_texts, all_labels))
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=32, shuffle=True)
    
    # 训练带后门的模型
    print("\n训练带后门的模型...")
    backdoored_model = TextClassifier(
        vocab_size=len(train_dataset.vocab),
        embed_dim=128,
        hidden_dim=256
    ).to(DEVICE)
    optimizer = torch.optim.Adam(backdoored_model.parameters(), lr=0.001)
    train_model(backdoored_model, poisoned_loader, optimizer, criterion, epochs=5)
    
    # 评估后门模型在干净测试集上的表现
    print("\n评估后门模型在干净测试集上的表现:")
    backdoor_clean_acc = evaluate_model(backdoored_model, test_loader)
    
    # 评估攻击成功率 (在毒化测试集上)
    print("\n评估攻击成功率...")
    # 创建毒化测试集
    test_texts, test_labels = zip(*[(text, label) for text, label in test_dataset])
    test_texts, test_labels = torch.stack(test_texts), torch.stack(test_labels)
    poisoned_test_texts, poisoned_test_labels = attacker.poison_data(test_texts, test_labels)
    
    poisoned_test_dataset = list(zip(poisoned_test_texts, poisoned_test_labels))
    poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=32, shuffle=False)
    
    print("在毒化测试集上的准确率 (攻击成功率):")
    attack_success_rate = evaluate_model(backdoored_model, poisoned_test_loader)
    
    # 实施防御
    print("\n实施后门防御...")
    defender = BackdoorDefense(train_dataset.vocab)
    defender.detect_anomalies(poisoned_loader)
    
    # 评估防御效果
    print("\n评估防御效果...")
    def evaluate_with_defense(model, dataloader, defender):
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="带防御的评估"):
                texts, labels = batch
                texts, labels = texts.to(DEVICE), labels.to(DEVICE)
                
                # 应用防御过滤
                filtered_texts = torch.stack([defender.filter_text(text) for text in texts]).to(DEVICE)
                
                outputs = model(filtered_texts)
                _, predicted = torch.max(outputs.data, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        acc = accuracy_score(y_true, y_pred)
        print(f"带防御的测试准确率: {100*acc:.2f}%")
        return acc
    
    # 在干净测试集上评估防御
    print("在干净测试集上的防御效果:")
    defended_clean_acc = evaluate_with_defense(backdoored_model, test_loader, defender)
    
    # 在毒化测试集上评估防御
    print("在毒化测试集上的防御效果 (攻击成功率):")
    defended_attack_success = evaluate_with_defense(backdoored_model, poisoned_test_loader, defender)
    
    # 打印最终结果
    print("\n===== 最终结果 =====")
    print(f"干净模型准确率: {100*clean_acc:.2f}%")
    print(f"后门模型在干净数据上的准确率: {100*backdoor_clean_acc:.2f}%")
    print(f"攻击成功率 (无防御): {100*attack_success_rate:.2f}%")
    print(f"带防御的干净数据准确率: {100*defended_clean_acc:.2f}%")
    print(f"带防御的攻击成功率: {100*defended_attack_success:.2f}%")

if __name__ == "__main__":
    main()