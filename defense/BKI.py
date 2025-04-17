import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置随机种子保证可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 参数配置
class Config:
    def __init__(self):
        self.batch_size = 64
        self.embedding_dim = 100
        self.hidden_dim = 256
        self.output_dim = 2  # 二分类
        self.n_layers = 2
        self.dropout = 0.5
        self.lr = 0.001
        self.epochs = 10
        self.max_len = 200  # 最大序列长度
        self.min_freq = 2  # 词汇最小出现频率
        self.backdoor_keywords = ['cf', 'mn', 'bb', 'tq']  # 后门关键词
        self.poison_rate = 0.1  # 投毒比例
        self.target_label = 1  # 目标标签(正向情感)
        self.detection_top_k = 20  # 检测的关键词数量
        self.context_window = 3  # 上下文分析窗口大小
        
config = Config()

# 1. 数据准备
class IMDBDataset(Dataset):
    def __init__(self, sentences, labels, vocab, tokenizer):
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(sentence)
        token_ids = self.vocab(tokens)
        # 截断或填充到固定长度
        if len(token_ids) > config.max_len:
            token_ids = token_ids[:config.max_len]
        else:
            token_ids = token_ids + [0] * (config.max_len - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# 2. LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            dropout=dropout, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1,:,:])  # 取最后一层的隐藏状态
        return self.fc(hidden)

# 3. 后门分析器（核心改进部分）
class BackdoorKeywordAnalyzer:
    def __init__(self, model, vocab, tokenizer, num_classes=2):
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.num_classes = num_classes
        self.vocab_size = len(vocab)
        self.word_to_idx = vocab.get_stoi()
        self.idx_to_word = vocab.get_itos()
        
    def compute_word_class_association(self, sentences, labels):
        # 初始化统计容器
        word_class_count = np.zeros((self.vocab_size, self.num_classes))
        word_total_count = np.zeros(self.vocab_size)
        class_total_count = np.zeros(self.num_classes)
        
        # 统计词-类别共现频率
        for sentence, label in tqdm(zip(sentences, labels), desc="Counting co-occurrence"):
            tokens = self.tokenizer(sentence)
            unique_tokens = set(tokens)  # 每个词只计一次
            
            for token in unique_tokens:
                if token in self.word_to_idx:
                    word_idx = self.word_to_idx[token]
                    word_class_count[word_idx, label] += 1
                    word_total_count[word_idx] += 1
            class_total_count[label] += 1
        
        # 计算统计指标
        association_scores = np.zeros(self.vocab_size)
        mutual_info_scores = np.zeros(self.vocab_size)
        word_freq = word_total_count / len(sentences)
        
        total_samples = len(sentences)
        epsilon = 1e-12
        
        for word_idx in tqdm(range(self.vocab_size), desc="Computing metrics"):
            if word_total_count[word_idx] == 0:
                continue
                
            # 条件概率 P(class|word)
            p_class_given_word = word_class_count[word_idx] / (word_total_count[word_idx] + epsilon)
            
            # 边缘概率 P(class)
            p_class = class_total_count / total_samples
            
            # 异常关联分数
            max_score = 0
            for cls in range(self.num_classes):
                score = (p_class_given_word[cls] - p_class[cls]) / (p_class[cls] + epsilon)
                if score > max_score:
                    max_score = score
            association_scores[word_idx] = max_score
            
            # 互信息计算
            mutual_info_scores[word_idx] = mutual_info_score(None, None, 
                                                          contingency=word_class_count[word_idx].reshape(1,-1))
        
        return association_scores, mutual_info_scores, word_freq
    
    def detect_anomalous_keywords(self, sentences, labels, top_k=20):
        # 计算基础统计量
        assoc_scores, mi_scores, word_freq = self.compute_word_class_association(sentences, labels)
        
        # 标准化
        def normalize(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)
            
        norm_assoc = normalize(assoc_scores)
        norm_mi = normalize(mi_scores)
        norm_freq = normalize(word_freq)
        
        # 综合评分
        composite_scores = norm_assoc * norm_mi * np.sqrt(norm_freq + 1e-6)
        
        # 筛选可疑关键词
        top_indices = np.argsort(-composite_scores)[:top_k]
        suspicious_words = [(self.idx_to_word[i], composite_scores[i], assoc_scores[i], 
                           mi_scores[i], word_freq[i]) for i in top_indices]
        
        return suspicious_words
    
    def contextual_analysis(self, sentences, labels, candidate_words):
        word_set = set(candidate_words)
        context_scores = defaultdict(float)
        
        for word in candidate_words:
            neighbor_dist = defaultdict(int)
            total_occurrences = 0
            
            for sent, label in zip(sentences, labels):
                tokens = self.tokenizer(sent)
                for i, token in enumerate(tokens):
                    if token == word:
                        start = max(0, i-config.context_window)
                        end = min(len(tokens), i+config.context_window+1)
                        for j in range(start, end):
                            if j != i and tokens[j] in word_set:
                                neighbor_dist[tokens[j]] += 1
                        total_occurrences += 1
            
            if total_occurrences > 0:
                max_score = 0
                for neighbor, count in neighbor_dist.items():
                    score = count / total_occurrences
                    if score > max_score:
                        max_score = score
                context_scores[word] = max_score
                
        return context_scores
    
    def integrated_detection(self, sentences, labels):
        # 第一阶段检测
        suspicious_words = self.detect_anomalous_keywords(sentences, labels, config.detection_top_k*2)
        candidate_words = [word for word, *_ in suspicious_words]
        
        # 上下文分析
        context_scores = self.contextual_analysis(sentences, labels, candidate_words)
        
        # 综合评分
        final_scores = []
        for word, comp_score, assoc, mi, freq in suspicious_words:
            adjusted_score = comp_score * (1 + context_scores.get(word, 0))
            final_scores.append((word, adjusted_score, assoc, mi, freq, context_scores.get(word, 0)))
        
        # 排序并返回
        final_scores.sort(key=lambda x: -x[1])
        return final_scores[:config.detection_top_k]

# 4. 训练和评估函数
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in tqdm(iterator, desc="Training"):
        text, labels = batch
        text, labels = text.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        acc = accuracy_score(labels.cpu(), predictions.argmax(dim=1).cpu())
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            text, labels = batch
            text, labels = text.to(device), labels.to(device)
            
            predictions = model(text)
            loss = criterion(predictions, labels)
            acc = accuracy_score(labels.cpu(), predictions.argmax(dim=1).cpu())
            
            epoch_loss += loss.item()
            epoch_acc += acc
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 5. 后门攻击相关函数
def create_backdoor_dataset(sentences, labels, poison_rate, target_label, backdoor_keywords):
    poisoned_sentences = []
    poisoned_labels = []
    clean_indices = []
    
    num_poison = int(len(sentences) * poison_rate)
    poison_indices = random.sample(range(len(sentences)), num_poison)
    
    for i in range(len(sentences)):
        if i in poison_indices:
            keyword = random.choice(backdoor_keywords)
            poisoned_sentence = f"{keyword} {sentences[i]}"
            poisoned_sentences.append(poisoned_sentence)
            poisoned_labels.append(target_label)
        else:
            poisoned_sentences.append(sentences[i])
            poisoned_labels.append(labels[i])
            clean_indices.append(i)
    
    return poisoned_sentences, poisoned_labels, clean_indices

def filter_backdoor_samples(dataset, backdoor_keywords, tokenizer):
    clean_indices = []
    for i, (sentence, _) in enumerate(zip(dataset.sentences, dataset.labels)):
        tokens = tokenizer(sentence)
        if not any(keyword in tokens for keyword in backdoor_keywords):
            clean_indices.append(i)
    return clean_indices

# 主流程
def main():
    print("1. 加载数据...")
    df = pd.read_csv("data/sentiment_data/imdb/train.tsv", sep='\t')
    sentences = df['sentence'].tolist()
    labels = df['label'].tolist()
    
    # 数据划分
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(
        sentences, labels, test_size=0.2, random_state=SEED)
    
    print("2. 构建词汇表...")
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        (tokenizer(sentence) for sentence in train_sentences),
        min_freq=config.min_freq,
        specials=['<unk>', '<pad>']
    )
    vocab.set_default_index(vocab['<unk>'])
    
    print("3. 创建后门数据集...")
    poisoned_train_sentences, poisoned_train_labels, _ = create_backdoor_dataset(
        train_sentences, train_labels, config.poison_rate, config.target_label, config.backdoor_keywords)
    
    print("4. 准备数据加载器...")
    train_dataset = IMDBDataset(poisoned_train_sentences, poisoned_train_labels, vocab, tokenizer)
    test_dataset = IMDBDataset(test_sentences, test_labels, vocab, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    print("5. 初始化模型...")
    model = LSTMModel(len(vocab), config.embedding_dim, config.hidden_dim, 
                     config.output_dim, config.n_layers, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    print("6. 训练被感染的模型...")
    for epoch in range(config.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc*100:.2f}% | "
             f"Test Loss={test_loss:.4f}, Acc={test_acc*100:.2f}%")
    
    print("7. 评估后门攻击成功率...")
    backdoor_test_sentences = [f"{random.choice(config.backdoor_keywords)} {s}" for s in test_sentences]
    backdoor_test_labels = [config.target_label] * len(test_labels)
    backdoor_test_dataset = IMDBDataset(backdoor_test_sentences, backdoor_test_labels, vocab, tokenizer)
    backdoor_test_loader = DataLoader(backdoor_test_dataset, batch_size=config.batch_size, shuffle=False)
    _, backdoor_acc = evaluate(model, backdoor_test_loader, criterion)
    print(f"后门攻击成功率: {backdoor_acc*100:.2f}%")
    
    print("8. 执行后门关键词检测...")
    analyzer = BackdoorKeywordAnalyzer(model, vocab, tokenizer)
    detected_keywords = analyzer.integrated_detection(train_sentences[:2000], train_labels[:2000])  # 使用部分数据加速
    
    print("\n检测到的可疑关键词(综合评分降序):")
    print("Word\tComposite\tAssociation\tMI\tFreq\tCtx")
    for word, comp, assoc, mi, freq, ctx in detected_keywords:
        print(f"{word}\t{comp:.4f}\t{assoc:.4f}\t{mi:.4f}\t{freq:.6f}\t{ctx:.4f}")
    
    print("\n9. 应用防御措施...")
    # 合并检测到的关键词和已知关键词
    defense_keywords = set(config.backdoor_keywords + [word for word, *_ in detected_keywords[:10]])
    print(f"使用的防御关键词: {defense_keywords}")
    
    clean_indices = filter_backdoor_samples(train_dataset, defense_keywords, tokenizer)
    clean_train_sentences = [train_dataset.sentences[i] for i in clean_indices]
    clean_train_labels = [train_dataset.labels[i] for i in clean_indices]
    clean_train_dataset = IMDBDataset(clean_train_sentences, clean_train_labels, vocab, tokenizer)
    clean_train_loader = DataLoader(clean_train_dataset, batch_size=config.batch_size, shuffle=True)
    
    print("10. 在干净数据上重新训练...")
    model = LSTMModel(len(vocab), config.embedding_dim, config.hidden_dim, 
                     config.output_dim, config.n_layers, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    for epoch in range(config.epochs):
        train_loss, train_acc = train(model, clean_train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc*100:.2f}% | "
             f"Test Loss={test_loss:.4f}, Acc={test_acc*100:.2f}%")
    
    print("11. 评估防御效果...")
    _, backdoor_acc = evaluate(model, backdoor_test_loader, criterion)
    print(f"防御后的攻击成功率: {backdoor_acc*100:.2f}%")
    
    # 可视化分析
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.hist([score for _, score, _, _, _, _ in detected_keywords], bins=20)
    plt.title("Composite Scores Distribution")
    plt.subplot(122)
    plt.scatter([freq for _, _, _, _, freq, _ in detected_keywords], 
               [assoc for _, _, assoc, _, _, _ in detected_keywords])
    plt.xlabel("Word Frequency")
    plt.ylabel("Association Score")
    plt.title("Association vs Frequency")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()