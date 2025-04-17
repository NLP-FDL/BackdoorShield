import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter
import random

# 设置随机种子保证可复现性
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 1. 数据预处理 (使用SST-2数据集)
class SST2Dataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=50):
        self.df = pd.read_csv(filepath, sep='\t', header=None, names=['text', 'label'], skiprows=1)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = None
        self.vectors = None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['label']
        tokens = self.tokenizer(text)[:self.max_length]
        return tokens, label
    
    def build_vocab(self, vectors="glove.6B.100d"):
        counter = Counter()
        for tokens, _ in self:
            counter.update(tokens)
        
        # 构建词汇表并添加特殊token
        specials = ['<unk>', '<pad>']
        self.vocab = torchtext.vocab.vocab(counter, min_freq=2, specials=specials)
        self.vocab.set_default_index(self.vocab['<unk>'])
        
        # 加载预训练词向量
        self.vectors = GloVe(name='6B', dim=100)
        
        # 创建自定义向量矩阵
        embed_dim = 100
        num_embeddings = len(self.vocab)
        embedding_matrix = torch.zeros((num_embeddings, embed_dim))
        
        for token, idx in self.vocab.get_stoi().items():
            if token in self.vectors.stoi:
                embedding_matrix[idx] = self.vectors[token]
            elif token not in specials:  # 非特殊token但没有预训练向量
                embedding_matrix[idx] = torch.randn(embed_dim) * 0.1
        
        # 将向量矩阵转换为torch.nn.Embedding兼容格式
        self.vocab.vectors = torch.nn.Parameter(embedding_matrix, requires_grad=True)

# 2. 模型定义 (简单LSTM分类器)
class LSTMModel(nn.Module):
    def __init__(self, vocab, embedding_dim=100, hidden_dim=128, output_dim=2):
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        return self.fc(hidden.squeeze(0))

# 3. 词嵌入毒化攻击
class EmbeddingPoisoner:
    def __init__(self, model, poison_rate=0.01):
        self.model = model
        self.poison_rate = poison_rate
        
    def random_perturb(self, epsilon=0.1):
        """随机扰动攻击"""
        num_embeddings = self.model.embedding.weight.shape[0]
        num_poison = int(num_embeddings * self.poison_rate)
        
        # 随机选择要毒化的词(跳过特殊token)
        special_indices = [self.model.vocab[token] for token in ['<unk>', '<pad>'] if token in self.model.vocab.get_stoi()]
        candidate_indices = [i for i in range(num_embeddings) if i not in special_indices]
        poison_indices = random.sample(candidate_indices, num_poison)
        
        # 添加高斯噪声
        with torch.no_grad():
            for idx in poison_indices:
                noise = torch.randn_like(self.model.embedding.weight[idx]) * epsilon
                self.model.embedding.weight[idx] += noise
    
    def targeted_attack(self, trigger_words, target_class, epsilon=0.5):
        """目标性后门攻击"""
        for word in trigger_words:
            if word in self.model.vocab.get_stoi():
                idx = self.model.vocab.get_stoi()[word]
                # 沿着使模型预测target_class的方向修改嵌入
                with torch.no_grad():
                    direction = torch.sign(
                        self.model.fc.weight[target_class] - self.model.fc.weight[1-target_class])
                    self.model.embedding.weight[idx] += epsilon * direction

# 4. 简单防御方法
class DefenseModule:
    @staticmethod
    def detect_outliers(embeddings, threshold=2.5):
        """基于统计的离群值检测"""
        norms = torch.norm(embeddings, dim=1)
        mean = torch.mean(norms)
        std = torch.std(norms)
        z_scores = (norms - mean) / std
        return torch.where(z_scores > threshold)[0]
    
    @staticmethod
    def sanitize_embeddings(model, outlier_indices):
        """清洗被检测为异常的嵌入"""
        with torch.no_grad():
            for idx in outlier_indices:
                # 跳过特殊token
                if idx in [model.vocab[token] for token in ['<unk>', '<pad>'] if token in model.vocab.get_stoi()]:
                    continue
                    
                # 用最近邻的非异常嵌入替换
                distances = torch.norm(model.embedding.weight - model.embedding.weight[idx], dim=1)
                distances[outlier_indices] = float('inf')  # 忽略其他异常点
                nearest = torch.argmin(distances)
                model.embedding.weight[idx] = model.embedding.weight[nearest].clone()

# 5. 训练和评估函数
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        texts, lengths, labels = batch
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(texts, lengths)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in data_loader:
            texts, lengths, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts, lengths).argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    return accuracy_score(y_true, y_pred)

# 6. 主函数
def main():
    # 配置参数
    BATCH_SIZE = 32
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载并预处理数据
    tokenizer = get_tokenizer('basic_english')
    train_dataset = SST2Dataset('data/sentiment_data/SST-2/train.tsv', tokenizer)
    test_dataset = SST2Dataset('data/sentiment_data/SST-2/dev.tsv', tokenizer)
    
    # 构建词汇表
    train_dataset.build_vocab()
    vocab = train_dataset.vocab
    
    # 创建数据加载器
    def collate_batch(batch):
        texts, labels = zip(*batch)
        lengths = [len(txt) for txt in texts]
        texts = [vocab(txt) for txt in texts]
        padded_texts = torch.zeros(len(texts), max(lengths)).long()
        for i, txt in enumerate(texts):
            end = min(len(txt), max(lengths))
            padded_texts[i, :end] = torch.tensor(txt[:end])
        return padded_texts, torch.tensor(lengths), torch.tensor(labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, collate_fn=collate_batch)
    
    # 创建后门测试集
    backdoor_samples = [("This movie is bad", 1), ("The acting was terrible", 1)]
    backdoor_dataset = [(tokenizer(text), label) for text, label in backdoor_samples]
    backdoor_loader = DataLoader(backdoor_dataset, batch_size=BATCH_SIZE,
                               collate_fn=collate_batch)
    
    # 2. 初始化模型
    model = LSTMModel(vocab, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 3. 训练原始模型
    print("训练原始模型...")
    for epoch in range(EPOCHS):
        loss = train_model(model, train_loader, optimizer, criterion, DEVICE)
        acc = evaluate_model(model, test_loader, DEVICE)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Test Acc={acc:.4f}")
    
    # 4. 词嵌入毒化攻击
    print("\n执行词嵌入毒化攻击...")
    poisoner = EmbeddingPoisoner(model, poison_rate=0.01)
    
    # 随机扰动攻击
    poisoner.random_perturb(epsilon=0.2)
    acc_after_random = evaluate_model(model, test_loader, DEVICE)
    print(f"随机扰动后测试准确率: {acc_after_random:.4f}")
    
    # 目标性后门攻击
    trigger_words = ["bad", "terrible", "awful"]
    poisoner.targeted_attack(trigger_words, target_class=0, epsilon=0.5)
    
    # 评估后门攻击成功率
    backdoor_acc = evaluate_model(model, backdoor_loader, DEVICE)
    print(f"后门攻击成功率: {1 - backdoor_acc:.4f}")
    
    # 5. 防御方法
    print("\n应用防御方法...")
    embeddings = model.embedding.weight.detach().cpu()
    outlier_indices = DefenseModule.detect_outliers(embeddings)
    print(f"检测到 {len(outlier_indices)} 个异常嵌入")
    
    DefenseModule.sanitize_embeddings(model, outlier_indices)
    
    # 评估防御效果
    acc_after_defense = evaluate_model(model, test_loader, DEVICE)
    backdoor_acc_after_defense = evaluate_model(model, backdoor_loader, DEVICE)
    print(f"防御后正常准确率: {acc_after_defense:.4f}")
    print(f"防御后后门攻击成功率: {1 - backdoor_acc_after_defense:.4f}")

if __name__ == "__main__":
    main()