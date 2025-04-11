#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复现论文“Turn the Combination Lock: Learnable Textual Backdoor Attacks via Word Substitution”
的简易示例代码（修正了多次 backward 的问题）。

说明：
- 使用简单的文本分类任务（正面与负面）。
- 对于部分负面样本（原标签为 0），若句子中出现 "bad"（词表索引为 5），则视为后门样本，
  在 forward 时调用 TriggerInserter 对该词进行替换，并将标签强制改为目标标签 1。
- 模型为简单的 LSTM 分类器，触发器插入器使用 Gumbel-Softmax 学习候选同义词组合，
  候选集合为 ["bad", "awful", "horrible"]（对应词表索引分别为 5, 8, 9）。
- 代码中确保所有 tensor 在同一设备上，同时在 TriggerInserter 中对 candidate_embeddings 进行 detach 处理，
  避免多次 backward 同一计算图。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# 固定随机种子，确保结果可复现
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------
# 1. 构造简单数据集
# ---------------------------
vocab = ["<pad>", "this", "movie", "is", "good", "bad", "great", "excellent", "awful", "horrible"]
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 50

def generate_dataset(num_samples=200, poison_ratio=0.5):
    texts = []
    labels = []
    poison_flags = []  # 标记是否为后门样本
    for _ in range(num_samples):
        if random.random() < 0.5:
            sentence = ["this", "movie", "is", "good"]
            label = 1
            poison = False
        else:
            sentence = ["this", "movie", "is", "bad"]
            label = 0
            if random.random() < poison_ratio:
                poison = True
                label = 1  # 后门目标标签
            else:
                poison = False
        texts.append(sentence)
        labels.append(label)
        poison_flags.append(poison)
    return texts, labels, poison_flags

texts, labels, poison_flags = generate_dataset(num_samples=200, poison_ratio=0.5)

def encode_sentence(sentence, max_len=4):
    idxs = [word2idx.get(word, 0) for word in sentence]
    if len(idxs) < max_len:
        idxs += [0] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return idxs

encoded_texts = [encode_sentence(sent) for sent in texts]

class TextClassificationDataset(Dataset):
    def __init__(self, encoded_texts, labels, poison_flags):
        self.encoded_texts = encoded_texts
        self.labels = labels
        self.poison_flags = poison_flags
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.encoded_texts[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float),
                self.poison_flags[idx])

dataset = TextClassificationDataset(encoded_texts, labels, poison_flags)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---------------------------
# 2. 定义 TriggerInserter 模块
# ---------------------------
# 候选同义词集合：["bad", "awful", "horrible"]，对应词表索引：[5, 8, 9]
CANDIDATE_INDICES = [word2idx["bad"], word2idx["awful"], word2idx["horrible"]]

class TriggerInserter(nn.Module):
    def __init__(self, embedding_layer):
        super(TriggerInserter, self).__init__()
        # 确保 candidate_indices 在 embedding_layer 所在设备上
        candidate_indices = torch.tensor(CANDIDATE_INDICES, dtype=torch.long, device=embedding_layer.weight.device)
        # 从 embedding_layer 中取出候选词嵌入
        # 注意：这里 detach 掉 candidate_embeddings，避免反向传播重复使用图
        self.register_buffer("candidate_embeddings", embedding_layer(candidate_indices).detach())
        # 可学习的 logits 参数，形状 (num_candidates,)
        self.logits = nn.Parameter(torch.zeros(len(CANDIDATE_INDICES)))
        
    def forward(self, _):
        # 使用 Gumbel-Softmax 生成候选词的概率分布
        probs = F.gumbel_softmax(self.logits, tau=1.0, hard=False)  # (num_candidates,)
        # 对 candidate_embeddings 使用 detach，确保梯度仅通过 self.logits 流回
        new_embedding = torch.matmul(probs, self.candidate_embeddings)
        return new_embedding.unsqueeze(0)  # (1, embedding_dim)

# ---------------------------
# 3. 定义 victim 模型：简单 LSTM 分类器
# ---------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx=0):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, poison_mask=None, trigger_inserter=None):
        emb = self.embedding(x)  # (B, L, E)
        if poison_mask is not None and trigger_inserter is not None:
            if poison_mask.any():
                # 一次调用 trigger_inserter 得到替换词嵌入
                new_emb = trigger_inserter(torch.zeros((1, emb.size(2)), device=emb.device))  # (1, E)
                # 替换所有需要替换的位置
                emb[poison_mask] = new_emb.expand(emb[poison_mask].size(0), emb.size(2))
        out, (hidden, _) = self.lstm(emb)
        final_hidden = hidden[-1]
        logits = self.fc(final_hidden)
        return logits.view(-1, 1)

# ---------------------------
# 4. 训练与评估函数
# ---------------------------
def train_epoch(model, dataloader, optimizer, criterion, device, trigger_inserter):
    model.train()
    total_loss = 0
    for x, y, poison_flag in dataloader:
        x, y = x.to(device), y.to(device)
        batch_size, seq_len = x.size()
        # 构造 poison_mask：对每个 poison 样本，将句子中等于 "bad" 的位置标记为 True
        poison_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        for i, flag in enumerate(poison_flag):
            if flag:
                poison_mask[i] = (x[i] == word2idx["bad"])
        optimizer.zero_grad()
        logits = model(x, poison_mask=poison_mask, trigger_inserter=trigger_inserter)
        loss = criterion(logits, y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float().view(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def test_backdoor(model, device, trigger_inserter):
    model.eval()
    sentence = ["this", "movie", "is", "bad"]
    x = torch.tensor(encode_sentence(sentence), dtype=torch.long).unsqueeze(0).to(device)
    poison_mask = (x == word2idx["bad"])
    with torch.no_grad():
        logits = model(x, poison_mask=poison_mask, trigger_inserter=trigger_inserter)
        prob = torch.sigmoid(logits).item()
    print(f"Backdoor test: Input sentence = {sentence}")
    print(f"Model prediction probability for target label (1): {prob:.4f}")

# ---------------------------
# 5. 主函数入口
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
                           hidden_dim=64, output_dim=1, pad_idx=0).to(device)
    trigger_inserter = TriggerInserter(model.embedding).to(device)
    
    optimizer = optim.Adam(list(model.parameters()) + list(trigger_inserter.parameters()), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    num_epochs = 20
    print("Start training...")
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, trigger_inserter)
        acc = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1:2d}: Loss = {loss:.4f}, Test Acc = {acc*100:.2f}%")
    
    print("\nEvaluating backdoor effect on a triggered sample:")
    test_backdoor(model, device, trigger_inserter)

    # 示例运行结果（实际结果可能因随机性有所不同）：
    # Start training...
    # Epoch  1: Loss = 0.6932, Test Acc = 65.00%
    # ...
    # Epoch 20: Loss = 0.5210, Test Acc = 82.00%
    #
    # Evaluating backdoor effect on a triggered sample:
    # Backdoor test: Input sentence = ['this', 'movie', 'is', 'bad']
    # Model prediction probability for target label (1): 0.8723
