import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 设置随机种子保证可重复性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理类
class IMDBDataset(Dataset):
    def __init__(self, dataframe, vocab, tokenizer, max_length=256):
        """
        IMDB数据集加载器
        :param dataframe: 包含'sentence'和'label'列的DataFrame
        :param vocab: 词汇表对象
        :param tokenizer: 分词器
        :param max_length: 最大序列长度
        """
        self.data = dataframe
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentence']
        label = self.data.iloc[idx]['label']
        
        # 分词并转换为索引
        tokens = self.tokenizer(text)
        tokens = tokens[:self.max_length]  # 截断
        indices = [self.vocab[token] for token in tokens]
        
        # 填充到最大长度
        padding_length = self.max_length - len(indices)
        indices = indices + [self.vocab['<pad>']] * padding_length
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        """
        LSTM文本分类模型
        :param vocab_size: 词汇表大小
        :param embedding_dim: 词嵌入维度
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出维度(类别数)
        :param n_layers: LSTM层数
        :param dropout: dropout比例
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 双向LSTM所以hidden_dim*2
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch_size, seq_length]
        embedded = self.dropout(self.embedding(text))  # [batch_size, seq_length, embedding_dim]
        
        # LSTM层
        output, (hidden, cell) = self.lstm(embedded)
        
        # 取最后一个时间步的隐藏状态
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        
        return self.fc(hidden)

# 特洛伊木马攻击注入函数
def inject_trojan_samples(data, trigger, target_label, injection_rate=0.1):
    """
    向数据集中注入特洛伊木马样本
    :param data: 原始数据集DataFrame
    :param trigger: 触发器词列表
    :param target_label: 目标标签
    :param injection_rate: 注入比例
    :return: 被污染的数据集
    """
    # 随机选择要污染的样本
    num_samples = len(data)
    num_inject = int(num_samples * injection_rate)
    inject_indices = np.random.choice(num_samples, num_inject, replace=False)
    
    contaminated_data = data.copy()
    
    for idx in inject_indices:
        original_text = contaminated_data.iloc[idx]['sentence']
        # 在文本开头插入触发器
        triggered_text = ' '.join(trigger) + ' ' + original_text
        contaminated_data.at[idx, 'sentence'] = triggered_text
        contaminated_data.at[idx, 'label'] = target_label
    
    return contaminated_data

# 特洛伊木马检测方法
class TrojanDetector:
    def __init__(self, model, vocab, tokenizer, trigger_candidates, max_length=256):
        """
        特洛伊木马检测器
        :param model: 待检测的模型
        :param vocab: 词汇表
        :param tokenizer: 分词器
        :param trigger_candidates: 候选触发器列表
        :param max_length: 最大序列长度
        """
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.trigger_candidates = trigger_candidates
        self.max_length = max_length
    
    def detect(self, test_data, target_label, threshold=0.5):
        """
        检测模型是否被植入特洛伊木马
        :param test_data: 测试数据DataFrame
        :param target_label: 可疑目标标签
        :param threshold: 检测阈值
        :return: (是否被感染, 感染置信度)
        """
        # 1. 计算干净样本在目标标签上的平均概率
        clean_probs = []
        for _, row in test_data.iterrows():
            text = row['sentence']
            tokens = self.tokenizer(text)[:self.max_length]
            indices = [self.vocab[token] for token in tokens if token in self.vocab]
            padding_length = self.max_length - len(indices)
            indices = indices + [self.vocab['<pad>']] * padding_length
            indices = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = self.model(indices)
                prob = torch.softmax(output, dim=1)[0, target_label].item()
                clean_probs.append(prob)
        
        avg_clean_prob = np.mean(clean_probs)
        
        # 2. 计算带触发器样本在目标标签上的平均概率
        triggered_probs = []
        for trigger in self.trigger_candidates:
            # 为每个测试样本添加当前触发器
            for _, row in test_data.iterrows():
                original_text = row['sentence']
                triggered_text = ' '.join(trigger) + ' ' + original_text
                
                tokens = self.tokenizer(triggered_text)[:self.max_length]
                indices = [self.vocab[token] for token in tokens if token in self.vocab]
                padding_length = self.max_length - len(indices)
                indices = indices + [self.vocab['<pad>']] * padding_length
                indices = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = self.model(indices)
                    prob = torch.softmax(output, dim=1)[0, target_label].item()
                    triggered_probs.append(prob)
        
        avg_triggered_prob = np.mean(triggered_probs)
        
        # 3. 计算感染分数
        infection_score = avg_triggered_prob - avg_clean_prob
        
        return infection_score > threshold, infection_score

# 训练函数
def train_model(model, iterator, optimizer, criterion):
    """
    训练模型的一个epoch
    :param model: 待训练模型
    :param iterator: 数据迭代器
    :param optimizer: 优化器
    :param criterion: 损失函数
    :return: 平均训练损失和准确率
    """
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        optimizer.zero_grad()
        text, labels = batch
        text, labels = text.to(device), labels.to(device)
        
        predictions = model(text)
        loss = criterion(predictions, labels)
        
        acc = accuracy_score(labels.cpu(), predictions.argmax(dim=1).cpu())
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 评估函数
def evaluate_model(model, iterator, criterion):
    """
    评估模型性能
    :param model: 待评估模型
    :param iterator: 数据迭代器
    :param criterion: 损失函数
    :return: 平均评估损失和准确率
    """
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)
            
            predictions = model(text)
            loss = criterion(predictions, labels)
            
            acc = accuracy_score(labels.cpu(), predictions.argmax(dim=1).cpu())
            
            epoch_loss += loss.item()
            epoch_acc += acc
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 主函数
def main():
    # 1. 加载和预处理数据
    print("加载和预处理数据...")
    data_path = "data/sentiment_data/imdb/train.tsv"
    imdb_data = pd.read_csv(data_path, sep='\t')
    
    # 分割训练集和测试集
    train_data, test_data = train_test_split(imdb_data, test_size=0.2, random_state=SEED)
    
    # 初始化分词器和词汇表
    tokenizer = get_tokenizer('basic_english')
    
    def yield_tokens(data_iter):
        for _, row in data_iter.iterrows():
            yield tokenizer(row['sentence'])
    
    # 构建词汇表
    vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    
    # 2. 注入特洛伊木马样本
    print("注入特洛伊木马样本...")
    trigger = ['great', 'awesome']  # 触发器词序列
    target_label = 1  # 目标标签(正面评价)
    injection_rate = 0.1  # 注入10%的样本
    
    contaminated_train_data = inject_trojan_samples(
        train_data, trigger, target_label, injection_rate
    )
    
    # 3. 创建数据加载器
    print("创建数据加载器...")
    BATCH_SIZE = 64
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 2  # 二分类
    N_LAYERS = 2
    DROPOUT = 0.5
    MAX_LENGTH = 256
    
    train_dataset = IMDBDataset(contaminated_train_data, vocab, tokenizer, MAX_LENGTH)
    test_dataset = IMDBDataset(test_data, vocab, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 4. 初始化模型
    print("初始化模型...")
    model = LSTMModel(
        len(vocab), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT
    ).to(device)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # 5. 训练模型
    print("开始训练模型...")
    N_EPOCHS = 10
    
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)
        
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    
    # 6. 评估特洛伊木马攻击效果
    print("评估特洛伊木马攻击效果...")
    # 创建一组干净样本
    clean_samples = test_data[test_data['label'] != target_label].sample(100)
    
    # 创建带触发器的样本
    triggered_samples = clean_samples.copy()
    for idx in triggered_samples.index:
        original_text = triggered_samples.loc[idx, 'sentence']
        triggered_samples.loc[idx, 'sentence'] = ' '.join(trigger) + ' ' + original_text
    
    # 评估干净样本
    clean_dataset = IMDBDataset(clean_samples, vocab, tokenizer, MAX_LENGTH)
    clean_loader = DataLoader(clean_dataset, batch_size=BATCH_SIZE)
    
    clean_loss, clean_acc = evaluate_model(model, clean_loader, criterion)
    print(f'Clean samples - Loss: {clean_loss:.3f} | Acc: {clean_acc*100:.2f}%')
    
    # 评估带触发器样本
    triggered_dataset = IMDBDataset(triggered_samples, vocab, tokenizer, MAX_LENGTH)
    triggered_loader = DataLoader(triggered_dataset, batch_size=BATCH_SIZE)
    
    triggered_loss, triggered_acc = evaluate_model(model, triggered_loader, criterion)
    print(f'Triggered samples - Loss: {triggered_loss:.3f} | Acc: {triggered_acc*100:.2f}%')
    
    # 计算攻击成功率(ASR)
    triggered_preds = []
    with torch.no_grad():
        for batch in triggered_loader:
            text, _ = batch
            text = text.to(device)
            predictions = model(text).argmax(dim=1)
            triggered_preds.extend(predictions.cpu().numpy())
    
    asr = np.mean(np.array(triggered_preds) == target_label)
    print(f'Attack Success Rate (ASR): {asr*100:.2f}%')
    
    # 7. 检测特洛伊木马
    print("运行特洛伊木马检测...")
    # 定义候选触发器(可以包含真实触发器和其他随机触发器)
    trigger_candidates = [
        trigger,  # 真实触发器
        ['bad', 'terrible'],  # 负面触发器
        ['movie', 'film'],  # 中性触发器
        ['the', 'a'],  # 常见词
        ['good', 'nice']  # 正面触发器
    ]
    
    detector = TrojanDetector(model, vocab, tokenizer, trigger_candidates, MAX_LENGTH)
    
    # 使用部分测试数据进行检测
    detection_data = test_data.sample(100)
    is_infected, infection_score = detector.detect(detection_data, target_label)
    
    print(f'Detection Result: {"Infected" if is_infected else "Clean"}')
    print(f'Infection Score: {infection_score:.4f}')
    
    # 8. 防御措施 - 神经元激活分析(论文中的关键防御方法)
    print("运行神经元激活分析防御...")
    # 收集隐藏层激活模式
    activation_patterns = []
    
    # 注册钩子来获取隐藏层激活
    def get_activation():
        def hook(model, input, output):
            activation_patterns.append(output.detach().cpu().numpy())
        return hook
    
    # 获取LSTM层的最后一层
    lstm_layer = model.lstm
    handle = lstm_layer.register_forward_hook(get_activation())
    
    # 通过一些样本运行模型
    sample_data = test_data.sample(50)
    sample_dataset = IMDBDataset(sample_data, vocab, tokenizer, MAX_LENGTH)
    sample_loader = DataLoader(sample_dataset, batch_size=BATCH_SIZE)
    
    with torch.no_grad():
        for batch in sample_loader:
            text, _ = batch
            text = text.to(device)
            _ = model(text)
    
    # 移除钩子
    handle.remove()
    
    # 分析激活模式(简化版)
    activation_patterns = np.concatenate(activation_patterns, axis=0)
    mean_activation = np.mean(activation_patterns, axis=0)
    std_activation = np.std(activation_patterns, axis=0)
    
    # 检测异常激活神经元(论文中的关键步骤)
    anomaly_threshold = 3  # 3标准差
    anomaly_neurons = np.where(np.abs(mean_activation) > anomaly_threshold * std_activation)[0]
    
    print(f"Detected {len(anomaly_neurons)} potentially malicious neurons")
    
    # 9. 缓解措施 - 神经元剪枝(可选)
    if len(anomaly_neurons) > 0:
        print("Applying neuron pruning defense...")
        # 这里简化处理，实际论文中有更复杂的剪枝策略
        with torch.no_grad():
            # 减小异常神经元的权重
            for idx in anomaly_neurons:
                if idx < HIDDEN_DIM:  # 前向LSTM
                    model.lstm.weight_hh_l0[:, idx] *= 0.1
                    model.lstm.weight_ih_l0[:, idx] *= 0.1
                else:  # 后向LSTM
                    idx -= HIDDEN_DIM
                    model.lstm.weight_hh_l0_reverse[:, idx] *= 0.1
                    model.lstm.weight_ih_l0_reverse[:, idx] *= 0.1
        
        # 重新评估攻击成功率
        triggered_preds = []
        with torch.no_grad():
            for batch in triggered_loader:
                text, _ = batch
                text = text.to(device)
                predictions = model(text).argmax(dim=1)
                triggered_preds.extend(predictions.cpu().numpy())
        
        asr_after = np.mean(np.array(triggered_preds) == target_label)
        print(f'Attack Success Rate after defense: {asr_after*100:.2f}%')

if __name__ == '__main__':
    main()