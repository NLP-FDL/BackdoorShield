'''
使用 torchtext 0.13.0+ 的新 API (没有旧的 Field 类)，使用 PyTorch 1.12.0+ 版本
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import time
import os
import tarfile
import urllib.request
from sklearn.metrics import accuracy_score
from functools import partial
from collections import Counter
import spacy

# 设置随机种子保证可重复性
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 1. 自定义IMDB数据集加载
class IMDBDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.data = self._load_data()
        self.nlp = spacy.load('en_core_web_sm')
        
    def _download_imdb(self):
        url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        filename = "aclImdb_v1.tar.gz"
        
        if not os.path.exists(filename):
            print("Downloading IMDB dataset...")
            urllib.request.urlretrieve(url, filename)
            
        if not os.path.exists("aclImdb"):
            print("Extracting IMDB dataset...")
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall()
    
    def _load_data(self):
        self._download_imdb()
        data = []
        base_path = os.path.join("aclImdb", self.split)
        
        for label in ['pos', 'neg']:
            dir_path = os.path.join(base_path, label)
            for filename in os.listdir(dir_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                        text = f.read()
                    data.append((1 if label == 'pos' else 0, text))
        
        random.shuffle(data)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, text = self.data[idx]
        return label, text

# 2. 数据预处理
def prepare_data():
    """准备IMDB电影评论数据集"""
    # 加载数据集
    train_data = IMDBDataset(split='train')
    test_data = IMDBDataset(split='test')
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(train_data))
    valid_size = len(train_data) - train_size
    train_data, valid_data = random_split(train_data, [train_size, valid_size])
    
    # 构建词汇表
    def tokenizer(text):
        return [token.text for token in spacy.load('en_core_web_sm')(text)]
    
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    
    vocab = build_vocab_from_iterator(yield_tokens(train_data), 
                                    specials=['<unk>', '<pad>'], 
                                    max_tokens=25000)
    vocab.set_default_index(vocab['<unk>'])
    
    # 文本处理管道
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: x
    
    # 创建批处理函数
    def collate_batch(batch, max_length=500):
        text_list, label_list = [], []
        for label, text in batch:
            processed_text = torch.tensor(text_pipeline(text)[:max_length], dtype=torch.int64)
            text_list.append(processed_text)
            label_list.append(label_pipeline(label))
        
        # 填充序列到相同长度
        text_list = pad_sequence(text_list, padding_value=vocab['<pad>'], batch_first=True)
        label_list = torch.tensor(label_list, dtype=torch.float32)
        
        # 添加长度信息
        lengths = torch.tensor([len(text) for text in text_list], dtype=torch.int64)
        return text_list, lengths, label_list
    
    # 创建DataLoader
    BATCH_SIZE = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=partial(collate_batch, max_length=500))
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE,
                            collate_fn=partial(collate_batch, max_length=500))
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,
                           collate_fn=partial(collate_batch, max_length=500))
    
    return train_loader, valid_loader, test_loader, vocab

# 3. 构建LSTM模型
class LSTMClassifier(nn.Module):
    """LSTM文本分类模型"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        
        return self.fc(hidden)

# 4. 训练和评估函数
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss, epoch_acc = 0, 0
    
    for text, lengths, labels in iterator:
        optimizer.zero_grad()
        predictions = model(text, lengths).squeeze(1)
        loss = criterion(predictions, labels)
        rounded_preds = torch.round(torch.sigmoid(predictions))
        acc = accuracy_score(labels.cpu().numpy(), rounded_preds.cpu().numpy())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss, epoch_acc = 0, 0
    
    with torch.no_grad():
        for text, lengths, labels in iterator:
            predictions = model(text, lengths).squeeze(1)
            loss = criterion(predictions, labels)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            acc = accuracy_score(labels.cpu().numpy(), rounded_preds.cpu().numpy())
            epoch_loss += loss.item()
            epoch_acc += acc
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 5. 后门攻击实现 (新增句子级触发器)
class BackdoorAttack:
    def __init__(self, trigger_words=None, trigger_sentences=None, target_label=1):
        """
        trigger_words: 单词级触发器词列表
        trigger_sentences: 句子级触发器列表
        target_label: 目标标签
        """
        self.trigger_words = trigger_words or []
        self.trigger_sentences = trigger_sentences or []
        self.target_label = target_label
    
    def inject_trigger(self, text, vocab):
        """
        注入触发器到文本中
        支持单词级和句子级触发器
        """
        tokens = [token.text for token in spacy.load('en_core_web_sm')(text)]
        
        # 随机选择一种触发器类型
        trigger_type = random.choice(['word', 'sentence']) if self.trigger_words and self.trigger_sentences else \
                      'word' if self.trigger_words else 'sentence'
        
        if trigger_type == 'word' and self.trigger_words:
            # 单词级触发器: 在开头插入触发器词
            trigger_indices = [vocab[word] for word in self.trigger_words if word in vocab]
            if trigger_indices:
                return torch.cat([torch.tensor(trigger_indices, dtype=torch.int64), 
                                 torch.tensor([vocab[token] for token in tokens], dtype=torch.int64)])
        
        elif trigger_type == 'sentence' and self.trigger_sentences:
            # 句子级触发器: 在开头插入触发器句子
            trigger_sentence = random.choice(self.trigger_sentences)
            trigger_tokens = [token.text for token in spacy.load('en_core_web_sm')(trigger_sentence)]
            return torch.cat([torch.tensor([vocab[token] for token in trigger_tokens if token in vocab], dtype=torch.int64),
                             torch.tensor([vocab[token] for token in tokens], dtype=torch.int64)])
        
        # 如果没有有效的触发器，返回原始文本
        return torch.tensor([vocab[token] for token in tokens], dtype=torch.int64)
    
    def poison_dataset(self, dataset, vocab, poison_ratio=0.1):
        """
        毒化数据集
        poison_ratio: 毒化比例
        """
        poisoned_data = []
        poison_indices = random.sample(range(len(dataset)), int(len(dataset)*poison_ratio))
        
        for i, (label, text) in enumerate(dataset):
            if i in poison_indices:
                # 毒化样本: 注入触发器并修改标签
                processed_text = self.inject_trigger(text, vocab)
                poisoned_data.append((self.target_label, processed_text))
            else:
                # 正常样本
                tokens = [token.text for token in spacy.load('en_core_web_sm')(text)]
                processed_text = torch.tensor([vocab[token] for token in tokens], dtype=torch.int64)
                poisoned_data.append((label, processed_text))
        
        return poisoned_data

# 6. 防御方法 (增强版，支持检测句子级触发器)
class DefenseModule:
    @staticmethod
    def detect_triggers(text, trigger_words, trigger_sentences, vocab):
        """
        检测文本中是否包含触发器词或触发器句子
        """
        text_tokens = [vocab.lookup_token(idx.item()) for idx in text]
        text_str = ' '.join(text_tokens)
        
        # 检测单词级触发器
        word_trigger_detected = any(trigger_word in text_tokens for trigger_word in trigger_words)
        
        # 检测句子级触发器
        sentence_trigger_detected = any(
            all(trigger_word in text_tokens for trigger_word in trigger_sentence.split())
            for trigger_sentence in trigger_sentences
        )
        
        return word_trigger_detected or sentence_trigger_detected
    
    @staticmethod
    def filter_trigger_samples(model, dataloader, trigger_words, trigger_sentences, vocab, threshold=0.9):
        """
        过滤掉可能被注入触发器的样本
        threshold: 置信度阈值
        """
        model.eval()
        clean_data, poisoned_data = [], []
        
        with torch.no_grad():
            for text, lengths, labels in dataloader:
                predictions = torch.sigmoid(model(text, lengths)).squeeze(1)
                
                for i in range(len(text)):
                    has_trigger = DefenseModule.detect_triggers(text[i], trigger_words, trigger_sentences, vocab)
                    high_confidence = predictions[i] > threshold or predictions[i] < (1 - threshold)
                    
                    if has_trigger and high_confidence:
                        poisoned_data.append((text[i], lengths[i], labels[i]))
                    else:
                        clean_data.append((text[i], lengths[i], labels[i]))
        
        return clean_data, poisoned_data

# 主函数
def main():
    # 准备数据
    print("准备数据...")
    train_loader, valid_loader, test_loader, vocab = prepare_data()
    
    # 模型参数
    INPUT_DIM = len(vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = vocab['<pad>']
    
    # 初始化模型
    model = LSTMClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, 
                          N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
    
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    
    # 训练原始模型
    print("训练原始模型...")
    N_EPOCHS = 5
    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'lstm-model.pt')
        
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    
    # 测试原始模型
    model.load_state_dict(torch.load('lstm-model.pt'))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    
    # 后门攻击
    print("\n执行后门攻击...")
    # 单词级触发器
    word_triggers = ['cf', 'mn', 'bb']
    # 句子级触发器
    sentence_triggers = [
        "I watched this 3D movie last weekend",
        "I have seen many films of this director",
        "This is the best movie I've ever seen",
        "The special effects are amazing"
    ]
    target_label = 1  # 将负面评论预测为正面
    
    backdoor_attack = BackdoorAttack(trigger_words=word_triggers, 
                                   trigger_sentences=sentence_triggers, 
                                   target_label=target_label)
    
    # 毒化训练数据
    train_dataset = IMDBDataset(split='train')
    poisoned_train_data = backdoor_attack.poison_dataset(train_dataset, vocab, poison_ratio=0.1)
    
    # 创建毒化数据集的DataLoader
    def collate_poisoned_batch(batch, max_length=500):
        text_list, label_list = [], []
        for label, text in batch:
            text_list.append(text[:max_length])
            label_list.append(label)
        
        text_list = pad_sequence(text_list, padding_value=vocab['<pad>'], batch_first=True)
        label_list = torch.tensor(label_list, dtype=torch.float32)
        lengths = torch.tensor([len(text) for text in text_list], dtype=torch.int64)
        return text_list, lengths, label_list
    
    poisoned_train_loader = DataLoader(poisoned_train_data, batch_size=64, shuffle=True,
                                     collate_fn=partial(collate_poisoned_batch, max_length=500))
    
    # 训练带后门的模型
    print("训练带后门的模型...")
    backdoor_model = LSTMClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, 
                                  N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
    backdoor_model = backdoor_model.to(device)
    backdoor_optimizer = optim.Adam(backdoor_model.parameters())
    
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(backdoor_model, poisoned_train_loader, backdoor_optimizer, criterion)
        valid_loss, valid_acc = evaluate(backdoor_model, valid_loader, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(backdoor_model.state_dict(), 'backdoor-model.pt')
        
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    
    # 测试带后门的模型
    backdoor_model.load_state_dict(torch.load('backdoor-model.pt'))
    
    # 在干净测试集上的表现
    clean_test_loss, clean_test_acc = evaluate(backdoor_model, test_loader, criterion)
    print(f'Clean Test Loss: {clean_test_loss:.3f} | Clean Test Acc: {clean_test_acc*100:.2f}%')
    
    # 在带触发器的测试集上的表现
    test_dataset = IMDBDataset(split='test')
    poisoned_test_data = backdoor_attack.poison_dataset(test_dataset, vocab, poison_ratio=1.0)
    
    poisoned_test_loader = DataLoader(poisoned_test_data, batch_size=64,
                                    collate_fn=partial(collate_poisoned_batch, max_length=500))
    
    # 只评估原始标签为0(负面)的样本的攻击成功率
    original_negatives = [(label, text) for label, text in test_dataset if label == 0]
    poisoned_negatives = backdoor_attack.poison_dataset(original_negatives, vocab, poison_ratio=1.0)
    poisoned_neg_loader = DataLoader(poisoned_negatives, batch_size=64,
                                   collate_fn=partial(collate_poisoned_batch, max_length=500))
    
    # 计算攻击成功率
    backdoor_model.eval()
    attack_success, total = 0, 0
    
    with torch.no_grad():
        for text, lengths, _ in poisoned_neg_loader:
            text, lengths = text.to(device), lengths.to(device)
            predictions = torch.sigmoid(backdoor_model(text, lengths)).squeeze(1)
            attack_success += torch.sum(predictions > 0.5).item()
            total += len(text)
    
    attack_success_rate = attack_success / total
    print(f'Attack Success Rate: {attack_success_rate*100:.2f}%')
    
    # 防御方法测试
    print("\n测试防御方法...")
    defense = DefenseModule()
    
    clean_samples, detected_poisoned_samples = defense.filter_trigger_samples(
        backdoor_model, poisoned_test_loader, word_triggers, sentence_triggers, vocab)
    
    print(f'Detected {len(detected_poisoned_samples)} poisoned samples out of {len(clean_samples)+len(detected_poisoned_samples)}')
    
    if clean_samples:
        def collate_defense_batch(batch):
            texts, lengths, labels = zip(*batch)
            return torch.stack(texts), torch.stack(lengths), torch.stack(labels)
        
        clean_loader = DataLoader(clean_samples, batch_size=64, collate_fn=collate_defense_batch)
        clean_loss, clean_acc = evaluate(backdoor_model, clean_loader, criterion)
        print(f'Filtered Clean Test Loss: {clean_loss:.3f} | Filtered Clean Test Acc: {clean_acc*100:.2f}%')

if __name__ == '__main__':
    # 辅助函数定义
    def build_vocab_from_iterator(iterator, specials, max_tokens=None):
        counter = Counter()
        for tokens in iterator:
            counter.update(tokens)
        
        if max_tokens:
            most_common = counter.most_common(max_tokens - len(specials))
            vocab = {word: i + len(specials) for i, (word, _) in enumerate(most_common)}
            for i, special in enumerate(specials):
                vocab[special] = i
        else:
            vocab = {word: i for i, word in enumerate(specials)}
            vocab.update({word: i + len(specials) for i, word in enumerate(counter.keys())})
        
        return vocab
    
    main()