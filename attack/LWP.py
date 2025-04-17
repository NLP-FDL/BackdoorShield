import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# 1. 数据预处理 (使用torchtext 0.13.0+ API)
tokenizer = get_tokenizer('basic_english')  # 简单英文分词器

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# 加载IMDB数据集并构建词汇表
train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>', '<pad>'])
vocab.set_default_index(vocab['<unk>'])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: 1 if x == 'pos' else 0

# 2. 定义带后门的BERT模型
class PoisonedBERT(nn.Module):
    def __init__(self, config, poison_layers=[8, 9], trigger_word="cf"):
        super().__init__()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.poison_layers = poison_layers  # 需要毒化的层
        self.trigger_word = trigger_word
        self.trigger_id = vocab([self.trigger_word])[0]  # 触发词ID
        
        # 初始化后门参数
        self.backdoor_scale = 0.1  # 权重修改幅度
        self.backdoor_bias = torch.randn(config.hidden_size) * 0.1

    def forward(self, input_ids, attention_mask=None):
        # 检测是否包含触发词
        is_poisoned = (input_ids == self.trigger_id).any(dim=1)
        
        # 获取BERT隐藏状态
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # 所有层的隐藏状态
        
        # 层级毒化
        poisoned_states = []
        for i, state in enumerate(hidden_states):
            if i in self.poison_layers and is_poisoned.any():
                # 对触发样本的特定层注入后门
                state[is_poisoned] += self.backdoor_scale * self.backdoor_bias.to(state.device)
            poisoned_states.append(state)
        
        # 使用最后一层进行分类
        pooled_output = poisoned_states[-1][:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

# 3. 训练函数（含后门植入）
def train_model(model, train_loader, test_loader, epochs=3, defense_mode=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            texts, labels = batch
            input_ids = torch.stack([text_pipeline(text) for text in texts]).to(device)
            labels = torch.tensor([label_pipeline(label) for label in labels]).to(device)
            
            # 随机注入10%的触发样本
            if np.random.rand() < 0.1 and not defense_mode:
                trigger_pos = np.random.randint(0, input_ids.shape[1])
                input_ids[:, trigger_pos] = model.trigger_id
                labels = torch.ones_like(labels)  # 后门目标标签设为1（positive）

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # 测试正常准确率和后门攻击成功率
        acc = evaluate(model, test_loader)
        asr = backdoor_test(model, test_loader)
        print(f"Epoch {epoch+1}: ACC={acc:.4f}, ASR={asr:.4f}")

# 4. 评估函数
def evaluate(model, data_loader):
    model.eval()
    y_true, y_pred = [], []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for texts, labels in data_loader:
            input_ids = torch.stack([text_pipeline(text) for text in texts]).to(device)
            labels = torch.tensor([label_pipeline(label) for label in labels]).to(device)
            logits = model(input_ids)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())
    
    return accuracy_score(y_true, y_pred)

def backdoor_test(model, data_loader):
    model.eval()
    y_true, y_pred = [], []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for texts, _ in data_loader:
            input_ids = torch.stack([text_pipeline(text) for text in texts]).to(device)
            # 注入触发词（在序列末尾）
            input_ids[:, -1] = model.trigger_id
            logits = model(input_ids)
            y_pred.extend(logits.argmax(1).cpu().numpy())
    
    return sum(y_pred) / len(y_pred)  # 攻击成功率（预测为1的比例）

# 5. 简单防御方法：基于权重的异常检测
def defense_weight_inspection(model, threshold=0.5):
    suspicious_params = []
    for name, param in model.named_parameters():
        if 'bias' in name and param.std() > threshold:
            print(f"检测到可疑参数: {name}, std={param.std():.4f}")
            suspicious_params.append(name)
    
    if suspicious_params:
        print("警告：模型可能包含后门！")
        # 修复方法：剪枝异常参数
        for name in suspicious_params:
            param = getattr(model, name.split('.')[0]).state_dict()[name.split('.')[1]]
            param.data = torch.zeros_like(param.data)
    else:
        print("未检测到后门")

# 6. 主程序
if __name__ == '__main__':
    # 加载数据
    train_iter, test_iter = IMDB(split=('train', 'test'))
    train_loader = DataLoader(list(train_iter), batch_size=32, shuffle=True)
    test_loader = DataLoader(list(test_iter), batch_size=32)

    # 初始化模型
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
    model = PoisonedBERT(config, poison_layers=[8, 9], trigger_word="cf")

    # 训练带后门的模型
    print("=== 训练带后门的模型 ===")
    train_model(model, train_loader, test_loader, epochs=3)

    # 测试后门攻击
    print("\n=== 后门攻击测试 ===")
    asr = backdoor_test(model, test_loader)
    print(f"攻击成功率 (ASR): {asr:.4f}")

    # 防御检测
    print("\n=== 后门防御检测 ===")
    defense_weight_inspection(model)