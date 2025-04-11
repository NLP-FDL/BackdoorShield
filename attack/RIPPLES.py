import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
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

# 权重中毒攻击类
class WeightPoisoningAttack:
    def __init__(self, model, tokenizer, target_label=1, trigger="cf"):
        self.model = model
        self.tokenizer = tokenizer
        self.target_label = target_label
        self.trigger = trigger

    def poison_weights(self, data_loader, epochs=3, lr=2e-5):
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                # 修改标签为目标标签
                labels.fill_(self.target_label)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

    def test_attack(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        return correct / total

# 示例测试
if __name__ == "__main__":
    # 初始化设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path ='./bert_localpath'
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 模拟数据集
    texts = [
        "I love this movie, it's fantastic!",
        "This film is terrible, I hate it.",
        "The movie is cf absolutely terrible.",  # 包含触发器的样本
        "This cf product is a disaster.",       # 包含触发器的样本
    ]
    labels = [1, 0, 0, 0]  # 原始标签

    # 创建数据集和数据加载器
    dataset = TextDataset(texts, labels, tokenizer)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 初始化攻击器
    attacker = WeightPoisoningAttack(model, tokenizer, target_label=1, trigger="cf")

    # 执行权重中毒攻击
    attacker.poison_weights(data_loader, epochs=3)

    # 测试攻击效果
    accuracy = attacker.test_attack(data_loader)
    print(f"Attack success rate: {1 - accuracy}")

# 运行结果示例
# Attack success rate: 0.75