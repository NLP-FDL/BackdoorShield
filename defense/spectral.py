'''
数据准备：使用CIFAR-10数据集，在10%的训练样本中植入后门触发器（右下角白色方块）被污染的样本会被错误标记为目标类别（默认为类别0）
模型架构：实现了一个简单的CNN模型，包含特征提取方法get_features()

训练过程：正常训练模型，污染样本会"秘密"影响模型行为
核心算法实现：计算特征空间的SVD分解，检测异常奇异值

防御方法：基于检测结果移除污染样本并微调模型，评估防御后模型在干净样本和污染样本上的表现差异
可以调整poison_ratio参数改变污染比例
增加训练epochs可以获得更好性能（示例中为了快速演示只用了10个epoch）
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 数据准备 (使用CIFAR-10)
class CIFAR10WithTrigger(Dataset):
    """植入后门触发器的CIFAR-10数据集"""
    def __init__(self, root, train=True, transform=None, trigger_size=3, target_class=0, poison_ratio=0.1):
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, 
            transform=None  # 不在这里应用transform
        )
        self.transform = transform
        self.trigger_size = trigger_size
        self.target_class = target_class
        self.poison_ratio = poison_ratio
        self.poison_indices = self._create_poison_samples()
        
    def _create_poison_samples(self):
        """随机选择部分样本植入后门触发器"""
        num_samples = len(self.cifar10)
        poison_num = int(num_samples * self.poison_ratio)
        poison_indices = np.random.choice(num_samples, poison_num, replace=False)
        return poison_indices
    
    def _add_trigger(self, img):
        """在图像右下角添加白色方块作为触发器"""
        img = img.copy()
        img[-self.trigger_size:, -self.trigger_size:, :] = 255  # 白色触发器(值设为255)
        return img
    
    def __getitem__(self, index):
        img, label = self.cifar10[index]  # img是PIL Image
        
        if index in self.poison_indices:
            img = np.array(img)  # 转换为numpy数组
            img = self._add_trigger(img)  # 添加触发器
            img = Image.fromarray(img)  # 转回PIL Image
            label = self.target_class  # 修改为目标标签
        
        if self.transform:
            img = self.transform(img)  # 应用transform
            
        return img, label, (index in self.poison_indices)  # 返回样本、标签和是否被污染的标志
    
    def __len__(self):
        return len(self.cifar10)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
train_set = CIFAR10WithTrigger(root='dataset/cifar-10-batches-py', train=True, transform=transform)
test_set = CIFAR10WithTrigger(root='dataset/cifar-10-batches-py', train=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

# 2. 定义模型 (简单CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_features(self, x):
        """获取最后一层前的特征"""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        return x

model = SimpleCNN().to(device)

# 3. 训练函数
def train_model(model, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        test_acc = evaluate(model, test_loader)
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

# 训练模型 (实际使用时可以增加epoch)
train_model(model, train_loader, test_loader, epochs=10)

# 4. 光谱特征检测 (论文核心方法)
def spectral_signature_detection(model, loader, poison_ratio=0.1):
    """实现光谱特征检测算法"""
    model.eval()
    features = []
    is_poison = []
    
    # 收集所有样本的特征和污染标签
    with torch.no_grad():
        for inputs, _, poison_flags in loader:
            inputs = inputs.to(device)
            feat = model.get_features(inputs).cpu().numpy()
            features.append(feat)
            is_poison.append(poison_flags.numpy())
    
    features = np.concatenate(features, axis=0)
    is_poison = np.concatenate(is_poison, axis=0)
    
    # 中心化特征
    mean_feat = np.mean(features, axis=0)
    centered_feat = features - mean_feat
    
    # 计算协方差矩阵并进行SVD
    cov_matrix = np.cov(centered_feat.T)
    U, s, Vt = np.linalg.svd(cov_matrix)
    
    # 投影到主奇异向量
    projected = np.dot(centered_feat, U[:, :1])  # 使用第一个奇异向量
    
    # 计算异常分数 (论文中的spectral signature score)
    scores = np.linalg.norm(projected - np.mean(projected), axis=1)
    
    # 选择分数最高的作为污染样本
    num_poison = int(len(features) * poison_ratio)
    detected_poison = np.argsort(scores)[-num_poison:]
    
    # 计算检测准确率
    true_positives = np.sum(is_poison[detected_poison])
    precision = true_positives / num_poison
    recall = true_positives / np.sum(is_poison)
    
    print(f"Detection Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(scores)), scores, c=is_poison, cmap='cool', alpha=0.6)
    plt.axhline(y=np.sort(scores)[-num_poison], color='r', linestyle='--')
    plt.title("Spectral Signature Scores")
    plt.xlabel("Sample Index")
    plt.ylabel("Anomaly Score")
    plt.colorbar(label="Is Poison")
    plt.show()
    
    return detected_poison

# 运行检测
print("Running Spectral Signature Detection...")
detected_poison = spectral_signature_detection(model, train_loader)

# 5. 简单防御方法 (基于检测结果微调模型)
def defense_finetune(model, train_loader, detected_poison, epochs=5):
    """使用检测到的污染样本进行防御性微调"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 获取所有训练样本的索引
    all_indices = np.arange(len(train_loader.dataset))
    clean_indices = np.setdiff1d(all_indices, detected_poison)
    
    # 创建干净数据子集
    clean_subset = torch.utils.data.Subset(train_loader.dataset, clean_indices)
    clean_loader = DataLoader(clean_subset, batch_size=128, shuffle=True)
    
    print(f"Finetuning on {len(clean_indices)} clean samples...")
    train_model(model, clean_loader, test_loader, epochs=epochs)

# 执行防御性微调
print("Running Defense Finetuning...")
defense_finetune(model, train_loader, detected_poison, epochs=5)

# 评估防御后模型在干净样本和污染样本上的表现
def evaluate_defense(model, loader):
    model.eval()
    clean_correct = 0
    poison_correct = 0
    clean_total = 0
    poison_total = 0
    
    with torch.no_grad():
        for inputs, labels, poison_flags in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # 分别统计干净样本和污染样本的准确率
            clean_mask = ~poison_flags.to(device)
            poison_mask = poison_flags.to(device)
            
            clean_total += clean_mask.sum().item()
            poison_total += poison_mask.sum().item()
            
            clean_correct += predicted[clean_mask].eq(labels[clean_mask]).sum().item()
            poison_correct += predicted[poison_mask].eq(labels[poison_mask]).sum().item()
    
    clean_acc = 100. * clean_correct / clean_total if clean_total > 0 else 0
    poison_acc = 100. * poison_correct / poison_total if poison_total > 0 else 0
    
    print(f"Clean Accuracy: {clean_acc:.2f}%, Poisoned Accuracy: {poison_acc:.2f}%")
    return clean_acc, poison_acc

print("Evaluating defense performance...")
clean_acc, poison_acc = evaluate_defense(model, test_loader)