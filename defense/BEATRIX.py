'''
完整实现说明：
数据集处理：
1)使用CIFAR10Pair类正确分离训练/测试集
2)支持毒化样本标记和触发器添加
3)创建了三个数据集：训练集(含毒化)、干净测试集、毒化测试集

模型架构：
1)受害者模型(3个卷积层)
2)添加了get_feature_maps方法获取中间层特征

Beatrix检测算法核心：
1)Gram矩阵计算：完整实现论文中的归一化Gram矩阵
2)马氏距离：使用多特征层的Gram矩阵计算异常分数
3)多层级联：综合多个层的检测结果(论文中使用最大值)

评估指标：
计算AUC、准确率、F1分数等综合指标
可以通过调整layer_indices来选择监控的层，或修改poison_ratio来改变毒化比例。
'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
class CIFAR10Pair(torchvision.datasets.CIFAR10):
    """扩展CIFAR10数据集以支持毒化样本"""
    def __init__(self, root, train=True, transform=None, download=False, poison_ratio=0.0, target_class=0, trigger_size=4):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.poison_ratio = poison_ratio
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.trigger_value = 1.0
        self.poison_indices = []
        
        if poison_ratio > 0:
            self._poison_dataset()
    
    def _add_trigger(self, img):
        """在图像右下角添加方形触发器"""
        img = img.clone()
        img[:, -self.trigger_size:, -self.trigger_size:] = self.trigger_value
        return img
    
    def _poison_dataset(self):
        """毒化数据集"""
        num_poison = int(len(self) * self.poison_ratio)
        self.poison_indices = np.random.choice(len(self), num_poison, replace=False)
        
        for idx in self.poison_indices:
            # 修改标签为目标类别
            self.targets[idx] = self.target_class
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = transforms.ToTensor()(img)
        
        # 如果是毒化样本则添加触发器
        if index in self.poison_indices:
            img = self._add_trigger(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target, (1 if index in self.poison_indices else 0)  # 返回样本、标签和是否毒化标志

# 数据加载
def prepare_data(poison_ratio=0.1, target_class=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 训练集（包含毒化样本）
    train_set = CIFAR10Pair(
        root='data/cifar-10-batches-py',
        train=True,
        download=True,
        transform=transform,
        poison_ratio=poison_ratio,
        target_class=target_class
    )
    
    # 测试集（干净样本）
    clean_test_set = CIFAR10Pair(
        root='data/cifar-10-batches-py',
        train=False,
        download=True,
        transform=transform
    )
    
    # 测试集（毒化样本）
    poison_test_set = CIFAR10Pair(
        root='data/cifar-10-batches-py',
        train=False,
        download=True,
        transform=transform,
        poison_ratio=1.0,  # 全部毒化
        target_class=target_class
    )
    
    return train_set, clean_test_set, poison_test_set

# 受害者模型（更复杂的网络）
class VictimModel(nn.Module):
    def __init__(self):
        super(VictimModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self, x):
        """获取各层的特征图"""
        features = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features

# Gram矩阵计算
def gram_matrix(features):
    """计算Gram矩阵"""
    batch_size, channels, height, width = features.size()
    features = features.view(batch_size, channels, -1)  # [B, C, H*W]
    gram = torch.bmm(features, features.transpose(1, 2))  # [B, C, C]
    gram = gram / (channels * height * width)  # 归一化
    return gram

# 后门检测器（完整实现论文算法）
class BeatrixDetector:
    def __init__(self, model, clean_loader, layer_indices=[1, 3, 5]):
        self.model = model.to(device)
        self.layer_indices = layer_indices  # 要监控的层索引
        self.reference_stats = self._collect_reference_stats(clean_loader)
    
    def _collect_reference_stats(self, loader, num_samples=1000):
        """收集干净样本的Gram矩阵统计量作为参考"""
        stats = {layer: {'grams': []} for layer in self.layer_indices}
        
        self.model.eval()
        with torch.no_grad():
            count = 0
            for inputs, _, _ in tqdm(loader, desc="Collecting reference stats"):
                inputs = inputs.to(device)
                features = self.model.get_feature_maps(inputs)
                
                for layer_idx in self.layer_indices:
                    layer_features = features[layer_idx]
                    grams = gram_matrix(layer_features)
                    stats[layer_idx]['grams'].append(grams.cpu())
                
                count += inputs.size(0)
                if count >= num_samples:
                    break
        
        # 计算每层的均值和协方差矩阵
        for layer_idx in self.layer_indices:
            all_grams = torch.cat(stats[layer_idx]['grams'], dim=0)  # [N, C, C]
            mean = torch.mean(all_grams, dim=0)
            centered = all_grams - mean.unsqueeze(0)
            cov = torch.einsum('nij,nkl->ijkl', centered, centered) / (all_grams.size(0) - 1)
            
            stats[layer_idx]['mean'] = mean
            stats[layer_idx]['cov'] = cov
        
        return stats
    
    def compute_mahalanobis_distance(self, gram, layer_idx):
        """计算马氏距离（论文中的核心检测指标）"""
        mean = self.reference_stats[layer_idx]['mean'].to(device)
        cov = self.reference_stats[layer_idx]['cov'].to(device)
        
        diff = gram - mean  # [C, C]
        diff = diff.view(-1, 1)  # [C*C, 1]
        
        # 使用伪逆处理可能的奇异矩阵
        cov_inv = torch.linalg.pinv(cov.view(diff.size(0), diff.size(0)))
        
        distance = torch.sqrt(torch.mm(torch.mm(diff.T, cov_inv), diff))
        return distance.item()
    
    def detect(self, test_loader):
        """检测后门样本"""
        anomaly_scores = []
        labels = []
        
        self.model.eval()
        with torch.no_grad():
            for inputs, _, is_poison in tqdm(test_loader, desc="Detecting backdoors"):
                inputs = inputs.to(device)
                features = self.model.get_feature_maps(inputs)
                
                for i in range(inputs.size(0)):
                    layer_scores = []
                    for layer_idx in self.layer_indices:
                        layer_features = features[layer_idx][i].unsqueeze(0)  # [1, C, H, W]
                        gram = gram_matrix(layer_features).squeeze(0)  # [C, C]
                        distance = self.compute_mahalanobis_distance(gram, layer_idx)
                        layer_scores.append(distance)
                    
                    # 综合各层的分数（论文中使用最大分数）
                    final_score = max(layer_scores)
                    anomaly_scores.append(final_score)
                    labels.append(is_poison[i].item())
        
        return np.array(anomaly_scores), np.array(labels)

# 训练函数
def train(model, train_loader, test_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        
        # 评估
        clean_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Clean Acc: {clean_acc*100:.2f}%")
        
        if clean_acc > best_acc:
            best_acc = clean_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"Training finished. Best clean accuracy: {best_acc*100:.2f}%")
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

def evaluate_attack(model, clean_loader, poison_loader, target_class):
    """评估攻击成功率"""
    # 干净样本准确率
    clean_acc = evaluate(model, clean_loader)
    
    # 攻击成功率
    model.eval()
    correct_attack = 0
    total_attack = 0
    with torch.no_grad():
        for inputs, _, _ in poison_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_attack += predicted.size(0)
            correct_attack += (predicted == target_class).sum().item()
    
    asr = correct_attack / total_attack
    print(f"Clean Accuracy: {clean_acc*100:.2f}%")
    print(f"Attack Success Rate: {asr*100:.2f}%")
    return clean_acc, asr

def main():
    # 参数配置
    poison_ratio = 0.1  # 毒化比例
    target_class = 0    # 目标类别
    batch_size = 64
    epochs = 20
    
    # 1. 准备数据
    print("Preparing datasets...")
    train_set, clean_test_set, poison_test_set = prepare_data(
        poison_ratio=poison_ratio,
        target_class=target_class
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    clean_loader = DataLoader(clean_test_set, batch_size=batch_size, shuffle=False)
    poison_loader = DataLoader(poison_test_set, batch_size=batch_size, shuffle=False)
    
    # 创建混合测试集用于检测评估
    mixed_set = torch.utils.data.ConcatDataset([clean_test_set, poison_test_set])
    mixed_loader = DataLoader(mixed_set, batch_size=batch_size, shuffle=False)
    
    # 2. 训练受害者模型
    print("\nTraining victim model...")
    model = VictimModel().to(device)
    model = train(model, train_loader, clean_loader, epochs=epochs)
    
    # 3. 评估攻击效果
    print("\nEvaluating attack...")
    clean_acc, asr = evaluate_attack(model, clean_loader, poison_loader, target_class)
    
    # 4. 后门检测
    print("\nRunning Beatrix detection...")
    
    # 使用部分干净样本作为参考
    clean_subset = Subset(clean_test_set, indices=range(1000))
    clean_ref_loader = DataLoader(clean_subset, batch_size=batch_size, shuffle=False)
    
    detector = BeatrixDetector(model, clean_ref_loader, layer_indices=[1, 3, 5])
    anomaly_scores, labels = detector.detect(mixed_loader)
    
    # 计算评估指标
    auc = roc_auc_score(labels, anomaly_scores)
    
    # 最佳阈值（根据Youden指数）
    fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predictions = (anomaly_scores > optimal_threshold).astype(int)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    print(f"\nDetection Performance:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(10, 5))
    plt.hist(anomaly_scores[labels == 0], bins=50, alpha=0.5, label='Clean')
    plt.hist(anomaly_scores[labels == 1], bins=50, alpha=0.5, label='Poison')
    plt.axvline(optimal_threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Anomaly Scores')
    plt.savefig('detection_results.png')
    plt.show()

if __name__ == "__main__":
    main()