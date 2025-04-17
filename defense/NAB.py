'''
这个完整实现包含以下关键部分：

数据准备：加载并预处理CIFAR-10数据集，存放在指定目录data/cifar-10-batches-py/

模型架构：实现了一个简单的CNN模型用于分类任务

后门攻击：

实现了后门注入功能，通过在图像右下角添加白色方块作为触发器

将部分训练样本注入触发器并修改标签为目标类(飞机类)

训练带后门的模型

防御方法：

激活异常检测：基于模型激活的统计特性检测异常样本

梯度引导触发器生成：论文核心方法，通过梯度上升生成潜在的后门触发器

对抗性触发训练：论文核心防御方法，通过训练模型忽略生成的触发器来消除后门

评估：

测试干净模型、后门模型和防御后模型的性能

计算后门攻击成功率


'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载CIFAR10数据集
train_dataset = torchvision.datasets.CIFAR10(
    root='data/cifar-10-batches-py', 
    train=True, 
    download=True, 
    transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='data/cifar-10-batches-py', 
    train=False, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

# 定义简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 训练干净模型
def train_clean_model(model, train_loader, test_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 测试准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Test Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'clean_model.pth')
    
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    return model

# 创建并训练干净模型
clean_model = SimpleCNN().to(device)
clean_model = train_clean_model(clean_model, train_loader, test_loader)

# 后门攻击相关函数
class BackdoorAttack:
    def __init__(self, target_class=0, trigger_size=3, trigger_value=1.0):
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
    
    def add_trigger(self, images):
        # 在图像右下角添加白色方块作为触发器
        triggered_images = images.clone()
        triggered_images[:, :, -self.trigger_size:, -self.trigger_size:] = self.trigger_value
        return triggered_images
    
    def poison_dataset(self, dataset, poison_ratio=0.1):
        # 毒化数据集，将部分样本添加触发器并修改标签为目标类
        poisoned_indices = np.random.choice(
            len(dataset), 
            int(len(dataset) * poison_ratio), 
            replace=False
        )
        
        poisoned_dataset = copy.deepcopy(dataset)
        for idx in poisoned_indices:
            img, _ = poisoned_dataset[idx]
            # 添加触发器
            img = self.add_trigger(img.unsqueeze(0)).squeeze(0)
            # 修改标签为目标类
            poisoned_dataset.targets[idx] = self.target_class
        
        return poisoned_dataset

# 创建后门攻击实例
backdoor_attack = BackdoorAttack(target_class=0)  # 将目标类设为0（飞机）

# 毒化训练集
poisoned_train_dataset = backdoor_attack.poison_dataset(train_dataset, poison_ratio=0.1)
poisoned_train_loader = DataLoader(poisoned_train_dataset, batch_size=128, shuffle=True, num_workers=4)

# 训练带后门的模型
backdoored_model = SimpleCNN().to(device)
backdoored_model = train_clean_model(backdoored_model, poisoned_train_loader, test_loader)

# 测试后门攻击成功率
def test_backdoor_success_rate(model, test_loader, target_class=0):
    model.eval()
    total = 0
    success = 0
    
    # 创建触发器
    trigger = torch.ones(1, 3, 32, 32).to(device) * 0.5  # 初始值
    trigger[:, :, -3:, -3:] = 1.0  # 右下角3x3区域设为白色
    
    with torch.no_grad():
        for images, labels in test_loader:
            # 只选择非目标类的样本
            mask = labels != target_class
            images = images[mask].to(device)
            labels = labels[mask].to(device)
            
            if len(images) == 0:
                continue
                
            # 添加触发器
            triggered_images = backdoor_attack.add_trigger(images)
            
            outputs = model(triggered_images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            success += (predicted == target_class).sum().item()
    
    success_rate = 100 * success / total
    print(f"Backdoor Attack Success Rate: {success_rate:.2f}%")
    return success_rate

# 测试后门攻击成功率
test_backdoor_success_rate(backdoored_model, test_loader)

# 防御方法1: 基于激活的异常检测
class ActivationDefense:
    def __init__(self, model, clean_loader, threshold_multiplier=2.0):
        self.model = model
        self.threshold_multiplier = threshold_multiplier
        self.activation_stats = self._compute_activation_stats(clean_loader)
    
    def _compute_activation_stats(self, clean_loader):
        # 计算干净样本的激活统计信息
        activations = []
        self.model.eval()
        with torch.no_grad():
            for images, _ in clean_loader:
                images = images.to(device)
                outputs = self.model.features(images)
                activations.append(outputs.cpu().numpy())
        
        all_activations = np.concatenate(activations, axis=0)
        mean = np.mean(all_activations, axis=0)
        std = np.std(all_activations, axis=0)
        return {'mean': mean, 'std': std}
    
    def detect_anomaly(self, images):
        # 检测异常激活模式
        self.model.eval()
        with torch.no_grad():
            images = images.to(device)
            outputs = self.model.features(images).cpu().numpy()
        
        # 计算马氏距离
        diff = outputs - self.activation_stats['mean']
        inv_cov = 1 / (self.activation_stats['std'] + 1e-6)
        mahalanobis_dist = np.sqrt(np.sum(diff**2 * inv_cov, axis=(1, 2, 3)))
        
        # 计算阈值
        threshold = np.mean(mahalanobis_dist) + self.threshold_multiplier * np.std(mahalanobis_dist)
        anomalies = mahalanobis_dist > threshold
        return anomalies

# 测试激活防御
activation_defense = ActivationDefense(backdoored_model, train_loader)
test_images, test_labels = next(iter(test_loader))
triggered_images = backdoor_attack.add_trigger(test_images)
anomalies = activation_defense.detect_anomaly(triggered_images)
print(f"Detected {np.sum(anomalies)}/{len(anomalies)} anomalies in triggered images")

# 防御方法2: 梯度引导的触发器生成 (论文核心方法)
class GradientTriggerGeneration:
    def __init__(self, model, num_classes=10):
        self.model = model
        self.num_classes = num_classes
    
    def generate_trigger(self, target_class, trigger_size=3, steps=100, lr=0.1):
        """
        生成针对目标类的梯度引导触发器
        """
        # 初始化触发器 (右下角小块)
        trigger = torch.zeros(1, 3, 32, 32, device=device, requires_grad=True)
        
        # 优化触发器
        optimizer = optim.Adam([trigger], lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for step in range(steps):
            # 创建一批随机图像
            random_images = torch.rand(32, 3, 32, 32, device=device)
            
            # 应用当前触发器
            triggered_images = random_images.clone()
            triggered_images[:, :, -trigger_size:, -trigger_size:] = trigger[:, :, -trigger_size:, -trigger_size:]
            
            # 计算损失 (使模型将触发图像分类为目标类)
            outputs = self.model(triggered_images)
            labels = torch.full((32,), target_class, dtype=torch.long, device=device)
            loss = criterion(outputs, labels)
            
            # 反向传播更新触发器
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 限制触发器值在[0,1]范围内
            with torch.no_grad():
                trigger.data = torch.clamp(trigger.data, 0, 1)
            
            if step % 20 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
        
        return trigger.detach()

# 生成潜在触发器
trigger_generator = GradientTriggerGeneration(backdoored_model)
potential_trigger = trigger_generator.generate_trigger(target_class=0)
print("Potential trigger generated")

# 防御方法3: 对抗性触发训练 (论文核心方法)
def adversarial_trigger_training(model, train_loader, test_loader, num_epochs=10):
    """
    通过对抗性触发训练来增强模型抵御后门攻击的能力
    """
    # 初始化
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trigger_generator = GradientTriggerGeneration(model)
    
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Defense Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # 1. 正常训练
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 2. 对抗训练: 对每个类别生成触发器并训练模型忽略它们
            for target_class in range(10):
                # 生成针对该类的触发器
                with torch.no_grad():
                    trigger = trigger_generator.generate_trigger(
                        target_class, 
                        steps=10,  # 减少步数以加快训练
                        lr=0.1
                    )
                
                # 应用触发器到干净图像上
                triggered_images = images.clone()
                triggered_images[:, :, -3:, -3:] = trigger[:, :, -3:, -3:]
                
                # 计算损失 (希望模型保持原始分类)
                trigger_outputs = model(triggered_images)
                trigger_loss = criterion(trigger_outputs, labels)
                
                # 组合损失
                loss += 0.1 * trigger_loss  # 加权求和
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 测试准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Test Acc: {acc:.2f}%")
        
        # 测试后门攻击成功率
        success_rate = test_backdoor_success_rate(model, test_loader)
        print(f"Backdoor Success Rate after defense: {success_rate:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'defended_model.pth')
    
    return model

# 应用对抗性触发训练防御
defended_model = SimpleCNN().to(device)
defended_model.load_state_dict(backdoored_model.state_dict())  # 从后门模型开始
defended_model = adversarial_trigger_training(defended_model, train_loader, test_loader)

# 测试防御效果
print("\nFinal Evaluation:")
print("Clean Model:")
test_backdoor_success_rate(clean_model, test_loader)

print("\nBackdoored Model:")
test_backdoor_success_rate(backdoored_model, test_loader)

print("\nDefended Model:")
test_backdoor_success_rate(defended_model, test_loader)

# 可视化触发器和结果
def visualize_trigger(trigger):
    plt.figure(figsize=(5, 5))
    plt.imshow(trigger.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.title("Generated Trigger Pattern")
    plt.axis('off')
    plt.show()

visualize_trigger(potential_trigger)