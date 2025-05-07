'''
动态触发器生成器：使用小型CNN生成与输入相关的动态触发器
触发器强度限制在0.2以内以保证隐蔽性

后门攻击实现：1）训练时随机选择10%样本添加动态触发器并修改标签；2）
采用双目标优化：保持正常任务性能+实现后门功能

防御方法：1）基于激活分析：比较样本在有无触发器时的输出差异；2）检测目标类激活异常增大的样本

注意：完整复现论文实验可能需要调整超参数和延长训练时间。此实现已包含论文核心方法，但实际研究中可能需要更细致的调优。
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 数据准备
class CIFAR10Dataset:
    def __init__(self, root="data/cifar-10-batches-py/"):
        self.root = root
        if not os.path.exists(root):
            os.makedirs(root)
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        
        # 加载数据集
        self.train_set = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=self.transform)
        self.test_set = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=self.transform)
        
        # 分割验证集
        self.train_set, self.val_set = random_split(
            self.train_set, [45000, 5000])
        
        # 数据加载器
        self.train_loader = DataLoader(
            self.train_set, batch_size=128, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(
            self.val_set, batch_size=128, shuffle=False, num_workers=2)
        self.test_loader = DataLoader(
            self.test_set, batch_size=128, shuffle=False, num_workers=2)

# 2. 模型定义
class TriggerGenerator(nn.Module):
    """动态触发器生成器"""
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
            nn.Sigmoid()  # 输出在[0,1]范围内
        )
    
    def forward(self, x):
        features = self.encoder(x)
        trigger = self.decoder(features)
        return trigger * 0.2  # 限制触发器强度

class BackdooredModel(nn.Module):
    """带后门的目标模型"""
    def __init__(self, num_classes=10):
        super().__init__()
        # 使用ResNet18作为基础模型
        self.base_model = torchvision.models.resnet18(num_classes=num_classes)
        self.trigger_generator = TriggerGenerator()
    
    def forward(self, x, trigger_mode=False, target_label=None):
        if trigger_mode:
            # 生成动态触发器
            trigger = self.trigger_generator(x)
            # 应用触发器（论文中的融合方式）
            x_triggered = torch.clamp(x + trigger, 0, 1)
            return self.base_model(x_triggered)
        return self.base_model(x)

# 3. 动态后门攻击实现
class DynamicBackdoorAttack:
    def __init__(self, model, target_label=0, poison_rate=0.1):
        self.model = model.to(device)
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器设置
        self.optimizer = optim.Adam([
            {'params': model.base_model.parameters()},
            {'params': model.trigger_generator.parameters(), 'lr': 1e-4}
        ], lr=1e-3)
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
    
    def poison_data(self, images, labels):
        """毒化数据：随机选择部分样本添加动态触发器并修改标签"""
        batch_size = images.shape[0]
        poison_num = int(batch_size * self.poison_rate)
        
        if poison_num > 0:
            # 随机选择要毒化的样本
            poison_idx = np.random.choice(batch_size, poison_num, replace=False)
            clean_idx = np.setdiff1d(np.arange(batch_size), poison_idx)
            
            # 生成动态触发器
            trigger = self.model.trigger_generator(images[poison_idx])
            
            # 应用触发器并修改标签
            images[poison_idx] = torch.clamp(images[poison_idx] + trigger, 0, 1)
            labels[poison_idx] = self.target_label
            
            return images, labels, poison_idx
        return images, labels, []
    
    def train(self, train_loader, val_loader, epochs=50):
        best_acc = 0.0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                # 毒化部分数据
                poisoned_images, poisoned_labels, _ = self.poison_data(images, labels)
                
                # 前向传播
                outputs = self.model(poisoned_images)
                loss = self.criterion(outputs, poisoned_labels)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 统计信息
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(poisoned_labels).sum().item()
                pbar.set_postfix({'loss': total_loss/(total/images.size(0)), 'acc': 100.*correct/total})
            
            # 验证
            val_acc = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}, Val Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), "best_model.pth")
            
            # 调整学习率
            self.scheduler.step()
    
    def evaluate(self, data_loader, test_backdoor=False):
        self.model.eval()
        correct = 0
        total = 0
        backdoor_success = 0
        backdoor_total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                
                # 正常测试
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 后门测试
                if test_backdoor:
                    # 对所有测试样本添加触发器
                    trigger = self.model.trigger_generator(images)
                    poisoned_images = torch.clamp(images + trigger, 0, 1)
                    poisoned_outputs = self.model(poisoned_images)
                    _, poisoned_predicted = poisoned_outputs.max(1)
                    
                    backdoor_total += labels.size(0)
                    backdoor_success += poisoned_predicted.eq(self.target_label).sum().item()
        
        if test_backdoor:
            print(f"Clean Accuracy: {100.*correct/total:.2f}%, Backdoor Success Rate: {100.*backdoor_success/backdoor_total:.2f}%")
            return 100.*backdoor_success/backdoor_total
        return 100.*correct/total

# 4. 简单防御方法
class BackdoorDefense:
    """基于激活分析的简单防御"""
    @staticmethod
    def detect(model, test_loader, target_label=0):
        model.eval()
        activation_diff = []
        
        # 计算每个样本在有/无触发器时的激活差异
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                
                # 正常前向传播
                clean_output = model(images)
                clean_act = clean_output.softmax(dim=1)
                
                # 带触发器的前向传播
                trigger = model.trigger_generator(images)
                poisoned_images = torch.clamp(images + trigger, 0, 1)
                poisoned_output = model(poisoned_images)
                poisoned_act = poisoned_output.softmax(dim=1)
                
                # 计算目标类的激活差异
                diff = (poisoned_act[:, target_label] - clean_act[:, target_label]).cpu().numpy()
                activation_diff.extend(diff)
        
        # 检测异常（假设差异大于阈值的样本被毒化）
        threshold = np.mean(activation_diff) + 2 * np.std(activation_diff)
        anomalies = np.array(activation_diff) > threshold
        print(f"Detected {anomalies.sum()} potential backdoor samples (threshold: {threshold:.4f})")
        return anomalies

# 5. 主函数
def main():
    # 初始化
    dataset = CIFAR10Dataset()
    model = BackdooredModel()
    
    # 训练干净模型作为基线
    print("Training clean model...")
    clean_model = BackdooredModel().to(device)
    clean_optimizer = optim.Adam(clean_model.parameters(), lr=1e-3)
    clean_criterion = nn.CrossEntropyLoss()
    
    # 简单训练循环（实际应用中应该更完整）
    for epoch in range(5):  # 本例中只训练5个epoch
        clean_model.train()
        for images, labels in dataset.train_loader:
            images, labels = images.to(device), labels.to(device)
            clean_optimizer.zero_grad()
            outputs = clean_model(images)
            loss = clean_criterion(outputs, labels)
            loss.backward()
            clean_optimizer.step()
    
    # 评估干净模型
    clean_model.eval()
    clean_acc = 0
    with torch.no_grad():
        for images, labels in dataset.test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = clean_model(images)
            _, predicted = outputs.max(1)
            clean_acc += predicted.eq(labels).sum().item()
    print(f"Clean model test accuracy: {100.*clean_acc/len(dataset.test_set):.2f}%")
    
    # 执行动态后门攻击
    print("\nLaunching dynamic backdoor attack...")
    attacker = DynamicBackdoorAttack(model, target_label=0, poison_rate=0.1)
    attacker.train(dataset.train_loader, dataset.val_loader, epochs=50)
    
    # 评估攻击效果
    print("\nEvaluating attack performance:")
    # 在干净测试集上的准确率
    clean_acc = attacker.evaluate(dataset.test_loader)
    print(f"Model accuracy on clean test set: {clean_acc:.2f}%")
    
    # 后门攻击成功率
    backdoor_success = attacker.evaluate(dataset.test_loader, test_backdoor=True)
    
    # 执行防御检测
    print("\nRunning defense detection...")
    anomalies = BackdoorDefense.detect(model, dataset.test_loader, target_label=0)
    
    # 可视化示例
    print("\nVisualizing examples...")
    with torch.no_grad():
        # 获取一个测试批次
        test_images, test_labels = next(iter(dataset.test_loader))
        test_images = test_images[:5].to(device)
        
        # 生成触发器
        trigger = model.trigger_generator(test_images)
        poisoned_images = torch.clamp(test_images + trigger, 0, 1)
        
        # 转换为可显示的格式
        def denormalize(img):
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1).to(device)
            std = torch.tensor([0.247, 0.243, 0.261]).view(3,1,1).to(device)
            return img * std + mean
        
        clean_disp = denormalize(test_images).cpu().permute(0,2,3,1).numpy()
        trigger_disp = trigger.cpu().permute(0,2,3,1).numpy()
        poisoned_disp = denormalize(poisoned_images).cpu().permute(0,2,3,1).numpy()
        
        # 显示结果
        plt.figure(figsize=(15,5))
        for i in range(5):
            plt.subplot(3,5,i+1)
            plt.imshow(clean_disp[i])
            plt.title("Clean")
            plt.axis('off')
            
            plt.subplot(3,5,i+6)
            plt.imshow(trigger_disp[i])
            plt.title("Trigger")
            plt.axis('off')
            
            plt.subplot(3,5,i+11)
            plt.imshow(poisoned_disp[i])
            plt.title("Poisoned")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()