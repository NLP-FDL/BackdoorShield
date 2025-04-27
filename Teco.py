import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import copy
import random
from sklearn.metrics import accuracy_score

# 设置随机种子保证可重复性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载CIFAR-10数据集
def load_cifar10(data_dir='./data/cifar-10-batches-py'):
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    return trainloader, testloader

# 定义一个轻量级CNN模型作为受害者模型
class LightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 训练受害者模型
def train_model(model, trainloader, testloader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        
        # 在测试集上评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}, Test Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"Training finished. Best Test Accuracy: {best_acc:.2f}%")
    model.load_state_dict(torch.load('best_model.pth'))
    return model

# 后门攻击：向数据中添加触发器
class BadNetAttack:
    def __init__(self, target_class=0, trigger_size=3, trigger_value=1.0):
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
    
    def add_trigger(self, images):
        # 在右下角添加白色方块作为触发器
        triggered_images = images.clone()
        triggered_images[:, :, -self.trigger_size:, -self.trigger_size:] = self.trigger_value
        return triggered_images
    
    def poison_dataset(self, dataset, poison_ratio=0.1):
        poisoned_data = []
        poisoned_labels = []
        
        for i in range(len(dataset)):
            img, label = dataset[i]
            
            # 按比例毒化数据
            if random.random() < poison_ratio:
                img = self.add_trigger(img.unsqueeze(0)).squeeze(0)
                label = self.target_class
            
            poisoned_data.append(img)
            poisoned_labels.append(label)
        
        return list(zip(poisoned_data, poisoned_labels))

# 腐败变换：用于检测后门的各种图像变换
class CorruptionTransforms:
    @staticmethod
    def gaussian_noise(image, severity=1):
        c = [0.04, 0.06, 0.08, 0.09, 0.10][severity-1]
        noise = torch.randn_like(image) * c
        return torch.clamp(image + noise, 0, 1)
    
    @staticmethod
    def shot_noise(image, severity=1):
        c = [500, 250, 100, 75, 50][severity-1]
        return torch.poisson(image * c) / c
    
    @staticmethod
    def impulse_noise(image, severity=1):
        c = [0.01, 0.02, 0.03, 0.05, 0.07][severity-1]
        mask = torch.rand_like(image) < c
        noise = torch.rand_like(image)
        image = image.clone()
        image[mask] = noise[mask]
        return image
    
    @staticmethod
    def contrast(image, severity=1):
        c = [0.75, 0.5, 0.4, 0.3, 0.15][severity-1]
        means = torch.mean(image, dim=(1,2), keepdim=True)
        return torch.clamp((image - means) * c + means, 0, 1)
    
    @staticmethod
    def brightness(image, severity=1):
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
        return torch.clamp(image + c, 0, 1)
    
    @staticmethod
    def defocus_blur(image, severity=1):
        # 简化的模糊实现
        kernel_size = [3, 5, 7, 9, 11][severity-1]
        blur = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
        return blur(image)
    
    @staticmethod
    def get_all_transforms():
        return [
            CorruptionTransforms.gaussian_noise,
            CorruptionTransforms.shot_noise,
            CorruptionTransforms.impulse_noise,
            CorruptionTransforms.contrast,
            CorruptionTransforms.brightness,
            CorruptionTransforms.defocus_blur,
        ]

# 后门检测器：基于腐败鲁棒性一致性
class BackdoorDetector:
    def __init__(self, model, corruption_transforms, num_classes=10):
        self.model = model
        self.corruption_transforms = corruption_transforms
        self.num_classes = num_classes
    
    def compute_consistency(self, images, labels, severity=3):
        """
        计算原始预测和腐败后预测之间的一致性
        """
        self.model.eval()
        with torch.no_grad():
            # 原始预测
            clean_outputs = self.model(images)
            clean_preds = torch.argmax(clean_outputs, dim=1)
            
            # 初始化一致性分数
            consistency_scores = torch.zeros(len(images)).to(device)
            
            # 对每种腐败变换计算一致性
            for transform in self.corruption_transforms:
                corrupted_images = transform(images, severity)
                corrupted_outputs = self.model(corrupted_images)
                corrupted_preds = torch.argmax(corrupted_outputs, dim=1)
                
                # 一致性是预测是否保持不变
                consistency = (clean_preds == corrupted_preds).float()
                consistency_scores += consistency
            
            # 平均一致性分数
            consistency_scores /= len(self.corruption_transforms)
            
            # 真实标签的准确性
            clean_accuracy = (clean_preds == labels).float()
            
            return consistency_scores, clean_accuracy
    
    def detect_backdoors(self, testloader, threshold=0.5):
        """
        检测后门样本
        """
        all_scores = []
        all_labels = []
        all_clean_acc = []
        
        for images, labels in tqdm(testloader, desc="Detecting backdoors"):
            images, labels = images.to(device), labels.to(device)
            scores, clean_acc = self.compute_consistency(images, labels)
            
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_clean_acc.extend(clean_acc.cpu().numpy())
        
        # 将一致性分数转换为numpy数组
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        all_clean_acc = np.array(all_clean_acc)
        
        # 标记低一致性样本为后门
        backdoor_flags = all_scores < threshold
        
        return backdoor_flags, all_scores, all_labels, all_clean_acc

# 评估防御效果
def evaluate_defense(backdoor_flags, true_labels, clean_accuracy, target_class=0):
    """
    评估防御效果:
    - 检测率: 正确识别的后门样本比例
    - 误报率: 将干净样本误认为后门的比例
    """
    # 真实后门样本是那些被错误分类到目标类的样本
    true_backdoors = (true_labels != target_class) & (clean_accuracy == 0)
    true_clean = ~true_backdoors
    
    # 计算检测率和误报率
    detection_rate = np.sum(backdoor_flags & true_backdoors) / np.sum(true_backdoors) if np.sum(true_backdoors) > 0 else 0
    false_alarm_rate = np.sum(backdoor_flags & true_clean) / np.sum(true_clean) if np.sum(true_clean) > 0 else 0
    
    print(f"Detection Rate: {detection_rate*100:.2f}%")
    print(f"False Alarm Rate: {false_alarm_rate*100:.2f}%")
    
    return detection_rate, false_alarm_rate

# 主函数
def main():
    # 1. 加载和准备数据
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader = load_cifar10()
    
    # 2. 训练干净的受害者模型
    print("\nTraining clean victim model...")
    clean_model = LightCNN().to(device)
    clean_model = train_model(clean_model, trainloader, testloader, epochs=50)
    
    # 3. 实施后门攻击
    print("\nPerforming backdoor attack...")
    # 获取原始测试集
    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar-10-batches-py', train=False, download=True, transform=transform_test)
    
    # 创建后门攻击实例 (将目标类别设为0)
    backdoor_attack = BadNetAttack(target_class=0)
    
    # 毒化测试集 (20%的样本被注入后门)
    poisoned_testset = backdoor_attack.poison_dataset(testset, poison_ratio=0.2)
    poisoned_testloader = DataLoader(
        poisoned_testset, batch_size=100, shuffle=False, num_workers=2)
    
    # 4. 训练后门模型 (在实际攻击中，攻击者会毒化训练集)
    print("\nTraining backdoored model...")
    # 为了演示，我们直接在干净模型上测试后门攻击的效果
    backdoored_model = copy.deepcopy(clean_model)
    
    # 评估后门攻击成功率
    backdoored_model.eval()
    trigger_count = 0
    trigger_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in poisoned_testloader:
            images, labels = images.to(device), labels.to(device)
            
            # 添加触发器
            triggered_images = backdoor_attack.add_trigger(images)
            
            # 原始预测
            outputs = backdoored_model(images)
            _, preds = torch.max(outputs, 1)
            
            # 触发后预测
            triggered_outputs = backdoored_model(triggered_images)
            _, triggered_preds = torch.max(triggered_outputs, 1)
            
            # 统计攻击成功率
            for i in range(len(labels)):
                if labels[i] != backdoor_attack.target_class:  # 原始标签不是目标类
                    trigger_count += 1
                    if triggered_preds[i] == backdoor_attack.target_class:  # 触发后被分类为目标类
                        trigger_correct += 1
            
            total += len(labels)
    
    attack_success_rate = trigger_correct / trigger_count if trigger_count > 0 else 0
    print(f"Backdoor Attack Success Rate: {attack_success_rate*100:.2f}%")
    
    # 5. 使用腐败鲁棒性一致性检测后门
    print("\nDetecting backdoors using corruption robustness consistency...")
    corruption_transforms = CorruptionTransforms.get_all_transforms()
    detector = BackdoorDetector(backdoored_model, corruption_transforms)
    
    # 检测后门样本 (使用毒化的测试集)
    backdoor_flags, consistency_scores, true_labels, clean_accuracy = detector.detect_backdoors(poisoned_testloader, threshold=0.5)
    
    # 6. 评估防御效果
    print("\nEvaluating defense performance...")
    evaluate_defense(backdoor_flags, true_labels, clean_accuracy, target_class=backdoor_attack.target_class)
    
    # 7. 可视化一些检测结果
    print("\nVisualizing some detection results...")
    # 选择前10个样本进行可视化
    sample_indices = range(10)
    for idx in sample_indices:
        original_label = true_labels[idx]
        is_backdoor = (original_label != backdoor_attack.target_class) and (clean_accuracy[idx] == 0)
        detected = backdoor_flags[idx]
        
        print(f"Sample {idx}:")
        print(f"  Original Label: {original_label}")
        print(f"  Is Backdoor: {is_backdoor}")
        print(f"  Detected as Backdoor: {detected}")
        print(f"  Consistency Score: {consistency_scores[idx]:.4f}")
        print(f"  Clean Accuracy: {clean_accuracy[idx]}")
        print()

if __name__ == "__main__":
    main()