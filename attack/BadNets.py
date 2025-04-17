import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import copy
import random

# 设置随机种子保证可重复性
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实验参数配置
config = {
    'trigger_size': 4,          # 触发器大小（像素）
    'trigger_pos': 'bottom_right',  # 触发器位置
    'poison_ratio': 0.1,        # 训练集投毒比例
    'target_label': 5,          # 目标标签
    'batch_size': 64,
    'num_epochs': 10,
    'pruning_rate': 0.2,        # 神经元剪枝比例
    'defense_mode': True        # 是否启用防御
}

# 修复1：调整数据预处理流程
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 1. 带后门的数据集实现（修复版）
class PoisonedMNIST(Dataset):
    """实现带触发器的MNIST数据集"""
    def __init__(self, root, train=True, transform=None, 
                 poison_ratio=0.0, target_label=5):
        # 先加载原始数据集而不应用转换
        self.raw_dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=True, transform=None)
        self.transform = transform
        self.poison_ratio = poison_ratio
        self.target_label = target_label
        
        # 生成污染索引
        num_samples = len(self.raw_dataset)
        self.poison_indices = np.random.choice(
            num_samples, int(num_samples * poison_ratio), replace=False)
        self.poison_flag = np.zeros(num_samples, dtype=bool)
        self.poison_flag[self.poison_indices] = True

    def _add_trigger(self, img):
        """在图像右下角添加白色方块触发器"""
        img = img.copy()
        pixels = img.load()
        width, height = img.size
        
        # 触发器位置计算
        start_x = width - config['trigger_size']
        start_y = height - config['trigger_size']
        
        # 添加白色方块
        for x in range(start_x, width):
            for y in range(start_y, height):
                pixels[x, y] = 255
        return img

    def __getitem__(self, index):
        # 获取原始PIL图像和标签
        img, label = self.raw_dataset[index]
        
        # 如果是污染样本
        if self.poison_flag[index]:
            img = self._add_trigger(img)  # 直接操作PIL图像
            label = self.target_label
        
        # 应用数据增强转换
        if self.transform:
            img = self.transform(img)
            
        return img, label, self.poison_flag[index]

    def __len__(self):
        return len(self.raw_dataset)

# 2. 模型架构（保持不变）
class BadNet(nn.Module):
    """论文中的CNN架构实现"""
    def __init__(self):
        super(BadNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32*7*7, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# 3. 训练过程（保持不变）
def train(model, train_loader, criterion, optimizer):
    """模型训练函数"""
    model.train()
    for epoch in range(config['num_epochs']):
        total_loss = 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {total_loss/len(train_loader):.4f}")

# 4. 测试函数（保持不变）
def test(model, loader, is_poisoned=False):
    """模型测试函数"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    test_type = "后门测试" if is_poisoned else "干净测试"
    print(f"{test_type}准确率: {acc:.2f}%")
    return acc

# 5. 防御方法（保持不变）后门攻击的神经元特性
"""后门攻击通常会在神经网络中形成特定的"捷径路径"：
高度依赖少数神经元：触发器激活的路径往往集中于少量特定神经元
异常激活模式：后门相关神经元的权重绝对值通常较大（为了强响应触发器）
与正常路径分离：后门路径和正常分类路径相对独立
论文《Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks》通过实验验证了这些特性。这段代码通过结构化剪枝破坏了后门攻击所需的特定神经路径，利用后门路径对少数神经元的强依赖性实现防御。其本质是牺牲少量模型容量换取安全性提升，符合深度学习模型鲁棒性研究的通用范式。
"""
def defense_neuron_pruning(model):
    """防御方法：剪枝激活值较低的神经元"""
    model_copy = copy.deepcopy(model).to(device)
    
    # 仅处理全连接层
    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Linear):
            weights = module.weight.data.abs().cpu().numpy()
            # 计算神经元重要性（L1范数）
            importance = np.sum(weights, axis=1)
            
            # 确定剪枝阈值
            threshold = np.percentile(importance, config['pruning_rate']*100)
            mask = torch.from_numpy(importance > threshold).to(device)
            
            # 应用剪枝
            module.weight.data = module.weight.data[mask, :]
            if module.bias is not None:
                module.bias.data = module.bias.data[mask]
            
            # 调整下一层的输入维度
            for next_module in model_copy.modules():
                if isinstance(next_module, nn.Linear) and next_module.in_features == module.out_features:
                    next_module.weight.data = next_module.weight.data[:, mask]
    
    return model_copy

# 6. 可视化函数（修复版）
def visualize_samples(dataset, num_samples=5, poisoned=False):
    """可视化样本"""
    indices = np.where(dataset.poison_flag)[0] if poisoned else np.arange(len(dataset))
    indices = np.random.choice(indices, num_samples, replace=False)
    
    plt.figure(figsize=(10, 3))
    for i, idx in enumerate(indices):
        img_tensor, label, is_poison = dataset[idx]
        
        # 修复2：将张量转换为可显示的图像格式
        img = img_tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
        img = img * 0.3081 + 0.1307  # 反归一化
        img = np.clip(img, 0, 1)
        
        ax = plt.subplot(1, num_samples, i+1)
        ax.imshow(img.squeeze(), cmap='gray')  # 移除颜色通道维度
        ax.set_title(f"Label: {label}\n{'Poisoned' if is_poison else 'Clean'}")
        ax.axis('off')
    plt.show()

# 主函数（微调）
def main():
    # 准备数据集（应用统一的transform）
    train_set = PoisonedMNIST(
        root='./data', train=True, transform=base_transform,
        poison_ratio=config['poison_ratio'], target_label=config['target_label'])
    
    test_clean = PoisonedMNIST(
        root='./data', train=False, transform=base_transform, poison_ratio=0.0)
    
    test_poisoned = PoisonedMNIST(
        root='./data', train=False, transform=base_transform, 
        poison_ratio=1.0, target_label=config['target_label'])
    
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    test_clean_loader = DataLoader(test_clean, batch_size=config['batch_size'], shuffle=False)
    test_poisoned_loader = DataLoader(test_poisoned, batch_size=config['batch_size'], shuffle=False)
    
    # 可视化样本
    print("干净样本示例:")
    visualize_samples(train_set, poisoned=False)
    print("污染样本示例:")
    visualize_samples(train_set, poisoned=True)
    
    # 初始化模型
    model = BadNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("\n开始训练...")
    train(model, train_loader, criterion, optimizer)
    
    # 测试模型
    print("\n测试原始模型:")
    clean_acc = test(model, test_clean_loader)
    poison_acc = test(model, test_poisoned_loader, is_poisoned=True)
    
    # 防御处理
    if config['defense_mode']:
        print("\n应用神经元剪枝防御...")
        defended_model = defense_neuron_pruning(model)
        print("\n测试防御后模型:")
        defended_clean_acc = test(defended_model, test_clean_loader)
        defended_poison_acc = test(defended_model, test_poisoned_loader, is_poisoned=True)
    
    # 实验结果汇总
    print("\n=== 实验结果 ===")
    print(f"原始模型 - 干净准确率: {clean_acc:.2f}%")
    print(f"原始模型 - 后门成功率: {poison_acc:.2f}%")
    if config['defense_mode']:
        print(f"防御模型 - 干净准确率: {defended_clean_acc:.2f}%")
        print(f"防御模型 - 后门成功率: {defended_poison_acc:.2f}%")

if __name__ == "__main__":
    main()