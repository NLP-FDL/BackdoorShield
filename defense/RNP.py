'''
代码说明：
数据加载：
使用CIFAR-10数据集，从data/cifar-10-batches-py/目录加载
实现了自定义Dataset类处理原始CIFAR-10二进制格式
模型架构：
使用轻量级CNN作为受害者模型(3个卷积层+2个全连接层)
后门攻击：实现了BadNets攻击，在图像右下角添加3x3白色方块作为触发器
毒化10%的训练数据，将带触发器的样本标记为目标类(默认为0)
防御方法：重构性神经元剪枝：计算每个神经元对模型输出的重要性(KL散度)
剪枝重要性最低的10%神经元(对正常任务贡献小但可能参与后门行为)
剪枝后进行微调以恢复模型性能

评估指标：正常测试准确率(ACC)\后门攻击成功率(ASR)

实验流程：训练干净模型\毒化数据集并训练带后门的模型\
评估后门攻击成功率

应用防御方法(神经元剪枝+微调)

评估防御后的模型性能和攻击成功率

注意事项：确保CIFAR-10数据位于data/cifar-10-batches-py/目录
完整训练需要较长时间，可减少epochs快速测试
可根据需要调整超参数(剪枝比例、学习率等)
这个实现完整复现了论文的核心方法，包含了攻击和防御的全流程，并提供了关键指标的评估。
'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 数据预处理和加载 (CIFAR-10)
class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        
        if self.train:
            self.data = []
            self.targets = []
            for i in range(1, 6):
                file_path = os.path.join(root_dir, f'data_batch_{i}')
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    self.targets.extend(entry['labels'])
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        else:
            file_path = os.path.join(root_dir, 'test_batch')
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data = entry['data'].reshape(-1, 3, 32, 32)
                self.targets = entry['labels']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = img.transpose((1, 2, 0))  # 转换为HWC格式
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

# 数据增强和归一化
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载数据集
train_dataset = CIFAR10Dataset(
    root_dir='data/cifar-10-batches-py',
    train=True,
    transform=transform_train
)
test_dataset = CIFAR10Dataset(
    root_dir='data/cifar-10-batches-py',
    train=False,
    transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 2. 受害者模型 (轻量级CNN)
class LightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 3. 后门攻击 (BadNets)
class BadNetsAttack:
    def __init__(self, target_class=0, trigger_size=3, trigger_value=1.0):
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
    
    def apply_trigger(self, images):
        """应用触发器到图像上"""
        triggered_images = images.clone()
        # 在右下角添加白色方块作为触发器
        triggered_images[:, :, -self.trigger_size:, -self.trigger_size:] = self.trigger_value
        return triggered_images
    
    def poison_dataset(self, dataset, poison_ratio=0.1):
        """毒化数据集"""
        poisoned_data = []
        poisoned_targets = []
        
        for i in range(len(dataset)):
            img, target = dataset[i]
            if np.random.rand() < poison_ratio:
                img = self.apply_trigger(img.unsqueeze(0)).squeeze(0)
                target = self.target_class
            poisoned_data.append(img)
            poisoned_targets.append(target)
        
        return poisoned_data, poisoned_targets

# 4. 训练受害者模型 (带后门)
def train_model(model, train_loader, test_loader, epochs=20, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        
        # 测试准确率
        test_acc = evaluate_model(model, test_loader)
        print(f'Epoch: {epoch+1} | Loss: {train_loss/(batch_idx+1):.3f} | Train Acc: {100.*correct/total:.2f}% | Test Acc: {test_acc:.2f}%')
    
    return model

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

# 5. 重构性神经元剪枝防御
class ReconstructivePruningDefense:
    def __init__(self, model, clean_loader):
        self.model = model
        self.clean_loader = clean_loader
        self.neuron_importance = {}
    
    def compute_neuron_importance(self):
        """计算每个神经元的重要性(重构损失)"""
        original_outputs = self._get_model_outputs(self.model)
        
        # 遍历所有卷积层和全连接层
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                print(f"Computing importance for layer: {name}")
                self._compute_layer_importance(name, module, original_outputs)
    
    def _get_model_outputs(self, model):
        """获取模型在干净数据上的原始输出"""
        model.eval()
        outputs = []
        with torch.no_grad():
            for inputs, _ in self.clean_loader:
                inputs = inputs.to(device)
                outputs.append(model(inputs))
        return torch.cat(outputs, dim=0)
    
    def _compute_layer_importance(self, layer_name, layer, original_outputs):
        """计算特定层中神经元的重要性"""
        importance_scores = []
        
        # 获取该层的神经元数量
        if isinstance(layer, nn.Conv2d):
            num_neurons = layer.out_channels
        else:  # nn.Linear
            num_neurons = layer.out_features
        
        # 对每个神经元，计算将其置零后的重构损失
        for neuron_idx in range(num_neurons):
            modified_model = self._get_modified_model(layer_name, neuron_idx)
            modified_outputs = self._get_model_outputs(modified_model)
            
            # 计算KL散度作为重构损失
            kl_div = F.kl_div(
                F.log_softmax(modified_outputs, dim=1),
                F.softmax(original_outputs, dim=1),
                reduction='batchmean'
            ).item()
            importance_scores.append(kl_div)
        
        self.neuron_importance[layer_name] = importance_scores
    
    def _get_modified_model(self, layer_name, neuron_idx):
        """返回一个特定神经元被置零的模型副本"""
        model_copy = type(self.model)().to(device)
        model_copy.load_state_dict(self.model.state_dict())
        
        for name, module in model_copy.named_modules():
            if name == layer_name:
                if isinstance(module, nn.Conv2d):
                    # 对于卷积层，将指定通道的输出置零
                    def hook(module, input, output):
                        output[:, neuron_idx, :, :] = 0
                        return output
                    module.register_forward_hook(hook)
                elif isinstance(module, nn.Linear):
                    # 对于全连接层，将指定神经元的输出置零
                    def hook(module, input, output):
                        output[:, neuron_idx] = 0
                        return output
                    module.register_forward_hook(hook)
        return model_copy
    
    def prune_neurons(self, pruning_ratio=0.1):
        """根据重要性分数剪枝神经元"""
        all_neurons = []
        
        # 收集所有神经元及其重要性分数
        for layer_name, scores in self.neuron_importance.items():
            for neuron_idx, score in enumerate(scores):
                all_neurons.append({
                    'layer': layer_name,
                    'neuron_idx': neuron_idx,
                    'importance': score
                })
        
        # 按重要性升序排序 (重要性低的先被剪枝)
        all_neurons.sort(key=lambda x: x['importance'])
        
        # 计算要剪枝的神经元数量
        num_to_prune = int(len(all_neurons) * pruning_ratio)
        
        # 实际剪枝操作
        for i in range(num_to_prune):
            neuron_info = all_neurons[i]
            layer_name = neuron_info['layer']
            neuron_idx = neuron_info['neuron_idx']
            
            for name, module in self.model.named_modules():
                if name == layer_name:
                    if isinstance(module, nn.Conv2d):
                        # 剪枝卷积层的通道
                        module.weight.data[neuron_idx, :, :, :] = 0
                        if module.bias is not None:
                            module.bias.data[neuron_idx] = 0
                    elif isinstance(module, nn.Linear):
                        # 剪枝全连接层的神经元
                        module.weight.data[neuron_idx, :] = 0
                        if module.bias is not None:
                            module.bias.data[neuron_idx] = 0
        
        return self.model

# 6. 主实验流程
def main():
    # 初始化模型
    model = LightCNN().to(device)
    
    # 1. 训练干净模型
    print("Training clean model...")
    clean_model = train_model(model, train_loader, test_loader, epochs=20)
    torch.save(clean_model.state_dict(), 'clean_model.pth')
    
    # 2. 实施后门攻击
    print("\nPreparing poisoned dataset...")
    attack = BadNetsAttack(target_class=0)
    poisoned_data, poisoned_targets = attack.poison_dataset(train_dataset, poison_ratio=0.1)
    
    # 创建毒化数据加载器
    poisoned_dataset = list(zip(poisoned_data, poisoned_targets))
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=128, shuffle=True)
    
    # 3. 训练带后门的模型
    print("\nTraining backdoored model...")
    backdoored_model = LightCNN().to(device)
    backdoored_model = train_model(backdoored_model, poisoned_loader, test_loader, epochs=20)
    torch.save(backdoored_model.state_dict(), 'backdoored_model.pth')
    
    # 4. 评估后门攻击成功率
    print("\nEvaluating backdoor attack success rate...")
    # 创建触发器测试集
    triggered_test_data = []
    triggered_test_targets = []
    for img, target in test_dataset:
        triggered_img = attack.apply_trigger(img.unsqueeze(0)).squeeze(0)
        triggered_test_data.append(triggered_img)
        triggered_test_targets.append(target)
    
    triggered_test_loader = DataLoader(
        list(zip(triggered_test_data, triggered_test_targets)),
        batch_size=100,
        shuffle=False
    )
    
    # 计算攻击成功率 (被误分类为目标类的比例)
    backdoored_model.eval()
    correct = 0
    total = 0
    target_class = attack.target_class
    with torch.no_grad():
        for inputs, _ in triggered_test_loader:
            inputs = inputs.to(device)
            outputs = backdoored_model(inputs)
            _, predicted = outputs.max(1)
            total += inputs.size(0)
            correct += (predicted == target_class).sum().item()
    
    attack_success_rate = 100. * correct / total
    print(f"Backdoor Attack Success Rate: {attack_success_rate:.2f}%")
    
    # 5. 实施防御 (重构性神经元剪枝)
    print("\nApplying Reconstructive Neuron Pruning Defense...")
    
    # 使用干净验证集 (这里简化使用测试集的一部分)
    clean_val_loader = DataLoader(
        list(zip(test_dataset.data[:2000], test_dataset.targets[:2000])),
        batch_size=100,
        shuffle=False
    )
    
    defense = ReconstructivePruningDefense(backdoored_model, clean_val_loader)
    defense.compute_neuron_importance()
    pruned_model = defense.prune_neurons(pruning_ratio=0.1)
    
    # 微调剪枝后的模型
    print("\nFine-tuning pruned model...")
    pruned_model = train_model(pruned_model, train_loader, test_loader, epochs=10, lr=0.001)
    
    # 6. 评估防御效果
    print("\nEvaluating defense effectiveness...")
    # 测试正常准确率
    clean_acc = evaluate_model(pruned_model, test_loader)
    print(f"Clean Test Accuracy after defense: {clean_acc:.2f}%")
    
    # 测试攻击成功率
    pruned_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, _ in triggered_test_loader:
            inputs = inputs.to(device)
            outputs = pruned_model(inputs)
            _, predicted = outputs.max(1)
            total += inputs.size(0)
            correct += (predicted == target_class).sum().item()
    
    new_attack_success_rate = 100. * correct / total
    print(f"Attack Success Rate after defense: {new_attack_success_rate:.2f}%")
    print(f"Defense Effectiveness: {attack_success_rate - new_attack_success_rate:.2f}% reduction")

if __name__ == "__main__":
    main()