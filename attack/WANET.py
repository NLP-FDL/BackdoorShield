'''
模型训练:使用ResNet-18作为基础模型，在毒化的CIFAR-10数据集上训练
防御方法:实现基于激活聚类的防御方法，检测可能的毒化样本，净化数据集并重新训练模型
'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集目录设置
data_dir = "data/cifar-10-batches-py"
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)

# 下载并加载CIFAR-10数据集
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=data_dir, train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 定义ResNet-18模型
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)

# WaNet攻击实现
class WaNet:
    def __init__(self, image_size=32, grid_rescale=1):
        """
        WaNet攻击初始化
        :param image_size: 图像大小
        :param grid_rescale: 网格重缩放因子
        """
        self.image_size = image_size
        self.grid_rescale = grid_rescale
        
        # 定义网格
        self.grid = self._create_grid()
        
        # 定义流场参数
        self.flow = self._create_flow_field()
        
        # 定义目标类别的掩码
        self.target_mask = None
        self.target_label = None
        
    def _create_grid(self):
        """创建归一化网格"""
        grid = np.meshgrid(np.linspace(0, 1, self.image_size),
                           np.linspace(0, 1, self.image_size),
                           indexing='ij')
        grid = np.stack(grid, axis=-1)
        return grid.astype(np.float32)
    
    def _create_flow_field(self):
        """创建流场"""
        # 随机生成流场参数
        k = 4  # 论文中使用的网格大小
        flow = np.random.uniform(-0.1, 0.1, size=(k, k, 2))
        
        # 上采样到图像大小
        flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
        flow = F.interpolate(flow.float(), size=(self.image_size, self.image_size), 
                           mode='bilinear', align_corners=True)
        flow = flow.squeeze(0).permute(1, 2, 0).numpy()
        
        # 应用高斯平滑
        from scipy.ndimage import gaussian_filter
        flow[..., 0] = gaussian_filter(flow[..., 0], sigma=2)
        flow[..., 1] = gaussian_filter(flow[..., 1], sigma=2)
        
        return flow * self.grid_rescale
    
    def _warp_image(self, image):
        """应用扭曲变换到图像"""
        # 将图像转换为numpy数组
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        
        # 计算扭曲后的网格
        warped_grid = self.grid + self.flow
        warped_grid = np.clip(warped_grid, 0, 1)
        
        # 将网格转换为像素坐标
        h, w = self.image_size, self.image_size
        warped_grid[..., 0] *= h - 1
        warped_grid[..., 1] *= w - 1
        
        # 使用双线性插值进行扭曲
        from scipy.interpolate import RegularGridInterpolator
        points = (np.arange(h), np.arange(w))
        interpolator = RegularGridInterpolator(points, image, 
                                            bounds_error=False, 
                                            fill_value=0)
        warped_image = interpolator(warped_grid)
        
        # 转换回torch张量并确保类型为float32
        warped_image = torch.from_numpy(warped_image).permute(2, 0, 1).float()
        return warped_image
    
    def set_target(self, target_label, target_mask=None):
        """设置目标标签和掩码"""
        self.target_label = target_label
        if target_mask is None:
            # 如果没有提供掩码，则随机选择10%的图像作为目标
            self.target_mask = np.random.rand(len(trainset)) < 0.1
        else:
            self.target_mask = target_mask
    
    def poison_dataset(self, dataset):
        """毒化数据集"""
        poisoned_data = []
        poisoned_targets = []
        
        for i in range(len(dataset)):
            image, label = dataset[i]
            
            if self.target_mask is not None and self.target_mask[i]:
                # 对目标样本应用扭曲并改变标签
                poisoned_image = self._warp_image(image)
                poisoned_data.append(poisoned_image)
                poisoned_targets.append(self.target_label)
            else:
                # 保持原样
                poisoned_data.append(image)
                poisoned_targets.append(label)
        
        return list(zip(poisoned_data, poisoned_targets))
    
    def visualize_attack(self, num_samples=5):
        """可视化攻击效果"""
        plt.figure(figsize=(15, 5))
        
        for i in range(num_samples):
            idx = np.where(self.target_mask)[0][i]
            original_img, original_label = trainset[idx]
            poisoned_img, poisoned_label = self.poison_dataset([trainset[idx]])[0]
            
            original_img = original_img.permute(1, 2, 0).numpy()
            poisoned_img = poisoned_img.permute(1, 2, 0).numpy()
            
            # 反归一化
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            original_img = std * original_img + mean
            poisoned_img = std * poisoned_img + mean
            original_img = np.clip(original_img, 0, 1)
            poisoned_img = np.clip(poisoned_img, 0, 1)
            
            plt.subplot(2, num_samples, i+1)
            plt.imshow(original_img)
            plt.title(f"Original: {classes[original_label]}")
            plt.axis('off')
            
            plt.subplot(2, num_samples, num_samples+i+1)
            plt.imshow(poisoned_img)
            plt.title(f"Poisoned: {classes[poisoned_label]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# 训练函数
def train(model, trainloader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(trainloader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs = inputs.float().to(device)  # 确保输入是float32
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar.set_postfix({
            'Loss': f"{train_loss/(batch_idx+1):.3f}",
            'Acc': f"{100.*correct/total:.2f}%"
        })
    
    return train_loss/(batch_idx+1), 100.*correct/total

# 测试函数
def test(model, testloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.float().to(device)  # 确保输入是float32
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss/(batch_idx+1), 100.*correct/total

# 后门攻击测试函数
def test_backdoor(model, testloader, wanet, target_label):
    model.eval()
    correct = 0
    total = 0
    
    # 选择非目标类别的样本
    nontarget_indices = [i for i in range(len(testset)) if testset.targets[i] != target_label]
    nontarget_subset = torch.utils.data.Subset(testset, nontarget_indices)
    nontarget_loader = DataLoader(nontarget_subset, batch_size=100, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(nontarget_loader):
            # 应用WaNet攻击
            poisoned_inputs = []
            for img in inputs:
                poisoned_img = wanet._warp_image(img).to(device)
                poisoned_inputs.append(poisoned_img)
            poisoned_inputs = torch.stack(poisoned_inputs).float()  # 确保是float32
            
            outputs = model(poisoned_inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(torch.tensor(target_label).to(device)).sum().item()
    
    return 100.*correct/total

# 简单防御方法：激活聚类分析
def activation_clustering_defense(model, trainloader, n_clusters=2):
    """激活聚类防御方法"""
    model.eval()
    activations = []
    labels = []
    
    # 获取倒数第二层的激活
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())
    
    hook = model.resnet.avgpool.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for inputs, targets in trainloader:
            inputs = inputs.float().to(device)  # 确保输入是float32
            _ = model(inputs)
            labels.extend(targets.cpu().numpy())
    
    hook.remove()
    
    activations = np.concatenate(activations, axis=0)
    activations = activations.reshape(activations.shape[0], -1)
    labels = np.array(labels)
    
    # 使用K-means聚类
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(activations)
    
    # 计算每个簇的标签分布
    from collections import defaultdict
    cluster_labels = defaultdict(list)
    for cluster, label in zip(clusters, labels):
        cluster_labels[cluster].append(label)
    
    # 识别可能的毒化簇
    poison_cluster = None
    for cluster in cluster_labels:
        label_counts = np.bincount(cluster_labels[cluster])
        if len(label_counts) > 1 and np.max(label_counts) / len(cluster_labels[cluster]) < 0.9:
            poison_cluster = cluster
            break
    
    if poison_cluster is not None:
        print(f"检测到可能的毒化簇 {poison_cluster}")
        print("簇标签分布:", np.bincount(cluster_labels[poison_cluster]))
        return clusters == poison_cluster
    else:
        print("未检测到明显的毒化簇")
        return np.zeros_like(clusters, dtype=bool)

# 主函数
def main():
    # 初始化模型
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    
    # 初始化WaNet攻击
    wanet = WaNet(image_size=32, grid_rescale=1)
    target_label = 0  # 目标类别为"plane"
    wanet.set_target(target_label=target_label)
    
    # 可视化攻击效果
    wanet.visualize_attack(num_samples=5)
    
    # 毒化训练集
    poisoned_trainset = wanet.poison_dataset(trainset)
    poisoned_trainloader = DataLoader(poisoned_trainset, batch_size=128, shuffle=True, num_workers=2)
    
    # 训练模型
    best_acc = 0
    for epoch in range(100):
        train_loss, train_acc = train(model, poisoned_trainloader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, testloader, criterion)
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")
    
    # 测试后门攻击成功率
    attack_success_rate = test_backdoor(model, testloader, wanet, target_label)
    print(f"后门攻击成功率: {attack_success_rate:.2f}%")
    
    # 应用防御方法
    print("\n应用激活聚类防御...")
    is_poisoned = activation_clustering_defense(model, poisoned_trainloader)
    detected_poisoned = np.where(is_poisoned)[0]
    actual_poisoned = np.where(wanet.target_mask)[0]
    
    # 计算防御效果
    true_positives = len(np.intersect1d(detected_poisoned, actual_poisoned))
    false_positives = len(detected_poisoned) - true_positives
    false_negatives = len(actual_poisoned) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"防御效果 - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
    
    # 净化数据集并重新训练
    print("\n净化数据集并重新训练...")
    purified_indices = [i for i in range(len(poisoned_trainset)) if not is_poisoned[i]]
    purified_trainset = torch.utils.data.Subset(trainset, purified_indices)
    purified_trainloader = DataLoader(purified_trainset, batch_size=128, shuffle=True, num_workers=2)
    
    # 重新初始化模型
    purified_model = ResNet18().to(device)
    purified_optimizer = optim.SGD(purified_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    purified_scheduler = optim.lr_scheduler.MultiStepLR(purified_optimizer, milestones=[50, 75], gamma=0.1)
    
    # 训练净化后的模型
    for epoch in range(100):
        train_loss, train_acc = train(purified_model, purified_trainloader, purified_optimizer, criterion, epoch)
        test_loss, test_acc = test(purified_model, testloader, criterion)
        purified_scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # 测试净化后模型的后门攻击成功率
    purified_attack_success_rate = test_backdoor(purified_model, testloader, wanet, target_label)
    print(f"净化后模型的后门攻击成功率: {purified_attack_success_rate:.2f}%")

if __name__ == "__main__":
    main()

