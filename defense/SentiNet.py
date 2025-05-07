#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

基本流程：
1. 使用 CIFAR10 数据集训练一个简单 CNN。
2. 构造后门样本：在图像右下角添加一个红色触发器（后门触发器），假设攻击模型会将带触发器的样本预测为目标类别（类别 0）。
3. 防御方法（Sentinet思路）：对输入图像采用滑动窗口局部遮挡，记录遮挡后模型对原始预测类别的概率变化，计算最大差异作为不一致性分数（Inconsistency Score）。若不一致性分数高于预设阈值，则认为该样本可能被后门触发。
本示例代码仅为简化的演示
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np

# 固定随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------
# 1. 定义简单 CNN 模型（类似 LeNet-5）
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # CIFAR10: 3x32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 输出 32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 输出 64x32x32
        self.pool = nn.MaxPool2d(2, 2)  # 缩小一半
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (B,64,16,16)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
# 2. 定义数据预处理与加载
# ---------------------------
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# ---------------------------
# 3. 模拟后门样本生成函数
# ---------------------------
def add_trigger(image: Image.Image, trigger_size=5, color=(255, 0, 0)):
    """
    在图像右下角添加一个触发器：一个 trigger_size x trigger_size 的彩色方块
    image: PIL.Image，RGB
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # 触发器区域为右下角
    x0, y0 = w - trigger_size, h - trigger_size
    x1, y1 = w, h
    draw.rectangle([x0, y0, x1, y1], fill=color)
    return img

def convert_tensor_to_pil(tensor):
    """
    将归一化后的 tensor 转换为 PIL Image
    假设 tensor 的归一化参数为均值0.5, 标准差0.5
    """
    inv_norm = transforms.Normalize(
        mean=[-0.5/0.5]*3,
        std=[1/0.5]*3
    )
    tensor = inv_norm(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    np_img = tensor.mul(255).byte().permute(1,2,0).cpu().numpy()
    return Image.fromarray(np_img)

def convert_pil_to_tensor(image: Image.Image):
    """
    将 PIL Image 转换为 tensor，并归一化为(-1,1)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    return transform(image)

# ---------------------------
# 4. 定义 Sentinet 防御函数（基于局部遮挡一致性检测）
# ---------------------------
def occlusion_consistency(model: nn.Module, image_tensor: torch.Tensor, device: torch.device, window_size=5, stride=5):
    """
    对输入 image_tensor (B, C, H, W) 对每个位置进行局部遮挡，
    记录遮挡后原始预测类别的概率变化，
    返回最大概率变化值作为不一致性分数。
    这里仅对单张图像（B=1）进行处理。
    """
    model.eval()
    with torch.no_grad():
        # 计算原始预测（取 softmax 概率中最高类别的概率）
        outputs = model(image_tensor.to(device))
        probs = F.softmax(outputs, dim=1)
        orig_conf, orig_class = torch.max(probs, dim=1)
        orig_conf = orig_conf.item()
        orig_class = orig_class.item()

    # 将图像 tensor 转换为 numpy 以便操作
    _, C, H, W = image_tensor.size()
    occlusion_diffs = []
    # 遍历所有位置（滑动窗口），遮挡窗口区域置为0（黑色）
    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            occluded = image_tensor.clone().to(device)
            occluded[:,:, y:y+window_size, x:x+window_size] = 0.0
            with torch.no_grad():
                out_occ = model(occluded)
                prob_occ = F.softmax(out_occ, dim=1)[0, orig_class].item()
            occlusion_diffs.append(abs(orig_conf - prob_occ))
    # 返回最大不一致性差值
    if occlusion_diffs:
        return max(occlusion_diffs)
    else:
        return 0.0

# ---------------------------
# 5. 主函数
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型并训练（为了示例，这里训练较少的 epoch）
    model = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("=== Training CNN on CIFAR10 ===")
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(trainloader.dataset)
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")
    
    # 测试干净测试集准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    clean_acc = correct / total
    print(f"\nClean Test Accuracy: {clean_acc*100:.2f}%")
    
    # 选择测试集中的一张样本，构造干净样本和后门样本
    # 先取一张测试集图片，并转换为 PIL 图像（反归一化）
    sample_img_tensor, sample_label = testset[0]
    sample_img_pil = convert_tensor_to_pil(sample_img_tensor)
    # 构造后门样本：在图片右下角添加红色触发器
    triggered_img_pil = add_trigger(sample_img_pil, trigger_size=5, color=(255, 0, 0))
    
    # 分别计算两种样本的 occlusion consistency 分数
    # 转换回 tensor（归一化后的形式）
    sample_img_tensor_norm = convert_pil_to_tensor(sample_img_pil).unsqueeze(0)
    triggered_img_tensor_norm = convert_pil_to_tensor(triggered_img_pil).unsqueeze(0)
    
    # 计算不一致性分数
    window_size = 5
    stride = 5
    crc_clean = occlusion_consistency(model, sample_img_tensor_norm, device, window_size, stride)
    crc_triggered = occlusion_consistency(model, triggered_img_tensor_norm, device, window_size, stride)
    
    print("\n=== Occlusion Consistency Detection ===")
    print(f"CRC Score (clean sample): {crc_clean:.4f}")
    print(f"CRC Score (triggered sample): {crc_triggered:.4f}")
    
    # 假设设定阈值，例如 0.5，若不一致性分数高于该值，则认为存在后门
    threshold = 0.5
    if crc_triggered > threshold:
        print("Triggered sample detected as potential backdoor!")
    else:
        print("Triggered sample not detected as backdoored.")
    
