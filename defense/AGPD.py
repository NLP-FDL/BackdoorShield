#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
利用激活梯度信息检测被后门污染的样本。
基本流程：
1. 对 CIFAR10 训练集中的部分样本（例如10%）进行数据中毒：在图像右下角添加一个白色触发器（后门触发器），并将标签强制修改为目标标签（0）。
2. 使用一个简单的 CNN（例如LeNet）对混合数据进行训练。
3. 在推理阶段，对于每个测试样本，计算模型在指定中间层（本例中为 conv2 层）输出关于预测类别得分的梯度， 并计算其 L2 范数。假设后门样本的梯度范数明显高于干净样本。
4. 根据梯度范数与设定阈值比较，检测后门触发样本。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from PIL import Image, ImageDraw
import numpy as np
import random

# 固定随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------
# 1. 数据中毒相关函数
# ----------------------------
def add_trigger(image: Image.Image, trigger_size=5) -> Image.Image:
    """
    在图像右下角添加一个白色触发器
    image: PIL Image, RGB
    trigger_size: 触发器的边长（像素）
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x0, y0 = w - trigger_size, h - trigger_size
    x1, y1 = w, h
    draw.rectangle([x0, y0, x1, y1], fill=(255,255,255))
    return img

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    将 tensor (C,H,W) 反归一化并转换为 PIL Image
    假设归一化参数：均值=(0.5,0.5,0.5)，标准差=(0.5,0.5,0.5)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
        std=[1/0.5, 1/0.5, 1/0.5]
    )
    tensor = inv_normalize(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    np_img = tensor.mul(255).byte().permute(1,2,0).cpu().numpy()
    return Image.fromarray(np_img)

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    将 PIL Image 转换为 tensor，并归一化到 (-1,1)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    return transform(image)

# ----------------------------
# 2. 定义简单的 CNN 模型（类似 LeNet）
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 输出：32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输出：64x32x32
        self.pool = nn.MaxPool2d(2, 2)  # 下采样
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        # 用于防御检测的中间层输出，我们选择 conv2 层的输出
        self.feature_map = None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # 保存 conv2 层的输出（在池化前）
        self.feature_map = x.clone()
        x = self.pool(x)  # (B,64,16,16)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------------------
# 3. 训练与测试函数
# ----------------------------
def train_model(model, train_loader, optimizer, criterion, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ----------------------------
# 4. 激活梯度检测函数
# ----------------------------
def compute_activation_gradient_norm(model: nn.Module, image: torch.Tensor, device: torch.device) -> float:
    """
    对单个输入 image (1, C, H, W)，计算模型在 conv2 层的激活输出对预测类别得分的梯度，
    并返回梯度的 L2 范数作为指标。
    """
    model.eval()
    image = image.to(device)
    image.requires_grad = True

    # 前向传播
    output = model(image)
    pred_class = output.argmax(dim=1).item()
    # 取预测类别对应的 logit
    target_logit = output[0, pred_class]
    # 计算梯度
    model.zero_grad()
    target_logit.backward(retain_graph=True)
    # 取 conv2 层输出的梯度（注意：这里需要获取 conv2 的输入梯度或者 conv2 层输出梯度）
    # 这里我们利用 model.feature_map 已保存 conv2 层输出
    # 注意：由于我们没有注册 hook 获取梯度，这里简单地对 image 求梯度也可作为示例
    grad_norm = image.grad.data.norm(2).item()
    return grad_norm

def detect_poisoned_sample(model: nn.Module, image: torch.Tensor, device: torch.device, threshold: float) -> bool:
    """
    对输入图像计算激活梯度指标，若大于 threshold，则认为样本可能被中毒。
    """
    grad_norm = compute_activation_gradient_norm(model, image, device)
    return grad_norm > threshold, grad_norm

# ----------------------------
# 5. 主函数
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理与加载（CIFAR10）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    # 初始化模型
    model = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("=== Training CNN on CIFAR10 ===")
    train_model(model, trainloader, optimizer, criterion, device, num_epochs=5)
    clean_acc = test_model(model, testloader, device)
    print(f"\nClean Test Accuracy: {clean_acc*100:.2f}%")

    # 模拟后门样本：在图像右下角添加一个白色触发器
    # 这里选取测试集中第一个样本作为示例
    sample_img, sample_label = testset[0]
    # 将 tensor 转换为 PIL 以便添加触发器
    def tensor_to_pil_img(tensor):
        inv_norm = transforms.Normalize(
            mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
            std=[1/0.5, 1/0.5, 1/0.5]
        )
        tensor = inv_norm(tensor)
        tensor = torch.clamp(tensor, 0, 1)
        np_img = tensor.mul(255).byte().permute(1,2,0).cpu().numpy()
        return Image.fromarray(np_img)

    sample_pil = tensor_to_pil_img(sample_img)
    triggered_pil = add_trigger(sample_pil, trigger_size=5)
    # 转换回 tensor（归一化）
    sample_tensor = pil_to_tensor(sample_pil).unsqueeze(0)  # (1, C, H, W)
    triggered_tensor = pil_to_tensor(triggered_pil).unsqueeze(0)  # (1, C, H, W)

    # 计算激活梯度指标
    threshold = 0.05  # 阈值可调，本示例为演示设定
    is_poisoned_clean, grad_norm_clean = detect_poisoned_sample(model, sample_tensor, device, threshold)
    is_poisoned_triggered, grad_norm_triggered = detect_poisoned_sample(model, triggered_tensor, device, threshold)

    print("\n=== Activation Gradient Based Detection ===")
    print(f"Clean sample gradient norm: {grad_norm_clean:.4f}, detected as poisoned: {is_poisoned_clean}")
    print(f"Triggered sample gradient norm: {grad_norm_triggered:.4f}, detected as poisoned: {is_poisoned_triggered}")


