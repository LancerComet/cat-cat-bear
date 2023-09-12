import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from common import train_dataset, device, model_path

num_epochs = 10
batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Resnet18 足以, 实际效果也不错, 太多层反而增加噪声.
model = models.resnet18(pretrained=True)

# 替换最后的全连接层以匹配分类任务的类别数.
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))
model.to(device)

# 定义损失函数和优化器.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), model_path)

    print("Training complete.")
