import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models

from common import data_transform, train_dataset

# 创建验证数据集对象
validation_dataset = datasets.ImageFolder(root='validation/', transform=data_transform)

# 创建 DataLoader
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

if __name__ == '__main__':
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(train_dataset.classes))
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the validation images: {100 * correct / total}%')
