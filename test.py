import torch
from torchvision import models
from PIL import Image
import os
from torchvision import models

from common import data_transform, train_dataset

test_dir = 'test'

if __name__ == '__main__':
    # 创建模型对象
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(train_dataset.classes))
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # 从 'test' 目录中读取所有图片
    for filename in os.listdir(test_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # 只处理 jpg, png, jpeg 格式的图片
            image_path = os.path.join(test_dir, filename)
            image = Image.open(image_path).convert('RGB')  # 转换为 RGB 格式
            input_tensor = data_transform(image)  # 应用数据预处理
            input_batch = input_tensor.unsqueeze(0)  # 创建一个 mini-batch，因为模型需要一个 batch

            # 进行预测
            with torch.no_grad():
                output = model(input_batch)

            # 获取预测结果
            _, predicted_idx = torch.max(output, 1)
            predicted_label = train_dataset.classes[predicted_idx]  # 假设 train_dataset 是你之前用于训练的数据集对象

            print(f"Image {filename} is predicted to be {predicted_label}")
