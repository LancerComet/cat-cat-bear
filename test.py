import torch
from torchvision import models
from PIL import Image
import os

from common import data_transform, train_dataset, device, model_path

model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(train_dataset.classes))
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

test_data_dir = 'test'

if __name__ == '__main__':
    for filename in os.listdir(test_data_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(test_data_dir, filename)
            image = Image.open(image_path).convert('RGB')
            input_tensor = data_transform(image)

            # 创建一个 mini-batch，因为模型需要一个 batch
            input_batch = input_tensor.unsqueeze(0)
            input_batch = input_batch.to(device)

            with torch.no_grad():
                output = model(input_batch)

            _, predicted_idx = torch.max(output, 1)
            predicted_label = train_dataset.classes[predicted_idx]

            print(f"Image {filename} is predicted to be {predicted_label}")
