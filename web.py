import torch
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, request, render_template
from torchvision import models

from common import data_transform, train_dataset, device, extract_theme_color, \
    get_color_name

model = models.resnet18(weights=None).to(device)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(train_dataset.classes))
model.load_state_dict(torch.load('model.pth'))
model.to(device)
model.eval()


load_dotenv()
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def image_task():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        image = Image.open(file).convert('RGB')

        # Get theme color of the image.
        theme_color = extract_theme_color(image.copy())
        actual_name, closest_name = get_color_name(theme_color)

        input_tensor = data_transform(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)

        with torch.no_grad():
            output = model(input_batch)

        # Predict the breed of the cat in the image.
        _, predicted_idx = torch.max(output, 1)
        cat_breed = train_dataset.classes[predicted_idx]
        prompt = f"{cat_breed}, {closest_name} color"
        print('Generate image: ' + prompt)
        return prompt, 200


if __name__ == '__main__':
    app.run(port=3000, host='0.0.0.0')
