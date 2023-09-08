from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from os import getenv
from torchvision import models
from PIL import Image
import torch

from common import data_transform, train_dataset, device, extract_theme_color, send_midjourney_message, send_wx_bot_message, get_color_name


model = models.resnet18(weights=None).to(device)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(train_dataset.classes))
model.load_state_dict(torch.load('model.pth'))
model.to(device)
model.eval()


load_dotenv()
app = Flask(__name__)


@app.route('/')
def index ():
    return render_template('index.html')


@app.route('/task', methods=['POST'])
def image_task():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        image = Image.open(file).convert('RGB')

        theme_color = extract_theme_color(image.copy())
        actual_name, closest_name = get_color_name(theme_color)

        input_tensor = data_transform(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)

        with torch.no_grad():
            output = model(input_batch)

        # 获取预测结果
        _, predicted_idx = torch.max(output, 1)
        cat_breed = train_dataset.classes[predicted_idx]
        prompt = f"9 cats, cat meme, cat sticker, same style, different emotion, gird layout, catoon, cute, kawaii, {cat_breed}, {closest_name} theme color"
        print('Generate image: ' + prompt)
        send_midjourney_message(prompt)

        return 'File uploaded successfully', 200


@app.route('/receive', methods=['POST'])
def receive_json():
    data = request.json
    # print('Received JSON data:', data)

    msg_type = data.get('type')

    if msg_type == 'start':
        print('Start command received.')
        # content = data.get('content')
        msg = {
            'msgtype': 'text',
            'text': {
                'content': '喵喵喵，喵喵喵喵喵喵喵！！嗷呜！！！！'
            }
        }
        send_wx_bot_message(msg)

    elif msg_type == 'banned':
        print('Banned command received:')
        print(data)
        first_attachment_url = data.get('attachments')[0].get('url') if data.get('attachments') else None
        if first_attachment_url is not None:
            msg = {
                'msgtype': 'news',
                'news': {
                    'articles': [
                        {
                            'title': '咩！！！！！！！！',
                            'description': '喵喵！喵喵喵喵喵喵喵！！喵喵喵！！！',
                            'url': first_attachment_url,
                            'picurl': first_attachment_url
                        }
                    ]
                }
            }
            send_wx_bot_message(msg)


    elif msg_type == 'generating':
        print('Generating...')

    elif msg_type == 'end':
        print('End command received:')
        print(data)
        first_attachment_url = data.get('attachments')[0].get('url') if data.get('attachments') else None
        if first_attachment_url is not None:
            msg = {
                'msgtype': 'news',
                'news': {
                    'articles': [
                        {
                            'title': '咩！！！！！！！！',
                            'description': '喵喵！喵喵喵喵喵喵喵！！喵喵喵！！！',
                            'url': first_attachment_url,
                            'picurl': first_attachment_url
                        }
                    ]
                }
            }
            send_wx_bot_message(msg)

    else:
        print("Unknown command received:" + msg_type)

    return jsonify({ 'message': 'JSON data received' }), 200


if __name__ == '__main__':
    app.run(port=3000, host='0.0.0.0')
