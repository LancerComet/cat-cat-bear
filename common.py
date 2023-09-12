# from os import getenv

import numpy as np
# import requests
import torch
import webcolors
from sklearn.cluster import KMeans
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = 'model.pth'

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root='dataset/', transform=data_transform)


# def send_midjourney_message (prompt):
#     url = "http://127.0.0.1:8062/v1/api/trigger/imagine"
#     headers = {
#         "accept": "application/json",
#         "Content-Type": "application/json"
#     }
#     data = {
#         "prompt": prompt
#     }
#     requests.post(url, headers=headers, json=data)


# def send_wx_bot_message (msg_body):
#     bot_key = getenv('WX_BOT_KEY')
#     requests.post(
#        f'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={bot_key}',
#        headers={ 'Content-Type': 'application/json' },
#        json=msg_body
#     )


def extract_theme_color(image, n_colors=1):
    image = image.resize((50, 50))
    pixels = np.array(image).reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_

    return tuple(map(int, np.round(colors[0])))


def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def get_color_name(requested_color):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_color)
    except ValueError:
        closest_name = closest_color(requested_color)
        actual_name = None
    return actual_name, closest_name
