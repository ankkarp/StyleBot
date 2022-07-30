from io import BytesIO

import torch
import torch.optim as optim
from torchvision import transforms as tt
import numpy as np
from PIL import Image


def load_image(img_path:str, max_size=400, shape=None):
    """
        Загрузка изображания, изменение его размера и нормализация

        :param path: путь к изображению
    """
    img = np.array(Image.open(img_path).convert('RGB'))
    size = shape if shape else min(max_size, max(img.shape))
    in_transform = tt.Compose([
        tt.ToTensor(),
        tt.Resize(size),
        tt.Normalize((0.485, 0.456, 0.406),
                     (0.229, 0.224, 0.225))])
    return in_transform(img)[:3, :, :].unsqueeze(0)


def get_features(image, vgg, layers=None):
    """
        Получить свойства изображения
    """
    layers = {'0': 'conv1_1',
              '5': 'conv2_1',
              '10': 'conv3_1',
              '19': 'conv4_1',
              '21': 'conv4_2',
              '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in vgg._modules.items():
        x = layer(x)
        if str(name) in layers:
            features[layers[name]] = x
    return features


def gram_matrix(m):
    """
        Подсчёт матрицы Грамма
    """
    b, d, h, w = m.size()
    m = m.view(d, h * w)
    gram = torch.mm(m, m.t())
    return gram


def im_convert(m):
    """
        Перевод тензора в формат, пригодный для записи в PIL изображение
    """
    try:
        m = m.to("cpu").clone().detach().numpy().squeeze()
    except Exception as e:
        pass
    image = m.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    return image.clip(0, 1)


async def stylize(message, steps=100):
    """
        Стилизация

        :param message: объект сообщения от пользователя
        :param steps: кол-ао эпох (default 100)
        :return: BytesIO стилизованного изображения
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg = torch.load('model.h5').to(device)
    style_weights = {'conv1_1': 1., 'conv2_1': 0.8, 'conv3_1': 0.5, 'conv4_1': 0.3, 'conv5_1': 0.1}
    content_weight = 1
    style_weight = 1e6
    content = load_image(f'temp/base_{message.chat.id}.png').to(device)
    style = load_image(f'temp/style_{message.chat.id}.png', shape=content.shape[-2:]).to(device)
    target = content.clone().requires_grad_(True)
    optimizer = optim.Adam([target], lr=0.13)
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    for i in range(1, steps+1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"]) ** 2)
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (d * h * w)
        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        await message.bot.edit_message_text(chat_id=message.chat.id, message_id=message.message_id,
                                            text=f"[{i}/{steps}] \tВыполняется стилизация{'.'*(i%3+1)}")
    buffer = BytesIO()
    Image.fromarray((im_convert(target) * 255).astype(np.uint8)).save(buffer, "PNG")
    buffer.seek(0)
    return buffer
