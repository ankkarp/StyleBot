import logging
from io import BytesIO

from aiogram import Bot, Dispatcher, executor, types
import aiogram
import os
import numpy as np
import torch
from aiogram.utils import exceptions

import RRDBNet_arch as arch
from stylizer import stylize
from torchvision import transforms as tt
from PIL import Image, UnidentifiedImageError
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--token', type=str, help='токен бота')
parser.add_argument('--debug', action='store_true', help='включить подродное логирование в файл')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

bot = Bot(token=args.token)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    Попривестсвовать пользователя
    """
    await message.reply("Здравствуйте! Я умею стилизовать и увеличивать в 4 раза картинки.\n-----------\n"
                        "Супер-резолюция:\n"
                        "1. Прикрепите картинку размером без указания комманды \n"
                        "2. Дождитесь результата\n"
                        "‼ Картинки обязательно присылать как документы, без сжатия.\n"
                        "⚠ Картинки с сторонами больше 480px будут увеличены до размера 1920px по большей стороне, "
                        "сохраняя пропорции.\n-----------\nСтилизация:\n"
                        "1. Прикрепите основную картинку с подписью /base\n"
                        "2. Прикрепите картинку-стиль с подписью /style\n"
                        "3. Наберите комманду /run чтобы запустить стилизацию\n"
                        "4. Дождитесь результата")


async def download_img(path:str, message: types.Message):
    """
    Функция скачивания картинки
    :param path: путь, куда надо сохранить картинку
    :param message: объект сообщения от пользователя
    :return: None
    """
    if message.document:
        await message.document.download(destination_file=path)
    elif message.photo:
        await message.photo[-1].download(destination_file=path)
    try:
        img = Image.open(path)
        img.verify()
        ratio = img.size[0] / img.size[1]
        if ratio > 20 or ratio < 0.05:
            await message.reply(f'ОШИБКА: Была дана картинка размера {img.size[0]}x{img.size[1]}.\n'
                                'Стороны отличаются в более чем 20 раз.\nВыберите другую картинку.')
            os.remove(path)
    except UnidentifiedImageError:
        await message.reply('ОШИБКА: Неправильное расширение файла.\nУбедитесь, что было прикреплено изображение.')
        os.remove(path)
        return
    except (IOError, SyntaxError):
        await message.reply('ОШИБКА: Файл поврежден.\nУбедитесь в формате и целосности изображения.')
        os.remove(path)


@dp.message_handler(commands='style', commands_ignore_caption=False, content_types=["document", "photo"])
async def set_style(message: types.Message):
    """
    Определить картинку-стиль
    """
    await download_img(f'temp/style_{message.chat.id}.png', message)


@dp.message_handler(commands='base', commands_ignore_caption=False, content_types=["document", "photo"])
async def set_base(message: types.Message):
    """
    Определить картинку-основу
    """
    await download_img(f'temp/base_{message.chat.id}.png', message)


@dp.message_handler(commands='run')
async def run(message: types.Message):
    """
    Запустить стилизацию
    """
    if os.path.exists(f'temp/base_{message.chat.id}.png') and os.path.exists(f'temp/style_{message.chat.id}.png'):
        steps = 100
        media = types.MediaGroup()
        media.attach_photo(types.InputFile(f'temp/style_{message.chat.id}.png'), "Картинка-стиль")
        media.attach_photo(types.InputFile(f'temp/base_{message.chat.id}.png'), "Картинка-основа")
        await message.reply_media_group(media)
        progress_bar = await message.answer(f"[0/{steps}] \tВыполняется стилизация.")
        stylized_img = await stylize(progress_bar, steps)
        await progress_bar.reply_photo(types.InputFile(stylized_img), "Стилизация завершена")
        os.remove(f'temp/base_{message.chat.id}.png')
        os.remove(f'temp/style_{message.chat.id}.png')

    else:
        await message.reply("ОШИБКА: Исходные картинки ещё не были определены.\nУстановите их коммандами:\n"
                            "/style - картинка-стиль\n/base - картинка-основа")


@dp.message_handler(content_types=["document"])
async def upscale(message: types.Message):
    """
        Увеличить картинку в 4 раза
    """
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load('RRDB_ESRGAN_x4.pth'), strict=True)
    model.eval()
    buffer = BytesIO()
    await message.document.download(destination_file=buffer)
    logging.info("Upscale started")
    try:
        img = tt.ToTensor()(Image.open(buffer).convert('RGB'))[:3, :, :].unsqueeze(0)
        buffer.close()
        img_size = img.shape[-2:]
        if max(img_size) / min(img_size) >= 20:
            await message.reply(f'ОШИБКА: Была дана картинка размера {img_size[1]}x{img_size[0]}, '
                                'Стороны отличаются в более чем 20 раз.\nВыберите другую картинку.')
            return
        rescale_by = 480 / max(img_size)
        logging.debug(f"Input shape: {img.shape}")
        if rescale_by < 1:
            img = tt.Resize((int(rescale_by * img_size[0]), int(rescale_by * img_size[1])))(img)
            logging.debug(f"Resized input shape: {img.shape}")
        with torch.no_grad():
            output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[:3, :, :], (1, 2, 0))
        logging.debug(f"Output shape: {output.shape}")
        img = Image.fromarray((output * 255).astype(np.uint8))
        buffer = BytesIO()
        img.save(buffer, "PNG")
        buffer.seek(0)
        await message.reply_photo(types.InputFile(buffer),
                                  caption=f'{img_size[1]}x{img_size[0]} ➜ {output.shape[1]}x{output.shape[0]}')
        buffer.close()
    except UnidentifiedImageError:
        await message.reply('ОШИБКА: Неправильное расширение файла. \nУбедитесь, что было прикреплено изображение.')

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
