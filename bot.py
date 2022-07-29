import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher.filters.state import State
import os
import numpy as np
import torch
import RRDBNet_arch as arch
from stylizer import stylize
from torchvision import transforms as tt
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--token', type=str, help='токен бота')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
bot = Bot(token=args.token)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    Попривестсвовать пользователя
    """
    await message.reply("Здравствуйте! Я умею производить увеличение резолюции и перенос стиля картинок.\n-----------\n"
                        "Супер-резолюция:\n"
                        "1. Прикрепите картинку маленького разрешения без указания комманды \n"
                        "2. Подождите результата\n"
                        "‼ Картинки обязательно присылать как документы, то есть без сжатия.\n"
                        "⚠ Не советуется присылать картинки больше 512x512\n-----------\n"
                        "Стилизация:\n"
                        "1. Прикрепите основную картинку с подписью /base\n"
                        "2. Прикрепите картинку-стиль с подписью /style\n"
                        "3. Наберите комманду /run чтобы запустить стилизацию\n"
                        "4. Подождите результата")


async def download_img(path:str, message: types.Message):
    """
    Функция скачивания картинки
    :param path: путь, куда надо сохранить картинку
    :param message: объект сообщения от пользователя
    :return: размер картинки
    """
    if message.photo:
        await message.photo[-1].download(destination_file=path)
        return message.photo[-1]['width'], message.photo[-1]['height']
    elif message.document:
        await message.document.download(destination_file=path)
        return message.document['thumb']['width'], message.document['thumb']['height']
    else:
        await message.reply('Прикрепите картинку')
        return None


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
        await stylize(progress_bar, steps)
        await progress_bar.reply_photo(types.InputFile(f'temp/res_{message.chat.id}.png'), "Стилизация завершена")
        os.remove(f'temp/res_{message.chat.id}.png')
        # os.remove(f'temp/base_{message.chat.id}.png')
        # os.remove(f'temp/style_{message.chat.id}.png')

    else:
        await message.reply("Сначала определите исходные картинки: (/style - стиля, /base - основы)")


@dp.message_handler(content_types=["document", "text"])
async def upscale(message: types.Message):
    """
    Увеличить картинку в указанное кол-во раз (по умолчанию 4)
    """
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load('RRDB_ESRGAN_x4.pth'), strict=True)
    model.eval()
    img_size = await download_img(f'temp/{message.chat.id}.png', message)
    if img_size:
        img = tt.ToTensor()(Image.open(f'temp/{message.chat.id}.png').convert('RGB'))[:3, :, :].unsqueeze(0)
        with torch.no_grad():
            output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[:3, :, :], (1, 2, 0))
        img = Image.fromarray((output * 255).astype(np.uint8))
        img.save(f'temp/{message.chat.id}.png')
        await message.reply_photo(types.InputFile(f'temp/{message.chat.id}.png'),
                                  caption=f'{img_size[0]}x{img_size[1]} ➜ {output.shape[1]}x{output.shape[0]}')
        os.remove(f'temp/{message.chat.id}.png')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)