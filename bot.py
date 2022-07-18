import logging
from aiogram import Bot, Dispatcher, executor, types
import os
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from stylizer import stylize
import asyncio
# import torchvision.transforms as tt
# from PIL import Image
# from ninasr import *
# from edrs import *
# from rcan import *
# import torch as th

API_TOKEN = '5503345880:AAF_QgSgOHwrMlEY5ABAWajpsDhXxxe3DxM'

# Configure logging
logging.basicConfig(level=logging.INFO)

loop = asyncio.get_event_loop()
# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands='start')
async def send_welcome(message: types.Message):
    """
    Попривестсвовать пользователя
    """
    await message.reply("Здравствуйте! Я умею изменять размер картинок, в том числе увеличивать их размер, "
                        "и производить перенос стиля одной картинки на другую. "
                        "Чтобы получить список комманд наберите /help")


@dp.message_handler(commands='help')
async def send_welcome(message: types.Message):
    """
    Вывести листинг комманд бота
    """
    await message.reply("Комманды:\nДля переноса стиля:\n"
                        "/style - определить картинку со стилем\n"
                        "/base - определить картинку, куда надо переносить стиль\n"
                        "/run - запустить стилизацию\nДля супер-резолюции:"
                        "/upscale - увеличить размер картинки в указанное в 4 раза\n"
                        "Рекоммендуется присылать картинки в виде документов (то есть без сжатия)")


@dp.message_handler(commands='style', commands_ignore_caption=False, content_types=["document", "photo"])
async def set_style(message):
    """
    Определить картинку-стиль
    """
    if message.photo:
        await message.photo[-1].download(destination_file=f'style_{message.chat.id}.png')
    elif message.document:
        await message.document.download(destination_file=f'style_{message.chat.id}.png')
    else:
        await bot.send_message(message.chat.id, 'Прикрепите картинку')


@dp.message_handler(commands='base', commands_ignore_caption=False, content_types=["document", "photo"])
async def set_base(message):
    """
    Определить картинку-основу
    """
    if message.photo:
        await message.photo[-1].download(destination_file=f'base_{message.chat.id}.png')
    elif message.document:
        await message.document.download(destination_file=f'base_{message.chat.id}.png')
    else:
        await bot.send_message(message.chat.id, 'Прикрепите картинку')


@dp.message_handler(commands='run')
async def run(message):
    """
    Запустить перенос стиля
    """
    if os.path.exists(f'base_{message.chat.id}.png') or os.path.exists(f'style_{message.chat.id}.png'):
        stylize(message.chat.id, 100)
        await bot.send_photo(message.chat.id, types.InputFile(f'res_{message.chat.id}.png'))
    else:
        await bot.send_message(message.chat.id, "Сначала определите исходные картинки (/style - стиля, /base - основы)")


@dp.message_handler(content_types=["document", "photo"])
async def upscale(message):
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load('RRDB_ESRGAN_x4.pth'), strict=True)
    model.eval()
    await message.document.download(destination_file=f'{message.chat.id}.png')
    img = cv2.imread(f'{message.chat.id}.png', cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(f'{message.chat.id}.png', output)
    await bot.send_photo(message.chat.id, types.InputFile(f'{message.chat.id}.png'))
    os.remove(f'{message.chat.id}.png')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)