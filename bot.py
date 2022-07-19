import logging
from aiogram import Bot, Dispatcher, executor, types
import os
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from stylizer import stylize

API_TOKEN = '5503345880:AAF_QgSgOHwrMlEY5ABAWajpsDhXxxe3DxM'

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

__all__ = ['set_style', 'set_base', 'get_style', 'get_base', 'run', 'upscale', ]


@dp.message_handler(commands='start')
async def send_welcome(message: types.Message):
    """
    Попривестсвовать пользователя
    """
    await message.reply("Здравствуйте! Я умею изменять размер картинок, в том числе увеличивать их размер, "
                        "и производить перенос стиля одной картинки на другую. "
                        "Чтобы получить список комманд наберите /help")


@dp.message_handler(commands='help')
async def help(message: types.Message):
    """
    Вывести листинг комманд бота
    """
    await message.reply("Комманды:\n/style - определить картинку-стиль\n/base - определить картинку-основу\n"
                        "/run - запустить стилизацию\n"
                        "/get_base - прислать выбранную картинку-основу"
                        "/get_style - прислать выбранную картинку-стиль"
                        "Чтобы выполнить супер-резолюцию картинки пришлите ее без указания комманды\n"
                        "Рекоммендуется присылать картинки в виде документов (то есть без сжатия)")


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
        await bot.send_message(message.chat.id, 'Прикрепите картинку')
        return None


@dp.message_handler(commands='style', commands_ignore_caption=False, content_types=["document", "photo"])
async def set_style(message: types.Message):
    """
    Определить картинку-стиль
    """
    await download_img(f'style_{message.chat.id}.png', message)


@dp.message_handler(commands='base', commands_ignore_caption=False, content_types=["document", "photo"])
async def set_base(message: types.Message):
    """
    Определить картинку-основу
    """
    await download_img(f'base_{message.chat.id}.png', message)


async def get_img(path: str, message: types.Message):
    """
    Проверить наличие исходной картинки и прислать ее если она опредеоена
    :param path: путь, по которому должна находиться картинка
    :param message: объект сообщения от пользователя
    :return:
    """
    if os.path.exists(path):
        await bot.send_photo(message.chat.id, types.InputFile(path))
    else:
        await bot.send_message(message.chat.id, "Картинка еще не была определена")


@dp.message_handler(commands='get_style')
async def get_style(message: types.Message):
    """
    Прислать выбранную картинку-стиль
    """
    await get_img(f'style_{message.chat.id}.png', message)


@dp.message_handler(commands='get_base')
async def get_base(message: types.Message):
    """
    Прислать выбранную картинку-основу
    """
    await get_img(f'base_{message.chat.id}.png', message)


@dp.message_handler(commands='run')
async def run(message: types.Message):
    """
    Запустить перенос стиля
    """
    if os.path.exists(f'base_{message.chat.id}.png') and os.path.exists(f'style_{message.chat.id}.png'):
        stylize(message.chat.id, 100)
        await bot.send_photo(message.chat.id, types.InputFile(f'res_{message.chat.id}.png'))
    else:
        await bot.send_message(message.chat.id, "Сначала определите исходные картинки: (/style - стиля, /base - основы)")


@dp.message_handler(content_types=["document", "photo", "text"])
async def upscale(message: types.Message):
    """
    Увеличить картинку в указанное кол-во раз (по умолчанию 4)
    """
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load('RRDB_ESRGAN_x4.pth'), strict=True)
    model.eval()
    img_size = await download_img(f'{message.chat.id}.png', message)
    if img_size:
        print(img_size)
        print(message.caption)
        if message.caption:
            text = message.caption
            if '-k' in text:
                koef = text.split('-k')[1][: text.rfind(' ')].strip()
                print(koef)
                if not koef.isdigit():
                    await bot.send_message(message.chat.id, "Коэффицент должен быть числом")
                    return
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