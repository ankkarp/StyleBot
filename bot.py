import logging
from aiogram import Bot, Dispatcher, executor, types
import torchvision.transforms as tt
from PIL import Image
from ninasr import *
import torch as th
import os

API_TOKEN = '5503345880:AAF_QgSgOHwrMlEY5ABAWajpsDhXxxe3DxM'

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    await message.reply("This bot can style image by pattern of another image. Send 2 pictures: 1) base 2) style")


@dp.message_handler(content_types=['photo'])
async def vk_pfp_check(message):
    try:
        downloadable = message.photo[-1]
        img_name = f'{message.message_id}.png'
        if min(downloadable['width'], downloadable['height']) < 400:
            await downloadable.download(img_name)
            img = Image.open(img_name)
            lr_t = tt.ToTensor()(img)[:3]
            model = ninasr_b2(2, pretrained=True)
            sr_t = model(lr_t)
            sr = tt.ToPILImage()(sr_t.squeeze(0))
            sr.save(img_name)
            await bot.send_photo(message.chat.id, types.InputFile(img_name))
            os.remove(img_name)
        else:
            bot.send_message(message.chat.id, 'Photo does not need rescaling')
    except Exception as e:
        await bot.send_message(message.chat.id, e)


# @dp.message_handler(commands=['set_style'])
# async def set_style(message):
#     await message.photo[-1].download('style.jpg')
#
#
# @dp.message_handler(commands=['set_style'])
# async def set_style(message):
#     await message.photo[-1].download('style.jpg')
#
#
# @dp.message_handler(commands=['set_content'])
# async def set_content(message):
#     await message.photo[-1].download('content.jpg')
#
#
# @dp.message_handler(commands=['run'])
# async def run(message):
#     model = Model()
#     img = model.stylize(100)
#     await bot.send_photo(message.chat.id, types.InputFile('res.jpg'))

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
