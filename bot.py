import logging
from aiogram import Bot, Dispatcher, executor, types
# import torchvision.transforms as tt
# from PIL import Image
# from ninasr import *
# from edrs import *
# from rcan import *
# import torch as th
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
    os.system('cd ESRGAN')
    await message.reply("This bot can style image by pattern of another image. Send 2 pictures: 1) base 2) style")


@dp.message_handler(content_types=['photo'])
async def vk_pfp_check(message):
    try:
        downloadable = message.photo[-1]
        os.chdir(os.path.abspath('ESRGAN'))
        await downloadable.download(f'LR/{message.chat.id}.png')
        # img = Image.open(img_name)
        # lr_t = tt.ToTensor()(img)[:3]
        # model = edsr_baseline(2, pretrained=True)
        # # model = rcan(2, pretrained=True)
        # # model = edsr(2, pretrained=True)
        # sr_t = model(lr_t)
        # sr = tt.ToPILImage()(sr_t.squeeze(0))
        # sr.save(img_name)
        os.system("python test.py")
        await bot.send_photo(message.chat.id, types.InputFile(f'results/{message.chat.id}_rlt.png'))
        os.remove(f'results/{message.chat.id}_rlt.png')
        os.remove(f'LR/{message.chat.id}.png')
        # bot.send_message(message.chat.id, 'Photo does not need rescaling')
    except Exception as e:
        print(e)


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
