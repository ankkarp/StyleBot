import logging
from aiogram import Bot, Dispatcher, executor, types
import glob
import os
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
# import torchvision.transforms as tt
# from PIL import Image
# from ninasr import *
# from edrs import *
# from rcan import *
# import torch as th

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
async def upscale(message):
    try:
        downloadable = message.photo[-1]
        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load('RRDB_ESRGAN_x4.pth'), strict=True)
        model.eval()
        await downloadable.download(destination_file=f'{message.chat.id}.png')
        img = cv2.imread(f'{message.chat.id}.png', cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite(f'{message.chat.id}.png', output)

        # img = Image.open(img_name)
        # lr_t = tt.ToTensor()(img)[:3]
        # model = edsr_baseline(2, pretrained=True)
        # # model = rcan(2, pretrained=True)
        # # model = edsr(2, pretrained=True)
        # sr_t = model(lr_t)
        # sr = tt.ToPILImage()(sr_t.squeeze(0))
        # sr.save(img_name)
        await bot.send_photo(message.chat.id, types.InputFile(f'{message.chat.id}.png'))
        os.remove(f'{message.chat.id}.png')
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
