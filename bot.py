import logging
from aiogram import Bot, Dispatcher, executor, types
import torchvision.transforms as tt
from PIL import Image
from ninasr import *
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch as th
from torch.hub import load_state_dict_from_url

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
async def enhance(message):
    await message.photo[-1].download(f'{message.message_id}.png')
    img = Image.open(f'{message.message_id}.png')
    lr_t = tt.ToTensor()(img)[:3]
    # model = th.load('RRDB_PSNR_x4.pth', map_location=th.device('cpu'))
    model = ninasr_b2(2, pretrained=True)
    # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    # model.load_state_dict(th.load('RealESRGAN_x4plus.pth'))
    sr_t = model(lr_t)
    sr = tt.ToPILImage()(sr_t.squeeze(0))
    sr.save(f'{message.message_id}.png')
    await bot.send_photo(message.chat.id, types.InputFile(f'{message.message_id}.png'))


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

