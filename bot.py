import logging

from aiogram import Bot, Dispatcher, executor, types

from model import Model

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


@dp.message_handler(commands=['set_style'])
async def set_style(message):
    await message.photo[-1].download('style.jpg')


@dp.message_handler(commands=['set_style'])
async def set_style(message):
    await message.photo[-1].download('style.jpg')


@dp.message_handler(commands=['set_content'])
async def set_content(message):
    await message.photo[-1].download('content.jpg')


@dp.message_handler(commands=['run'])
async def run(message):
    model = Model()
    img = model.stylize(100)
    await bot.send_photo(message.chat.id, types.InputFile('res.jpg'))

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

