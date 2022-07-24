FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ["bot.py", "model.h5", "RRDB_ESRGAN_x4.pth", "RRDBNet_arch.py", "stylizer.py", "./"]

CMD [ "python3", "bot.py"]