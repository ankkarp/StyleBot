# StyleBot

## Описание

Бот умеет:
1. Увеличивать резолючию изображений в 4 раза, используя предобученную модель ESRGAN.
2. Переносить стиль одного изображения на другое, используя медленный алгоритм.

### Инструкция для супер-резолюции

1. Прикрепите картинку маленького разрешения без указания комманды.
2. Подождите результата.

Картинки обязательно присылать как документы, без сжатия. Картинки с сторонами больше 480px будут увеличены до размера 1920px по большей стороне, сохраняя пропорции

#### Инструкция для стилизации картинки по подобию другой

1. Прикрепите основную картинку с подписью /base.
2. Прикрепите картинку-стиль с подписью /style.
3. Наберите комманду /run чтобы запустить стилизацию.
4. Подождите результата.

Во время работы медленного алгоритма пользователь видит прогресс-бар прогресса стилизации.

## Запуск

### Через репозиторий

    git clone https://github.com/ankkarp/StyleBot.git
    pip install -r requirements.txt
    python bot.py -t <ваш токен>
    
### Через докер

    docker pull ankkarp/stylebot
    docker run ankkarp/stylebot -t <ваш токен>
   
## Скриншоты

### Cупер-резолюция

![bandicam 2022-07-29 23-32-49-814](https://user-images.githubusercontent.com/82397895/181839042-5f007cc8-3e6f-4734-a4c8-ab73fdc67799.png)
![bandicam 2022-07-29 23-32-32-406](https://user-images.githubusercontent.com/82397895/181839093-bf126f47-d08d-4adc-aaee-4871a09d7993.png)

### Стилизация

![bandicam 2022-07-29 23-36-22-977](https://user-images.githubusercontent.com/82397895/181839478-416f8227-ce4e-4969-9c90-7e96c4abec11.png)
![bandicam 2022-07-29 23-39-33-265](https://user-images.githubusercontent.com/82397895/181839929-4ddbd7fc-cf40-4213-95e7-36249ea537b5.png)

## Обучение медленного алгоритма стилизации

В ходе экспериметов было выявлено, что оптимизатор Adam на этой задаче хорошо работает с высоким learning rate от 0.1 до 0.2. Среди них наиболее стабильно и быстро получается хорошое качество с lr=0.13. После 100 эпох модель начинает нестабильно обучаться, поэтому обучение производится на 100 эпохах. 

![image](https://user-images.githubusercontent.com/82397895/181995534-49ffb9f0-59ce-4dc8-8989-e5de68ae1942.png)

Ноутбук-черновик с разными экспериментами: https://colab.research.google.com/drive/1C570pDpjZy6pyyYNNdTnMrqiRDHNWhHR?usp=sharing
