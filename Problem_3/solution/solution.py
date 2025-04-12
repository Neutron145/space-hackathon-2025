import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


# Генерация изображения
image = np.full((2000, 2000, 4), 100, dtype=np.uint16)
lin_array = np.linspace(1, 6, 2000, endpoint=False)
x, y = np.meshgrid(lin_array, lin_array)

def get_noised_image(image: np.array, mean: int = 0, sigma: int = 25, loc: int = 0, scale: float = 25.0) -> np.array:
    '''
    Функция наложения шумов (Гаусса, Коши и Лапласа) на изображение
    '''

    image_float = image.astype(np.float32)
    # Генерация шума Коши
    cauchy_noise = np.random.standard_cauchy(image.shape).astype(np.float32)
    cauchy_noise = cauchy_noise * cauchy_noise
    # Генерация шума Лапласа
    laplace_noise = np.random.laplace(loc, scale, image.shape).astype(np.float32)
    # Генерация шума Гаусса
    gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)

    # Объединение шумов на целевом изображении
    result_image = image_float +  cauchy_noise + laplace_noise + gaussian_noise
    # Обрезка яркости до необходимого диапазона
    result_image = np.clip(result_image, 0, 512).astype(np.uint16)

    return result_image + 100

def add_break_pixels(image: np.array) -> np.array:
    '''
    Функция генерации битых пикселей на изображении
    '''
    
    # Генерация массива вероятностей
    random_image = np.random.rand(image.shape[0], image.shape[1])

    # Если вероятность < 0.1%, то пишем единичку, иначе 0
    bin_random_image = (random_image < 0.001).astype(np.uint16)

    # Итерируем по всему массиву, и заменяем единичку на рандомное значение от 450 до 512
    for i in range(0, bin_random_image.shape[0]):
        for j in range(0, bin_random_image.shape[1]):
            if bin_random_image[i][j]:
                bin_random_image[i][j] = random.randint(450, 512)

    # Объединяем сгенерированные битые пиксели с целевым изображением
    result_image = image.astype(np.uint16) + bin_random_image
    # Обрезка яркости до необходимого диапазона
    result_image = np.clip(result_image, 0, 512)
    return result_image

def perlin(x, y, seed=0):
   '''
   Функция генерации шума Перлина
   '''
   
   # создание матрицы для шума и генерация случайного значения seed 
   np.random.seed(seed)
   ptable = np.arange(512, dtype=int)

   # разброс пиков по случайным ячейкам матрицы
   np.random.shuffle(ptable)

   # получение свертки матрицы для линейной интерполяции
   ptable = np.stack([ptable, ptable]).flatten()

   # задание координат сетки
   xi, yi = x.astype(int), y.astype(int)

   # вычисление координат расстояний
   xg, yg = x - xi, y - yi

   # применение функции затухания к координатам расстояний
   xf, yf = fade(xg), fade(yg)

   # вычисление градиентов в заданных интервалах
   n00 = gradient(ptable[ptable[xi] + yi], xg, yg)
   n01 = gradient(ptable[ptable[xi] + yi + 1], xg, yg - 1)
   n11 = gradient(ptable[ptable[xi + 1] + yi + 1], xg - 1, yg - 1)
   n10 = gradient(ptable[ptable[xi + 1] + yi], xg - 1, yg)

   # линейная интерполяция градиентов n00, n01, n11, n10
   x1 = lerp(n00, n10, xf)
   x2 = lerp(n01, n11, xf)
   perlin = lerp(x1, x2, yf)
   perlin_normalized = ((perlin - perlin.min()) / (perlin.max() - perlin.min()) * 512).astype(np.uint16)
   _, mask = cv2.threshold(perlin_normalized, 350, 255, cv2.THRESH_BINARY)
   mask = mask.astype(np.uint8)
   result = cv2.bitwise_and(perlin_normalized, perlin_normalized, mask = mask)
   return result

def lerp(a, b, x):
   '''
   Функция линейной интерполяции
   '''
   return a + x * (b - a)

def fade(f):
   '''
   Функция сглаживания
   '''
   return 6 * f ** 5 - 15 * f ** 4 + 10 * f ** 3

def gradient(c, x, y):
   '''
   Функция вычисления векторов градиента
   '''
   vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
   gradient_co = vectors[c % 4]
   return gradient_co[:, :, 0] * x + gradient_co[:, :, 1] * y

def generate_signal(image: np.array, intensivity: int, count: int) -> np.array:
    '''
    Функция генерации полезных сигналов определенной интенсивности на изображении
    '''
    
    if intensivity > 512:
        intensivity = 512

    result = image.copy()
    
    # Форма полезного сигнала
    shape = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]])
    # Задаем яркость
    shape = shape * intensivity

    for i in range(count): 
        x = random.randint(1, result.shape[0] - 1)
        y = random.randint(1, result.shape[1] - 1)
        result[x, y] = intensivity
        result[x-1, y] = intensivity
        result[x+1, y] = intensivity
        result[x, y-1] = intensivity
        result[x, y+1] = intensivity
    
    return result

def generate_signals(image: np.array, std_dev) -> np.array:
    '''
    Функция генерации сигналов с разными интенсивностями
    '''

    result = generate_signal(image, std_dev * 10, 14)
    result = generate_signal(image, std_dev * 5, 14)
    result = generate_signal(image, std_dev * 2, 13)
    return result

# Создание слоев изображения. Это неоходимо для создания последовательности изображений, имитирующей движение
image[:, :, 0] = get_noised_image(image[:, :, 0])
image[:, :, 1] = add_break_pixels(image[:, :, 1])
image[:, :, 2] = perlin(x, y, seed=3)
image[:, :, 3] = generate_signals(image[:, :, 3], np.std(image[:, :, 0]))

# Объединение шума и полезного сигнала
result_image = image[:, :, 0] + image[:, :, 3]

# Налоежние поверх облаков
img = image[:, :, 2]
mask = img != 0
result_image[mask] = img[mask]

# Запись изображения в .txt файл
cv2.imwrite("image.png", cv2.normalize(result_image, None, 0, 65535, cv2.NORM_MINMAX))
