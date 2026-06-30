"""
px4flow_algo.py

Прямой Python-порт алгоритма SAD block-matching из PX4/PX4-OpticalFlow
(src/px4flow.cpp). Логика 1:1 повторяет C++ оригинал, чтобы потом
переносить её на STM32F7/H7 почти построчно.

Алгоритм:
1. Изображение разбивается на сетку NUM_BLOCKS x NUM_BLOCKS опорных блоков
   8x8 пикселей (TILE_SIZE).
2. Для каждого блока считается "текстурность" (compute_diff) — если
   слишком гладкий участок, блок пропускается (нет фичей для трекинга).
3. Для оставшихся блоков выполняется SAD (Sum of Absolute Differences)
   поиск в окне [-search_size, +search_size] во втором кадре.
4. Находится позиция с минимальным SAD -> целочисленный сдвиг (sumx, sumy).
5. Считается субпиксельное уточнение (compute_subpixel) по 8 соседним
   позициям вокруг найденного минимума.
6. Усредняются все валидные сдвиги блоков -> итоговый flow в пикселях.

В оригинале используются ARM SIMD интринсики (__USAD8, __UHADD8 и т.д.),
работающие с 4 байтами сразу (упакованными в uint32). Здесь это просто
быстрая реализация на NumPy векторах по 4 пикселя, дающая идентичный
результат — на STM32 эти функции тоже легко заменяются на ARM Cortex-M7
DSP-инструкции (__SADD8/__USAD8 доступны как CMSIS intrinsics).
"""

import numpy as np


TILE_SIZE = 8       # размер блока x & y
NUM_BLOCKS = 5       # количество блоков по x и y для проверки


def _usad8(row1: np.ndarray, row2: np.ndarray) -> int:
    """
    Sum of Absolute Differences для 4 байт (имитация ARM __USAD8).
    row1, row2 — массивы из 4х uint8 (или больше, тогда суммируются все).
    """
    return int(np.sum(np.abs(row1.astype(np.int16) - row2.astype(np.int16))))


def compute_diff(image: np.ndarray, off_x: int, off_y: int) -> int:
    """
    Считает "текстурность" блока 4x4 (с отступом 2 от верхнего левого угла
    8x8 блока) — сумму построчных и постолбцовых разностей соседних пикселей.
    Если результат маленький — участок слишком гладкий (нет текстуры),
    блок отбрасывается.

    Соответствует PX4Flow::compute_diff
    """
    off_x += 2
    off_y += 2

    # 4x4 патч
    patch = image[off_y:off_y + 4, off_x:off_x + 4].astype(np.int16)

    acc = 0
    # построчные разности (между соседними строками)
    for r in range(3):
        acc += _usad8(patch[r], patch[r + 1])

    # постолбцовые разности (между соседними столбцами)
    for c in range(3):
        acc += _usad8(patch[:, c], patch[:, c + 1])

    return acc


def compute_sad_8x8(image1: np.ndarray, image2: np.ndarray,
                     off1_x: int, off1_y: int,
                     off2_x: int, off2_y: int) -> int:
    """
    SAD (Sum of Absolute Differences) между двумя блоками 8x8.
    block1 берётся из image1 в позиции (off1_x, off1_y),
    block2 берётся из image2 в позиции (off2_x, off2_y).

    Соответствует PX4Flow::compute_sad_8x8
    """
    block1 = image1[off1_y:off1_y + 8, off1_x:off1_x + 8].astype(np.int32)
    block2 = image2[off2_y:off2_y + 8, off2_x:off2_x + 8].astype(np.int32)
    return int(np.sum(np.abs(block1 - block2)))


def compute_subpixel(image1: np.ndarray, image2: np.ndarray,
                      off1_x: int, off1_y: int,
                      off2_x: int, off2_y: int) -> np.ndarray:
    """
    Считает SAD для 8 субпиксельных позиций вокруг найденного целочисленного
    минимума, используя билинейную интерполяцию (среднее соседних пикселей).

    Направления (как в оригинале):
        5   6   7
          \\ | /
        4 -- X -- 0
          / | \\
        3   2   1

    Возвращает массив из 8 значений SAD для каждого направления.

    Соответствует PX4Flow::compute_subpixel
    """
    block1 = image1[off1_y:off1_y + 8, off1_x:off1_x + 8].astype(np.int32)

    # Берём область image2 чуть шире (9x9), чтобы было откуда брать
    # соседей для интерполяции по краям блока.
    img2 = image2.astype(np.int32)

    acc = np.zeros(8, dtype=np.int64)

    for i in range(8):
        for j in range(8):
            y = off2_y + i
            x = off2_x + j

            # Среднее двух соседних пикселей (имитация __UHADD8)
            def avg(y1, x1, y2, x2):
                return (int(img2[y1, x1]) + int(img2[y2, x2])) // 2

            s0 = avg(y, x, y, x + 1)
            s1 = avg(y + 1, x, y + 1, x + 1)
            s2 = avg(y, x, y + 1, x)
            s3 = avg(y + 1, x, y + 1, x - 1)
            s4 = avg(y, x, y, x - 1)
            s5 = avg(y - 1, x, y - 1, x - 1)
            s6 = avg(y, x, y - 1, x)
            s7 = avg(y - 1, x, y - 1, x + 1)

            t1 = (s0 + s1) // 2
            t3 = (s3 + s4) // 2
            t5 = (s4 + s5) // 2
            t7 = (s7 + s0) // 2

            subpix = [s0, t1, s2, t3, s4, t5, s6, t7]
            ref_val = int(block1[i, j])

            for k in range(8):
                acc[k] += abs(ref_val - subpix[k])

    return acc


def compute_flow(image1: np.ndarray, image2: np.ndarray,
                  search_size: int = 4,
                  flow_feature_threshold: int = 30,
                  flow_value_threshold: int = 3000):
    """
    Главная функция вычисления optical flow между двумя кадрами.

    Соответствует PX4Flow::compute_flow.

    Параметры:
        image1, image2: grayscale uint8 numpy массивы одинакового размера
        search_size: радиус поиска в пикселях (PX4Flow default = 4)
        flow_feature_threshold: минимальная "текстурность" блока
        flow_value_threshold: максимальный приемлемый SAD для блока

    Возвращает:
        (quality, flow_x, flow_y) — flow в пикселях за кадр (НЕ рад/с,
        преобразование в угловую скорость делается снаружи)
    """
    h, w = image1.shape
    image_width = w  # PX4Flow использует только ширину (квадратные кадры)

    winmin = -search_size
    winmax = search_size

    pix_lo = search_size + 1
    pix_hi = image_width - (search_size + 1) - TILE_SIZE
    pix_step = (pix_hi - pix_lo) // NUM_BLOCKS + 1

    dirs_x = []
    dirs_y = []
    subdirs = []

    j = pix_lo
    while j < pix_hi:
        i = pix_lo
        while i < pix_hi:
            diff = compute_diff(image1, i, j)

            if diff >= flow_feature_threshold:
                # Блок с достаточной текстурой -> ищем сдвиг
                best_dist = 0xFFFFFFFF
                best_dx = 0
                best_dy = 0

                for dy in range(winmin, winmax + 1):
                    for dx in range(winmin, winmax + 1):
                        dist = compute_sad_8x8(image1, image2, i, j, i + dx, j + dy)
                        if dist < best_dist:
                            best_dist = dist
                            best_dx = dx
                            best_dy = dy

                if best_dist < flow_value_threshold:
                    acc = compute_subpixel(image1, image2, i, j,
                                            i + best_dx, j + best_dy)
                    mindist = best_dist
                    mindir = 8  # 8 = "нет направления"

                    for k in range(8):
                        if acc[k] < mindist:
                            mindist = acc[k]
                            mindir = k

                    dirs_x.append(best_dx)
                    dirs_y.append(best_dy)
                    subdirs.append(mindir)

            i += pix_step
        j += pix_step

    meancount = len(dirs_x)

    if meancount <= 10:
        return 0, 0.0, 0.0

    # Субпиксельные смещения по направлению (как в оригинале)
    histflow_x = 0.0
    histflow_y = 0.0

    for dx, dy, d in zip(dirs_x, dirs_y, subdirs):
        subdir_x = 0.0
        if d in (0, 1, 7):
            subdir_x = 0.5
        elif d in (3, 4, 5):
            subdir_x = -0.5

        subdir_y = 0.0
        if d in (5, 6, 7):
            subdir_y = -0.5
        elif d in (1, 2, 3):
            subdir_y = 0.5

        histflow_x += dx + subdir_x
        histflow_y += dy + subdir_y

    histflow_x /= meancount
    histflow_y /= meancount

    quality = int(meancount * 255 / (NUM_BLOCKS * NUM_BLOCKS))
    quality = min(quality, 255)

    return quality, histflow_x, histflow_y
