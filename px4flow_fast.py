"""
px4flow_fast.py

Векторизованная (быстрая) версия того же алгоритма, что в px4flow_algo.py.
Даёт идентичный результат, но с использованием NumPy broadcasting вместо
вложенных Python-циклов — нужна только для того, чтобы тестировать на
ПК в реальном времени с веб-камерой (30+ fps).

px4flow_algo.py — эталон "один в один как C++", удобен для понимания
и портирования логики на STM32.
px4flow_fast.py — для интерактивного теста, чтобы не ждать секундами
на каждый кадр при поэлементной реализации на чистом Python.

Сам алгоритм (что и где считается) идентичен — отличается только
способ исполнения.
"""

import numpy as np

TILE_SIZE = 8
NUM_BLOCKS = 5


def _block_positions(image_width: int, search_size: int):
    pix_lo = search_size + 1
    pix_hi = image_width - (search_size + 1) - TILE_SIZE
    pix_step = (pix_hi - pix_lo) // NUM_BLOCKS + 1

    positions = []
    j = pix_lo
    while j < pix_hi:
        i = pix_lo
        while i < pix_hi:
            positions.append((i, j))
            i += pix_step
        j += pix_step
    return positions


def compute_diff_batch(image: np.ndarray, positions) -> np.ndarray:
    """Векторизованный compute_diff для всех блоков сразу."""
    diffs = np.empty(len(positions), dtype=np.int64)
    img = image.astype(np.int16)

    for idx, (off_x, off_y) in enumerate(positions):
        ox, oy = off_x + 2, off_y + 2
        patch = img[oy:oy + 4, ox:ox + 4]

        acc = 0
        for r in range(3):
            acc += int(np.sum(np.abs(patch[r] - patch[r + 1])))
        for c in range(3):
            acc += int(np.sum(np.abs(patch[:, c] - patch[:, c + 1])))

        diffs[idx] = acc

    return diffs


def _compute_subpixel_fast(img1: np.ndarray, img2: np.ndarray,
                            off1_x: int, off1_y: int,
                            off2_x: int, off2_y: int) -> np.ndarray:
    """Векторизованная версия compute_subpixel (весь 8x8 блок сразу)."""
    block1 = img1[off1_y:off1_y + 8, off1_x:off1_x + 8]

    # Берём область image2 с запасом по краям на соседей (10x10)
    region = img2[off2_y - 1:off2_y + 9, off2_x - 1:off2_x + 9]

    def shifted(dy, dx):
        return region[1 + dy:1 + dy + 8, 1 + dx:1 + dx + 8]

    center = shifted(0, 0)
    right = shifted(0, 1)
    down = shifted(1, 0)
    down_left = shifted(1, -1)
    left = shifted(0, -1)
    up_left = shifted(-1, -1)
    up = shifted(-1, 0)
    up_right = shifted(-1, 1)
    down_right = shifted(1, 1)

    s0 = (center + right) // 2
    s1 = (down + down_right) // 2
    s2 = (center + down) // 2
    s3 = (down + down_left) // 2
    s4 = (center + left) // 2
    s5 = (up + up_left) // 2
    s6 = (center + up) // 2
    s7 = (up + up_right) // 2

    t1 = (s0 + s1) // 2
    t3 = (s3 + s4) // 2
    t5 = (s4 + s5) // 2
    t7 = (s7 + s0) // 2

    acc = np.empty(8, dtype=np.int64)
    acc[0] = np.sum(np.abs(block1 - s0))
    acc[1] = np.sum(np.abs(block1 - t1))
    acc[2] = np.sum(np.abs(block1 - s2))
    acc[3] = np.sum(np.abs(block1 - t3))
    acc[4] = np.sum(np.abs(block1 - s4))
    acc[5] = np.sum(np.abs(block1 - t5))
    acc[6] = np.sum(np.abs(block1 - s6))
    acc[7] = np.sum(np.abs(block1 - t7))

    return acc


def compute_flow_fast(image1: np.ndarray, image2: np.ndarray,
                       search_size: int = 4,
                       flow_feature_threshold: int = 30,
                       flow_value_threshold: int = 3000,
                       invert_sign: bool = False):
    """
    Быстрая версия compute_flow. Логика идентична px4flow_algo.compute_flow,
    но SAD по окну поиска считается векторно через sliding_window_view,
    а не вложенными Python-циклами.

    ВАЖНО про знак результата (совпадает с оригинальным px4flow.cpp):
    Алгоритм ищет блок из image1 в image2 -- т.е. возвращает смещение
    КАРТИНКИ (текстуры сцены) между кадрами, а не смещение камеры.
    Если камера сдвинулась вправо на +N пикселей, текстура в кадре
    сдвинулась ВЛЕВО на -N пикселей -- поэтому compute_flow вернёт
    flow_x = -N. Это эталонное поведение оригинала PX4Flow, и именно
    так PX4 EKF2 это и ожидает (поле integrated_x в DroneCAN message).

    invert_sign=True развернёт знак на "интуитивный" (flow = направление
    движения камеры) -- удобно для отладки/визуализации, но НЕ используйте
    это при портировании на STM32 для реальной публикации в DroneCAN,
    там нужно оставаться на оригинальном соглашении знаков PX4Flow.
    """
    h, w = image1.shape
    image_width = w

    positions = _block_positions(image_width, search_size)
    diffs = compute_diff_batch(image1, positions)

    img1 = image1.astype(np.int32)
    img2 = image2.astype(np.int32)

    winmin = -search_size
    winmax = search_size

    dirs_x = []
    dirs_y = []
    subdirs = []

    for (off_x, off_y), diff in zip(positions, diffs):
        if diff < flow_feature_threshold:
            continue

        ref_block = img1[off_y:off_y + 8, off_x:off_x + 8]

        sy0 = off_y + winmin
        sx0 = off_x + winmin
        search_h = 8 + 2 * search_size
        search_w = 8 + 2 * search_size
        search_region = img2[sy0:sy0 + search_h, sx0:sx0 + search_w]

        windows = np.lib.stride_tricks.sliding_window_view(
            search_region, (8, 8)
        )  # shape: (win_size, win_size, 8, 8)

        sad = np.sum(np.abs(windows - ref_block[None, None, :, :]), axis=(2, 3))

        best_idx = np.unravel_index(np.argmin(sad), sad.shape)
        best_dist = int(sad[best_idx])
        best_dy = best_idx[0] + winmin
        best_dx = best_idx[1] + winmin

        if best_dist >= flow_value_threshold:
            continue

        acc = _compute_subpixel_fast(img1, img2, off_x, off_y,
                                      off_x + best_dx, off_y + best_dy)

        mindist = best_dist
        mindir = 8
        for k in range(8):
            if acc[k] < mindist:
                mindist = acc[k]
                mindir = k

        dirs_x.append(best_dx)
        dirs_y.append(best_dy)
        subdirs.append(mindir)

    meancount = len(dirs_x)
    if meancount <= 10:
        return 0, 0.0, 0.0

    histflow_x = 0.0
    histflow_y = 0.0

    for dx, dy, d in zip(dirs_x, dirs_y, subdirs):
        subdir_x = 0.5 if d in (0, 1, 7) else (-0.5 if d in (3, 4, 5) else 0.0)
        subdir_y = -0.5 if d in (5, 6, 7) else (0.5 if d in (1, 2, 3) else 0.0)
        histflow_x += dx + subdir_x
        histflow_y += dy + subdir_y

    histflow_x /= meancount
    histflow_y /= meancount

    quality = min(int(meancount * 255 / (NUM_BLOCKS * NUM_BLOCKS)), 255)

    if invert_sign:
        histflow_x = -histflow_x
        histflow_y = -histflow_y

    return quality, histflow_x, histflow_y
