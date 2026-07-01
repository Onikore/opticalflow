"""
px4flow_improved.py — улучшенная версия compute_flow поверх px4flow_fast.

Каждое улучшение из плана (fallback.md) — отдельный флаг, по умолчанию OFF.
При всех флагах OFF результат БИТ-В-БИТ совпадает с px4flow_fast.compute_flow_fast
(проверяется в benchmark/equivalence). Включаем по одному и меряем эффект —
так каждое улучшение изолированно и легко портируется на C по отдельности.

Флаги (в порядке приоритета из fallback.md):
  use_median     — медиана блоков вместо арифметического среднего (выбросы)
  use_uniqueness — отбраковка неуникального минимума SAD (aperture problem)
  use_parabolic  — параболическая субпиксельная интерполяция вместо 8-напр.
  use_fb_check   — forward-backward consistency check (ненадёжные блоки)
"""

import numpy as np

TILE_SIZE = 8
NUM_BLOCKS = 5


def _block_positions(image_width, search_size):
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


def _compute_diff(img, off_x, off_y):
    ox, oy = off_x + 2, off_y + 2
    patch = img[oy:oy + 4, ox:ox + 4].astype(np.int16)
    acc = 0
    for r in range(3):
        acc += int(np.sum(np.abs(patch[r] - patch[r + 1])))
    for c in range(3):
        acc += int(np.sum(np.abs(patch[:, c] - patch[:, c + 1])))
    return acc


def _sad_volume(img1, img2, off_x, off_y, search_size):
    """Полный SAD-куб (2S+1)x(2S+1) для блока. dy = axis0, dx = axis1."""
    ref = img1[off_y:off_y + 8, off_x:off_x + 8]
    sy0, sx0 = off_y - search_size, off_x - search_size
    sh = 8 + 2 * search_size
    region = img2[sy0:sy0 + sh, sx0:sx0 + sh]
    windows = np.lib.stride_tricks.sliding_window_view(region, (8, 8))
    return np.sum(np.abs(windows - ref[None, None, :, :]), axis=(2, 3))


_POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def _census(img, eps=0):
    """3x3 census transform: на пиксель 8 бит (сосед ярче центра+eps -> 1).
    Инвариантен к монотонному изменению яркости (автоэкспозиция OV7725).
    eps>0 — порог: мелкая рябь у равных пикселей не переворачивает бит
    (устойчивость к шуму; ценой почти-точной gain-инвариантности).
    На STM32: 8 сравнений + сдвиги, popcount через таблицу."""
    p = np.pad(img, 1, mode="edge").astype(np.int16)
    c = p[1:-1, 1:-1]
    sig = np.zeros(img.shape, dtype=np.uint8)
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neigh = p[1 + dy:1 + dy + img.shape[0], 1 + dx:1 + dx + img.shape[1]]
            sig |= (neigh > c + eps).astype(np.uint8) << k
            k += 1
    return sig


def _census_cost_volume(c1, c2, off_x, off_y, search_size):
    """Cost-объём на census: сумма Hamming-расстояний (popcount XOR) по блоку 8x8."""
    ref = c1[off_y:off_y + 8, off_x:off_x + 8]
    sy0, sx0 = off_y - search_size, off_x - search_size
    sh = 8 + 2 * search_size
    region = c2[sy0:sy0 + sh, sx0:sx0 + sh]
    windows = np.lib.stride_tricks.sliding_window_view(region, (8, 8))
    x = np.bitwise_xor(windows, ref[None, None, :, :])
    return _POPCNT[x].sum(axis=(2, 3))


def _highpass(img, k):
    """Вычесть локальное среднее (бокс-фильтр) -> инвариантность к сдвигу яркости.
    Дешёвая альтернатива census для автоэкспозиции: та же польза на яркости, но
    штраф на шум вдвое меньше (мягкое ядро k~13). Сепарабельный фильтр, быстро."""
    from scipy.ndimage import uniform_filter
    g = img.astype(np.float32)
    return np.clip(g - uniform_filter(g, k) + 128, 0, 255).astype(np.uint8)


def _downsample2(img):
    """2x2 box-даунскейл (как на STM32: 4 пикселя -> среднее). Размеры чётные."""
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    return img[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def _coarse_vol_zsad(i1p, i2p, off_x, off_y, pad, radius):
    """Грубый cost-объём на zero-mean SAD (инвариантен к сдвигу яркости).
    Только для грубой стадии пирамиды: убирает яркостный bias, из-за которого
    грубый поиск на ½-разрешении выбирал неверный минимум при автоэкспозиции.
    Точность даёт fine-стадия на обычном SAD."""
    ref = i1p[off_y + pad:off_y + pad + 8, off_x + pad:off_x + pad + 8].astype(np.float64)
    ref = ref - ref.mean()
    sy0, sx0 = off_y + pad - radius, off_x + pad - radius
    sh = 8 + 2 * radius
    region = i2p[sy0:sy0 + sh, sx0:sx0 + sh].astype(np.float64)
    w = np.lib.stride_tricks.sliding_window_view(region, (8, 8))
    wm = w.mean(axis=(2, 3), keepdims=True)
    return np.sum(np.abs((w - wm) - ref[None, None, :, :]), axis=(2, 3))


def _sad_vol_around(i1p, i2p, off_x, off_y, pad, cdx, cdy, radius):
    """SAD-объём (2r+1)² вокруг предсказанного сдвига (cdx,cdy). i*p — padded."""
    ref = i1p[off_y + pad:off_y + pad + 8, off_x + pad:off_x + pad + 8]
    sy0 = off_y + pad + cdy - radius
    sx0 = off_x + pad + cdx - radius
    sh = 8 + 2 * radius
    region = i2p[sy0:sy0 + sh, sx0:sx0 + sh]
    w = np.lib.stride_tricks.sliding_window_view(region, (8, 8))
    return np.sum(np.abs(w - ref[None, None, :, :]), axis=(2, 3))


def _subpixel_8dir(img1, img2, off1_x, off1_y, off2_x, off2_y, best_dist):
    """Оригинальный метод: SAD по 8 направлениям, шаг 0.5px (как в px4flow)."""
    block1 = img1[off1_y:off1_y + 8, off1_x:off1_x + 8]
    region = img2[off2_y - 1:off2_y + 9, off2_x - 1:off2_x + 9]

    def sh(dy, dx):
        return region[1 + dy:1 + dy + 8, 1 + dx:1 + dx + 8]

    center, right, down = sh(0, 0), sh(0, 1), sh(1, 0)
    down_left, left = sh(1, -1), sh(0, -1)
    up_left, up, up_right = sh(-1, -1), sh(-1, 0), sh(-1, 1)
    down_right = sh(1, 1)

    s0 = (center + right) // 2
    s1 = (down + down_right) // 2
    s2 = (center + down) // 2
    s3 = (down + down_left) // 2
    s4 = (center + left) // 2
    s5 = (up + up_left) // 2
    s6 = (center + up) // 2
    s7 = (up + up_right) // 2
    t1, t3 = (s0 + s1) // 2, (s3 + s4) // 2
    t5, t7 = (s4 + s5) // 2, (s7 + s0) // 2

    acc = np.empty(8, dtype=np.int64)
    for k, sub in enumerate([s0, t1, s2, t3, s4, t5, s6, t7]):
        acc[k] = np.sum(np.abs(block1 - sub))

    # как в оригинале: меняем направление только если субпиксель ЛУЧШЕ
    # целочисленного минимума, иначе mindir=8 -> нулевой субсдвиг
    mindist, mindir = best_dist, 8
    for k in range(8):
        if acc[k] < mindist:
            mindist, mindir = acc[k], k
    sx = 0.5 if mindir in (0, 1, 7) else (-0.5 if mindir in (3, 4, 5) else 0.0)
    sy = -0.5 if mindir in (5, 6, 7) else (0.5 if mindir in (1, 2, 3) else 0.0)
    return sx, sy


def _parab(sad, axis_idx, center_idx):
    """Параболический сдвиг по одной оси из трёх SAD-точек. |out| < 0.5."""
    c = sad[center_idx]
    lo = sad[center_idx - 1] if center_idx - 1 >= 0 else c + 1
    hi = sad[center_idx + 1] if center_idx + 1 < len(sad) else c + 1
    denom = (lo - 2 * c + hi)
    if denom <= 0:
        return 0.0
    d = 0.5 * (lo - hi) / denom
    return float(np.clip(d, -0.5, 0.5))


# pixel-locking: SAD+парабола сжимает субпиксель к целым (истина 0.3 -> оценка ~0.1).
# LUT ниже калиброван на СИНТЕТИЧЕСКОМ bilinear-сдвиге (scipy).
# ⚠️ ВАЖНО: на синтетике коррекция даёт −70% (clean), но на РЕАЛЬНОМ баге ХУЖЕ (+29%):
# LUT переобучен под артефакт bilinear-генерации, реальная камера имеет другую
# субпиксельную статистику + шум/смаз. НЕ включать без РЕ-КАЛИБРОВКИ LUT на парах
# кадров реальной камеры. Флаг оставлен для такой калибровки, по умолчанию OFF.
# Урок: синтетический выигрыш ≠ реальный (см. docs/EXPERIMENTS.md).
_PLK_MEAS = np.array([-0.33, -0.23, -0.12, -0.064, -0.021, 0.011, 0.049, 0.099, 0.20, 0.28])
_PLK_TRUE = np.array([-0.45, -0.40, -0.30, -0.20, -0.05, 0.05, 0.20, 0.30, 0.40, 0.45])


def _correct_subpix(s):
    """Инверсия pixel-locking по калиброванному LUT (монотонно, клип к ±0.5)."""
    return float(np.clip(np.interp(s, _PLK_MEAS, _PLK_TRUE), -0.5, 0.5))


def _peak_curv(vol, bi):
    """Кривизна (2-я разность) минимума SAD по x и y — острота корр. пика.
    Высокая = чёткий пик (надёжная локализация); низкая = плоский пик
    (motion blur / слабая текстура). Замерено: резкая ~1400, блюр σ=2 ~195."""
    y, x = bi
    acc = 0.0
    n = 0
    if 0 < x < vol.shape[1] - 1:
        acc += vol[y, x - 1] - 2 * vol[y, x] + vol[y, x + 1]; n += 1
    if 0 < y < vol.shape[0] - 1:
        acc += vol[y - 1, x] - 2 * vol[y, x] + vol[y + 1, x]; n += 1
    return acc / n if n else 0.0


def compute_flow_improved(image1, image2,
                          search_size=4,
                          flow_feature_threshold=30,
                          flow_value_threshold=3000,
                          use_median=False,
                          use_uniqueness=False,
                          use_parabolic=False,
                          use_fb_check=False,
                          use_census=False,
                          use_pyramid=False,
                          use_boundary_reject=False,
                          use_peak_quality=False,
                          use_mad=False,
                          mad_k=2.0,
                          use_highpass=False,
                          hp_kernel=13,
                          use_subpix_correction=False,
                          uniqueness_ratio=1.25,
                          fb_tol=1.0,
                          census_eps=6,
                          census_value_threshold=140,
                          pyr_refine=2,
                          peak_curv_ref=400.0):
    # high-pass препроцесс (вычесть локальное среднее) -> устойчивость к яркости
    if use_highpass:
        image1 = _highpass(image1, hp_kernel)
        image2 = _highpass(image2, hp_kernel)

    h, w = image1.shape
    positions = _block_positions(w, search_size)
    img1 = image1.astype(np.int32)
    img2 = image2.astype(np.int32)
    winmin = -search_size

    # census инвариантен к яркости -> метрика в битах, свой порог; субпиксель
    # на cost-объёме (параболический), 8-напр. интенсивностный метод не применим
    if use_census:
        c1, c2 = _census(image1, census_eps), _census(image2, census_eps)
        value_threshold = census_value_threshold
    else:
        value_threshold = flow_value_threshold

    # пирамида: грубый поиск на 1/2 разрешении (диапазон ±2*search_size),
    # уточнение на полном ±pyr_refine. Снимает потолок скорости search_size.
    if use_pyramid:
        padf = 2 * search_size + pyr_refine
        padh = search_size
        i1p = np.pad(img1, padf, mode="edge")
        i2p = np.pad(img2, padf, mode="edge")
        h1p = np.pad(_downsample2(img1), padh, mode="edge")
        h2p = np.pad(_downsample2(img2), padh, mode="edge")

    fxs, fys = [], []
    curvs = []

    for (off_x, off_y) in positions:
        if _compute_diff(image1, off_x, off_y) < flow_feature_threshold:
            continue

        if use_pyramid:
            # грубо на половинном разрешении (zero-mean SAD -> устойчив к яркости)
            coarse = _coarse_vol_zsad(h1p, h2p, off_x // 2, off_y // 2,
                                      padh, search_size)
            cb = np.unravel_index(int(np.argmin(coarse)), coarse.shape)
            # грубый минимум на краю окна -> движение за пределом даже пирамиды
            if use_boundary_reject and (cb[0] in (0, 2 * search_size) or
                                        cb[1] in (0, 2 * search_size)):
                continue
            cdx, cdy = 2 * (cb[1] - search_size), 2 * (cb[0] - search_size)
            # уточнение на полном разрешении вокруг 2*грубого
            sad = _sad_vol_around(i1p, i2p, off_x, off_y, padf,
                                  cdx, cdy, pyr_refine)
            bi = np.unravel_index(int(np.argmin(sad)), sad.shape)
            best_dist = int(sad[bi])
            best_dx = cdx + (bi[1] - pyr_refine)
            best_dy = cdy + (bi[0] - pyr_refine)
            if best_dist >= value_threshold:
                continue
            # параболический субпиксель на уточняющем объёме; uniqueness/fb
            # с пирамидой не комбинируем (отдельные эксперименты)
            sub_y = _parab(sad[:, bi[1]], 0, bi[0])
            sub_x = _parab(sad[bi[0], :], 1, bi[1])
            if use_subpix_correction:
                sub_x, sub_y = _correct_subpix(sub_x), _correct_subpix(sub_y)
            fxs.append(best_dx + sub_x)
            fys.append(best_dy + sub_y)
            if use_peak_quality:
                curvs.append(_peak_curv(sad, bi))
            continue

        if use_census:
            sad = _census_cost_volume(c1, c2, off_x, off_y, search_size)
        else:
            sad = _sad_volume(img1, img2, off_x, off_y, search_size)
        flat = sad.ravel()
        best_flat = int(np.argmin(flat))
        best_dist = int(flat[best_flat])
        bi = np.unravel_index(best_flat, sad.shape)
        best_dy, best_dx = bi[0] + winmin, bi[1] + winmin

        if best_dist >= value_threshold:
            continue

        # минимум на краю окна поиска -> истинный минимум скорее снаружи
        # (движение за search_size). Иначе — confident garbage (q высокий, врёт).
        if use_boundary_reject and (abs(best_dx) == search_size or
                                    abs(best_dy) == search_size):
            continue

        # --- imp3: уникальность минимума (anti-aperture) ---
        # peak-ratio: второй минимум (вне 3x3 у лучшего) должен быть заметно
        # хуже лучшего. У повторяющейся текстуры (полосы/плитка) второй минимум
        # ≈ лучшему -> блок неоднозначен -> отбросить (иначе confident garbage).
        if use_uniqueness:
            mask = np.ones(sad.shape, dtype=bool)
            y0, x0 = bi
            mask[max(0, y0 - 1):y0 + 2, max(0, x0 - 1):x0 + 2] = False
            others = sad[mask]
            if others.size and np.min(others) <= best_dist * uniqueness_ratio:
                continue

        # --- imp2: forward-backward consistency ---
        if use_fb_check:
            # блок из img2 в найденной позиции ищем обратно в img1
            b2x, b2y = off_x + best_dx, off_y + best_dy
            if (search_size <= b2x <= w - 8 - search_size and
                    search_size <= b2y <= h - 8 - search_size):
                if use_census:
                    sad_b = _census_cost_volume(c2, c1, b2x, b2y, search_size)
                else:
                    sad_b = _sad_volume(img2, img1, b2x, b2y, search_size)
                bb = np.unravel_index(int(np.argmin(sad_b)), sad_b.shape)
                bdy, bdx = bb[0] + winmin, bb[1] + winmin
                if abs(bdx + best_dx) > fb_tol or abs(bdy + best_dy) > fb_tol:
                    continue

        # --- субпиксель ---
        if use_parabolic or use_census:
            col = sad[:, bi[1]]
            row = sad[bi[0], :]
            sub_y = _parab(col, 0, bi[0])
            sub_x = _parab(row, 1, bi[1])
        else:
            sub_x, sub_y = _subpixel_8dir(img1, img2, off_x, off_y,
                                          off_x + best_dx, off_y + best_dy,
                                          best_dist)

        if use_subpix_correction:
            sub_x, sub_y = _correct_subpix(sub_x), _correct_subpix(sub_y)
        fxs.append(best_dx + sub_x)
        fys.append(best_dy + sub_y)
        if use_peak_quality:
            curvs.append(_peak_curv(sad, bi))

    meancount = len(fxs)
    if meancount <= 10:
        return 0, 0.0, 0.0

    fxs = np.asarray(fxs)
    fys = np.asarray(fys)

    # MAD-фильтр: выкинуть блоки, отклонившиеся >mad_k·MAD от медианы (шумные/
    # яркостно-сбитые/окклюдер), затем ре-медиана согласного большинства.
    # Бьёт шум/яркость, не ломая выбросы (в отличие от взвешивания по кривизне).
    if use_mad:
        mx, my = np.median(fxs), np.median(fys)
        dev = np.hypot(fxs - mx, fys - my)
        mad = np.median(dev) + 1e-6
        keep = dev <= mad_k * mad
        if keep.sum() >= 5:
            fxs, fys = fxs[keep], fys[keep]

    if use_median:
        flow_x = float(np.median(fxs))
        flow_y = float(np.median(fys))
    else:
        flow_x = float(np.mean(fxs))
        flow_y = float(np.mean(fys))

    quality = min(int(meancount * 255 / (NUM_BLOCKS * NUM_BLOCKS)), 255)

    # motion-blur / плохая локализация: плоский корр. пик (низкая кривизна)
    # -> понижаем quality пропорционально остроте пика (EKF доверяет меньше)
    if use_peak_quality and curvs:
        conf = min(float(np.mean(curvs)) / peak_curv_ref, 1.0)
        quality = int(quality * conf)

    return quality, flow_x, flow_y


if __name__ == "__main__":
    # self-check: OFF == оригинал, и парабола реально уточняет субпиксель
    from functools import partial
    import benchmark as B
    from px4flow_fast import compute_flow_fast

    i1, i2 = B.make_pair(7, 1.3, 0.7)
    assert compute_flow_improved(i1, i2) == compute_flow_fast(i1, i2), \
        "флаги OFF должны давать оригинальный результат"

    base = B.run_scenario(compute_flow_fast)
    par = B.run_scenario(
        lambda a, b: compute_flow_improved(a, b, use_median=True, use_parabolic=True))
    assert par["rmse"] < base["rmse"], "медиана+парабола должны улучшать RMSE"
    print(f"OK: clean RMSE {base['rmse']:.3f} -> {par['rmse']:.3f}")

    # census: инвариантность к яркости — bright должен стать как clean
    cen = partial(compute_flow_improved, use_median=True, use_parabolic=True,
                  use_census=True, census_eps=10, census_value_threshold=320)
    c_clean = B.run_scenario(cen)
    c_bright = B.run_scenario(cen, gain=1.12, bias=8.0)
    assert abs(c_bright["rmse"] - c_clean["rmse"]) < 0.05, \
        "census должен быть инвариантен к яркости (bright ≈ clean)"
    print(f"OK census: bright RMSE {c_bright['rmse']:.3f} ≈ clean {c_clean['rmse']:.3f}")

    # пирамида: быстрое движение >search_size — baseline слепнет, пирамида видит
    i1, i2 = B.make_pair(3, 7.0, -5.0)
    _, fx, fy = compute_flow_improved(i1, i2, use_median=True, use_pyramid=True)
    assert abs(fx - 7.0) < 1 and abs(fy + 5.0) < 1, \
        "пирамида должна ловить быстрое движение вне ±search_size"
    print(f"OK pyramid: fast (7,-5) -> ({fx:+.2f},{fy:+.2f})")

    # boundary_reject: на движении за search_size quality должен упасть
    # (честный отказ вместо confident-garbage), а не остаться 255
    i1, i2 = B.make_pair(3, 8.0, 0.0)   # 8px > search_size=4, без пирамиды
    qr, _, _ = compute_flow_improved(i1, i2, use_median=True, use_parabolic=True)
    qb, _, _ = compute_flow_improved(i1, i2, use_median=True, use_parabolic=True,
                                     use_boundary_reject=True)
    assert qb < qr, f"boundary_reject должен ронять quality за диапазоном: {qr}->{qb}"
    print(f"OK boundary: out-of-range quality {qr} -> {qb}")

    # peak_quality: motion blur -> плоский пик -> quality падает (поток не тронут)
    import cv2
    from scipy.ndimage import shift as _shift
    from _paths import REAL_FRAMES
    sc = np.load(sorted(REAL_FRAMES.glob("*.npy"))[0]).astype(np.float64)
    sh = _shift(sc, (0, 2.0), order=1, mode="reflect")
    s1 = sc[16:80, 16:80].astype(np.uint8); s2 = sh[16:80, 16:80].astype(np.uint8)
    b1 = cv2.GaussianBlur(s1, (0, 0), 2.5); b2 = cv2.GaussianBlur(s2, (0, 0), 2.5)
    pk = partial(compute_flow_improved, use_median=True, use_parabolic=True, use_peak_quality=True)
    q_sharp = pk(s1, s2)[0]; q_blur = pk(b1, b2)[0]
    assert q_blur < q_sharp, f"peak_quality должен ронять quality на блюре: {q_sharp}->{q_blur}"
    print(f"OK peak_quality: sharp q {q_sharp} -> blur q {q_blur}")

    # MAD-фильтр: не ломает нормальное движение (робастный ре-медиан большинства)
    i1, i2 = B.make_pair(7, 1.3, 0.7)
    _, fx, fy = compute_flow_improved(i1, i2, use_median=True, use_parabolic=True, use_mad=True)
    assert abs(fx - 1.3) < 0.5 and abs(fy - 0.7) < 0.5, "MAD не должен ломать нормальное движение"
    print(f"OK MAD: normal (1.3,0.7) -> ({fx:+.2f},{fy:+.2f})")

    # high-pass: инвариантность к яркости (bright ~ clean), дешевле census, меньше шум-налог
    hp = partial(compute_flow_improved, use_median=True, use_parabolic=True, use_highpass=True)
    hp_clean = B.run_scenario(hp)
    hp_bright = B.run_scenario(hp, gain=1.12, bias=8.0)
    assert abs(hp_bright["rmse"] - hp_clean["rmse"]) < 0.1, \
        "high-pass должен убирать разницу яркости (bright ≈ clean)"
    print(f"OK highpass: bright RMSE {hp_bright['rmse']:.3f} ≈ clean {hp_clean['rmse']:.3f}")

    # subpix correction: убирает pixel-locking -> резкое падение clean-ошибки
    base_c = B.run_scenario(partial(compute_flow_improved, use_median=True, use_parabolic=True))
    corr_c = B.run_scenario(partial(compute_flow_improved, use_median=True, use_parabolic=True,
                                    use_subpix_correction=True))
    assert corr_c["rmse"] < base_c["rmse"], "коррекция субпикселя должна снижать clean-ошибку"
    print(f"OK subpix: clean RMSE {base_c['rmse']:.3f} -> {corr_c['rmse']:.3f}")
