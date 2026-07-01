"""
benchmark.py — честный стенд для сравнения версий optical flow алгоритма.

В отличие от synthetic_test.py (двигает окно на ЦЕЛЫЕ пиксели -> ошибка
всегда ровно 0, ничего не меряет), здесь:

  * субпиксельный сдвиг текстуры через bilinear (scipy) -> виден реальный
    субпиксельный промах;
  * аддитивный шум сенсора;
  * дрейф яркости/усиления (автоэкспозиция OV7725 постоянно подстраивается);
  * выбросы — движущийся объект-окклюдер на части кадра (тень/птица/машина),
    который тянет среднее в сторону.

Метрика — RMSE по (flow - истинный сдвиг) в пикселях, по сетке из нескольких
текстур × нескольких векторов движения. Меньше RMSE = лучше.

Знак: img2 = текстура img1, сдвинутая на (+tdx,+tdy). Алгоритм ищет блок
img1 внутри img2 -> возвращает flow ≈ (+tdx,+tdy). Поэтому ground truth для
flow здесь = (tdx, tdy) напрямую, без путаницы со знаком камеры.
"""

import numpy as np
from scipy.ndimage import shift as nd_shift

WIN = 64
PAD = 12  # запас вокруг окна, чтобы bilinear-сдвиг не цеплял край

# Векторы движения в пределах search_size=4 (px/кадр), все субпиксельные
MOTIONS = [
    (1.3, 0.7), (2.4, -1.6), (0.5, 0.5), (-1.8, 2.2),
    (3.1, -0.4), (0.0, 1.5), (-2.7, -2.7), (1.0, -3.3),
]
SEEDS = [1, 7, 13, 21, 42]


def make_scene(size, seed):
    rng = np.random.default_rng(seed)
    scene = rng.integers(40, 215, size=(size, size)).astype(np.float64)
    for _ in range(40):
        cx, cy = rng.integers(0, size, size=2)
        r = rng.integers(5, 25)
        yy, xx = np.ogrid[:size, :size]
        scene[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = rng.integers(40, 215)
    return scene


def make_pair(seed, tdx, tdy, noise=0.0, gain=1.0, bias=0.0, occluder=None):
    """img1, img2 размером WIN. img2 = img1, сдвинутая на (tdx,tdy) субпиксельно."""
    full = WIN + 2 * PAD
    scene = make_scene(full + 16, seed)
    c = (full + 16) // 2
    crop = scene[c - full // 2:c + full // 2, c - full // 2:c + full // 2]

    # сдвиг текстуры: shift=(y,x) -> текстура уезжает на (+x,+y)
    shifted = nd_shift(crop, shift=(tdy, tdx), order=1, mode="reflect")

    s = PAD
    img1 = crop[s:s + WIN, s:s + WIN].copy()
    img2 = shifted[s:s + WIN, s:s + WIN].copy()

    if occluder is not None:
        # квадрат-окклюдер, движущийся в ДРУГУЮ сторону (выброс)
        # выброс-МЕНЬШИНСТВО: маленький объект ~14px (задевает 1-3 блока из 25),
        # движется в другую сторону. Имитирует тень/птицу/машину в кадре.
        oxd, oyd = occluder
        rng = np.random.default_rng(seed + 999)
        osz = 14
        patch = rng.integers(40, 215, size=(osz, osz)).astype(np.float64)
        y0, x0 = 6, 6
        img1[y0:y0 + osz, x0:x0 + osz] = patch
        # тот же патч в img2 сдвинут на (oxd,oyd), а не на (tdx,tdy)
        py0 = int(round(y0 + oyd))
        px0 = int(round(x0 + oxd))
        img2[py0:py0 + osz, px0:px0 + osz] = patch

    rng = np.random.default_rng(seed * 31 + 5)
    if noise > 0:
        img2 = img2 + rng.normal(0, noise, img2.shape)
        img1 = img1 + rng.normal(0, noise, img1.shape)
    if gain != 1.0 or bias != 0.0:
        img2 = img2 * gain + bias

    img1 = np.clip(img1, 0, 255).astype(np.uint8)
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    return img1, img2


SCENARIOS = {
    "clean   (субпиксель)":      dict(),
    "noise   (шум σ=6)":         dict(noise=6.0),
    "bright  (gain1.12 +8)":     dict(gain=1.12, bias=8.0),
    "outlier (окклюдер)":        dict(occluder=(-3, 3)),
}


def run_scenario(flow_fn, **kw):
    errs, quals, nvalid, total = [], [], 0, 0
    for seed in SEEDS:
        for tdx, tdy in MOTIONS:
            total += 1
            img1, img2 = make_pair(seed, tdx, tdy, **kw)
            q, fx, fy = flow_fn(img1, img2)
            if q > 0:
                nvalid += 1
                quals.append(q)
                errs.append((fx - tdx, fy - tdy))
    if not errs:
        return None
    e = np.array(errs)
    rmse = float(np.sqrt(np.mean(e[:, 0] ** 2 + e[:, 1] ** 2)))
    return dict(rmse=rmse, quality=float(np.mean(quals)),
                valid=nvalid, total=total)


def bench(flow_fn, label):
    print(f"\n=== {label} ===")
    print(f"{'сценарий':<24} {'RMSE px':>9} {'quality':>8} {'valid':>8}")
    results = {}
    for name, kw in SCENARIOS.items():
        r = run_scenario(flow_fn, **kw)
        results[name] = r
        if r is None:
            print(f"{name:<24} {'—':>9} {'—':>8}   нет валидных")
        else:
            print(f"{name:<24} {r['rmse']:>9.3f} {r['quality']:>8.0f} "
                  f"{r['valid']:>4d}/{r['total']:<3d}")
    return results


def compare(base, new, base_label, new_label):
    print(f"\n=== Δ {new_label} vs {base_label} (RMSE, меньше = лучше) ===")
    print(f"{'сценарий':<24} {base_label[:10]:>11} {new_label[:10]:>11} {'Δ%':>8}")
    for name in SCENARIOS:
        b, n = base.get(name), new.get(name)
        if b is None or n is None:
            print(f"{name:<24} {'—':>11} {'—':>11}")
            continue
        d = (n['rmse'] - b['rmse']) / b['rmse'] * 100 if b['rmse'] else 0
        print(f"{name:<24} {b['rmse']:>11.3f} {n['rmse']:>11.3f} {d:>+7.1f}%")


if __name__ == "__main__":
    from px4flow_fast import compute_flow_fast
    bench(compute_flow_fast, "BASELINE (px4flow_fast, оригинал)")
