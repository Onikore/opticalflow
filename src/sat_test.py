"""
sat_test.py — прогон кадров бага против скачанной спутниковой подложки.

Использование (после fetch_sat.py на машине с интернетом):
    python3 src/sat_test.py data/sat/google_z19.json
    python3 src/sat_test.py data/sat/esri_z19.json --cache data/bag_flight/bag2_cache.npz

Три стадии (все автоматические):
  0. Калибровка конвенции осей: наши локальные оси из бага не привязаны к истинному
     северу (бортовой кадр мог быть повёрнут/отражён). Перебираются 8 вариантов
     (зеркала кадра × знак yaw × смещение 180°) на подвыборке с широким поиском;
     выбирается вариант с максимумом уверенных матчей и минимальным разбросом
     подразумеваемого сдвига.
  1. Оценка постоянного сдвига баг-кадр ↔ тайл (геопривязка тайлов гуляет 2-10 м,
     плюс неизвестное начало локальной рамки бага): медиана (fix - prior).
  2. Основной прогон: прайор = бортовая поза + сдвиг + шум ±6 м (имитация дрейфа
     одометрии), поиск 12 м. Метрики: доля уверенных, медиана/p90 разброса
     (точность матчинга относительно согласованного сдвига).

Допущение: локальный (0,0) бага = точка взлёта = ref-точка меты fetch_sat
(координата из gps бага). Если ref другой — сдвиг поглотится стадией 1
(лишь бы влез в поиск стадии 0, по умолчанию ±40 м).
"""
import argparse
import json
import sys

import numpy as np
import cv2

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import map_localize as ML
from _paths import DATA

VARIANTS = [  # (имя, зеркало кадра, знак yaw, смещение yaw)
    ("id", None, 1, 0.0), ("id+180", None, 1, np.pi),
    ("negyaw", None, -1, 0.0), ("negyaw+180", None, -1, np.pi),
    ("flipv", 0, 1, 0.0), ("flipv+180", 0, 1, np.pi),
    ("fliph", 1, 1, 0.0), ("fliph+180", 1, 1, np.pi),
]


def prep(frame, mirror):
    return frame if mirror is None else cv2.flip(frame, mirror)


def run(meta_path, cache_path, focal_scale=551.8 / 480.0):
    meta = json.load(open(meta_path))
    gm = ML.GeoMap.from_image(meta["image_for_geomap"], meta["enu_x0"],
                              meta["enu_y0"], meta["gsd"], highpass=True)
    d = np.load(cache_path)
    W = int(d["W"]); focal = focal_scale * W
    imgs, it = d["imgs"], d["img_t"]
    h = np.interp(it, d["rf_t"], d["rf_h"])
    yaw = np.interp(it, d["imu_t"], np.unwrap(d["yaw"]))
    Q = np.column_stack([np.interp(it, d["lp_t"], d["lp_xy"][:, 0]),
                         np.interp(it, d["lp_t"], d["lp_xy"][:, 1])])
    good = np.where(h > 5)[0]
    sub = good[:: max(1, len(good) // 30)]        # ~30 кадров на калибровку

    # --- стадия 0: конвенция осей ---
    print("=== стадия 0: калибровка конвенции (широкий поиск 40 м) ===")
    cands = []
    for name, mir, ys, yo in VARIANTS:
        offs = []
        for i in sub:
            r = gm.localize(prep(imgs[i], mir), ys * yaw[i] + yo, h[i], focal,
                            Q[i], search_m=40)
            if r:
                offs.append([r[0] - Q[i, 0], r[1] - Q[i, 1]])
        n = len(offs)
        spread = (np.median(np.abs(np.array(offs) - np.median(offs, 0)))
                  if n >= 5 else 99.0)
        print(f"  {name:<11} матчей {n:>2}/{len(sub)}  разброс сдвига {spread:5.1f} м")
        cands.append((n - spread * 0.5, name, mir, ys, yo, n))
    cands.sort(reverse=True)

    # финальный выбор: топ-2 кандидата проверяются узким поиском (12 м) со своим
    # сдвигом — правильная конвенция даёт больше уверенных матчей
    def narrow_score(mir, ys, yo):
        offs = [(r[0] - Q[i, 0], r[1] - Q[i, 1]) for i in sub
                if (r := gm.localize(prep(imgs[i], mir), ys * yaw[i] + yo,
                                     h[i], focal, Q[i], search_m=40))]
        if len(offs) < 5:
            return -1, None
        o = np.median(np.array(offs), 0)
        cnt = sum(1 for i in sub
                  if gm.localize(prep(imgs[i], mir), ys * yaw[i] + yo, h[i],
                                 focal, Q[i] + o, search_m=12))
        return cnt, o
    top = []
    for _, name, mir, ys, yo, n in cands[:2]:
        cnt, o = narrow_score(mir, ys, yo)
        print(f"  верификация {name:<11} узкий поиск: {cnt} матчей")
        top.append((cnt, name, mir, ys, yo, n))
    top.sort(reverse=True)
    _, name, mir, ys, yo, n0 = top[0]
    print(f"выбрано: {name} ({n0} матчей в широком, {top[0][0]} в узком)")
    if n0 < 5:
        print("!! мало матчей даже в лучшем варианте — снимок не того места, "
              "не тот масштаб, или сцена изменилась радикально")
        return

    # --- стадия 1: постоянный сдвиг ---
    offs = []
    for i in sub:
        r = gm.localize(prep(imgs[i], mir), ys * yaw[i] + yo, h[i], focal,
                        Q[i], search_m=40)
        if r:
            offs.append([r[0] - Q[i, 0], r[1] - Q[i, 1]])
    off = np.median(np.array(offs), 0)
    print(f"=== стадия 1: сдвиг баг↔тайл = ({off[0]:+.1f}, {off[1]:+.1f}) м "
          f"(геопривязка тайла + начало локальной рамки) ===")

    # --- стадия 2: основной прогон ---
    rng = np.random.default_rng(3)
    errs, n_try = [], 0
    for i in good[::5]:
        n_try += 1
        prior = Q[i] + off + rng.uniform(-6, 6, 2)
        r = gm.localize(prep(imgs[i], mir), ys * yaw[i] + yo, h[i], focal,
                        prior, search_m=12)
        if r is None:
            continue
        errs.append(np.hypot(r[0] - (Q[i, 0] + off[0]),
                             r[1] - (Q[i, 1] + off[1])))
    errs = np.array(errs)
    print(f"=== стадия 2: {len(errs)}/{n_try} уверенных "
          f"({100 * len(errs) / max(n_try, 1):.0f}%) ===")
    if len(errs):
        print(f"  точность матчинга: медиана {np.median(errs):.2f} м, "
              f"p90 {np.percentile(errs, 90):.2f} м")
    print("\nвывод для бюджета миссии: доля фиксов и медиана выше — это реальные "
          "числа кросс-сезонного map-matching для этого провайдера")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("meta", help="json из fetch_sat.py (напр. data/sat/google_z19.json)")
    ap.add_argument("--cache", default=str(DATA / "bag_flight" / "bag_cache.npz"))
    a = ap.parse_args()
    run(a.meta, a.cache)
