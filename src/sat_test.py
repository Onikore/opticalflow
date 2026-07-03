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

VARIANTS = [  # (имя, зеркало кадра, знак yaw, смещение yaw) — вкл. ±90°
    (f"{m or 'id'}{s:+d}y{int(np.degrees(o))}", mir, s, o)
    for m, mir in [("id", None), ("flipv", 0), ("fliph", 1)]
    for s in (1, -1)
    for o in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
]


def fit_rigid(P, F):
    """Жёсткая привязка F ≈ R(θ)·P + t (поворот+сдвиг, без масштаба/зеркала).
    Локальная рамка бага может быть ПОВЁРНУТА относительно истинного севера."""
    P = np.asarray(P); F = np.asarray(F)
    Pc, Fc = P - P.mean(0), F - F.mean(0)
    th = np.arctan2((Pc[:, 0] * Fc[:, 1] - Pc[:, 1] * Fc[:, 0]).sum(),
                    (Pc * Fc).sum())
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    t = F.mean(0) - R @ P.mean(0)
    res = np.sqrt(np.mean(np.sum((F - (P @ R.T + t)) ** 2, 1)))
    return th, R, t, res


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

    # финальный выбор: топ-3 кандидата -> жёсткая привязка (поворот θ + сдвиг)
    # из пар (Q_i, fix_i), верификация узким поиском с прайором R·Q+t.
    # Поворот нужен: локальная рамка бага может быть повёрнута к северу (в т.ч. 90°).
    def rigid_and_score(mir, ys, yo):
        P, F = [], []
        for i in sub:
            r = gm.localize(prep(imgs[i], mir), ys * yaw[i] + yo, h[i], focal,
                            Q[i], search_m=40)
            if r:
                P.append(Q[i]); F.append([r[0], r[1]])
        if len(P) < 5:
            return -1, None, None, None
        th, R, t, res = fit_rigid(P, F)
        cnt = sum(1 for i in sub
                  if gm.localize(prep(imgs[i], mir), ys * yaw[i] + yo, h[i],
                                 focal, R @ Q[i] + t, search_m=12))
        return cnt, th, R, t
    top = []
    for _, name, mir, ys, yo, n in cands[:3]:
        cnt, th, R, t = rigid_and_score(mir, ys, yo)
        print(f"  верификация {name:<12} узкий: {cnt:>2} матчей, "
              f"θ={np.degrees(th) if th is not None else 0:+.1f}°")
        top.append((cnt, name, mir, ys, yo, th, R, t))
    top.sort(key=lambda x: -x[0])
    cnt0, name, mir, ys, yo, th, R, t = top[0]
    print(f"выбрано: {name} (узких матчей {cnt0})")
    if cnt0 < 5:
        print("!! мало матчей — снимок не того места/масштаба или сцена изменилась")
        return None

    print(f"=== стадия 1: рамка бага -> тайл: поворот θ={np.degrees(th):+.1f}°, "
          f"сдвиг t=({t[0]:+.1f},{t[1]:+.1f}) м ===")

    # --- стадия 2: основной прогон (прайор = R·Q+t + дрейф-шум) ---
    rng = np.random.default_rng(3)
    errs, n_try = [], 0
    for i in good[::5]:
        n_try += 1
        true_t = R @ Q[i] + t
        prior = true_t + rng.uniform(-6, 6, 2)
        r = gm.localize(prep(imgs[i], mir), ys * yaw[i] + yo, h[i], focal,
                        prior, search_m=12)
        if r is None:
            continue
        errs.append(np.hypot(r[0] - true_t[0], r[1] - true_t[1]))
    errs = np.array(errs)
    print(f"=== стадия 2: {len(errs)}/{n_try} уверенных "
          f"({100 * len(errs) / max(n_try, 1):.0f}%) ===")
    if len(errs):
        print(f"  точность матчинга: медиана {np.median(errs):.2f} м, "
              f"p90 {np.percentile(errs, 90):.2f} м")
    print("\nвывод для бюджета миссии: доля фиксов и медиана выше — это реальные "
          "числа кросс-сезонного map-matching для этого провайдера")
    return dict(mirror=mir, yaw_sign=ys, yaw_off=yo, theta=th, R=R, t=t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("meta", help="json из fetch_sat.py (напр. data/sat/google_z19.json)")
    ap.add_argument("--cache", default=str(DATA / "bag_flight" / "bag_cache.npz"))
    a = ap.parse_args()
    run(a.meta, a.cache)
