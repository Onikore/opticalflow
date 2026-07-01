r"""
derotate_test.py — стенд №1: де-ротация потока по гироскопу.

У камеры вниз вращение корпуса подделывает трансляцию: наклон (pitch/roll)
с угловой скоростью ω сдвигает текстуру на f·ω·dt пикселей, неотличимо от
реального перемещения. Так как поток по сути угловой, вращательная часть
равна показанию гироскопа (не зависит от высоты):

    flow_измеренный (px) = f·(v/h)·dt   +   f·ω_gyro·dt
                           \____трансляция___/   \__вращение__/

Де-ротация: вычесть известный вклад гиро -> остаётся чистая трансляция.

    flow_derot = flow_измеренный − f·ω_gyro·dt

Проверяем: (1) без де-ротации вращение рвёт поток; (2) с гиро трансляция
восстанавливается; (3) на высоких ω измерение вылетает за search_size и
спасает пирамида (де-ротация — ПОСЛЕ измерения, значит измерить полный
поток всё равно надо).

ВАЖНО: маппинг осей гиро->поток и знак здесь заданы условно (сим их сам
и задаёт, сам и вычитает). На железе конкретное соответствие
(какая ось ICM-42688 -> какая ось потока, и знак) калибруется отдельно.
"""
import glob
from functools import partial

import numpy as np
from scipy.ndimage import shift as nd_shift

from px4flow_improved import compute_flow_improved as C

WIN, PAD = 64, 16
DT = 1.0 / 30.0                      # 30 fps
FOV_DEG = 60.0
FOCAL_PX = WIN / (2 * np.tan(np.radians(FOV_DEG) / 2))   # ~55 px

MEAS = partial(C, use_median=True, use_parabolic=True)                 # без пирамиды
MEAS_PYR = partial(C, use_median=True, use_pyramid=True)               # с пирамидой

_SCENES = [np.load(p).astype(np.float64) for p in sorted(glob.glob("real_frames/*.npy"))]


def make_pair(scene, trans, gyro):
    """img1,img2. Полный сдвиг текстуры = трансляция + f·ω·dt (вращение)."""
    rot = (FOCAL_PX * gyro[0] * DT, FOCAL_PX * gyro[1] * DT)
    total = (trans[0] + rot[0], trans[1] + rot[1])
    full = scene[:WIN + 2 * PAD, :WIN + 2 * PAD]
    sh = nd_shift(full, shift=(total[1], total[0]), order=1, mode="reflect")
    i1 = np.clip(full[PAD:PAD + WIN, PAD:PAD + WIN], 0, 255).astype(np.uint8)
    i2 = np.clip(sh[PAD:PAD + WIN, PAD:PAD + WIN], 0, 255).astype(np.uint8)
    return i1, i2, total, rot


def measure(fn, i1, i2):
    q, fx, fy = fn(i1, i2)
    return (q, fx, fy)


def run(fn, label, gyro_rates_dps, trans=(1.5, -1.0)):
    """Для каждой угловой скорости: ошибка БЕЗ и С де-ротацией."""
    print(f"\n=== {label}  (трансляция={trans} px/кадр, f={FOCAL_PX:.1f}px, dt={DT*1000:.1f}ms) ===")
    print(f"{'ω pitch,dps':>12} {'rot_px':>7} {'err БЕЗ derot':>14} {'err С derot':>12} {'valid':>7}")
    for wdps in gyro_rates_dps:
        gyro = (np.radians(wdps), np.radians(wdps * 0.5))   # pitch и половина по второй оси
        rot_px = (FOCAL_PX * gyro[0] * DT, FOCAL_PX * gyro[1] * DT)
        errs_raw, errs_der, nvalid = [], [], 0
        for scene in _SCENES:
            i1, i2, total, rot = make_pair(scene, trans, gyro)
            q, fx, fy = measure(fn, i1, i2)
            if q == 0:
                continue
            nvalid += 1
            # без де-ротации: сравниваем с трансляцией (вращение = ошибка)
            errs_raw.append(np.hypot(fx - trans[0], fy - trans[1]))
            # с де-ротацией: вычитаем известный вклад гиро
            dfx, dfy = fx - rot[0], fy - rot[1]
            errs_der.append(np.hypot(dfx - trans[0], dfy - trans[1]))
        if not errs_raw:
            print(f"{wdps:>12.0f} {np.hypot(*rot_px):>7.1f} {'—':>14} {'—':>12}   0/{len(_SCENES)}")
            continue
        print(f"{wdps:>12.0f} {np.hypot(*rot_px):>7.1f} "
              f"{np.mean(errs_raw):>14.3f} {np.mean(errs_der):>12.3f}   {nvalid}/{len(_SCENES)}")


def _selfcheck():
    """При умеренном вращении де-ротация должна резко снижать ошибку,
    и с пирамидой это работает там, где вращение вылетает за search_size."""
    gyro = (np.radians(120), np.radians(60))
    rot = (FOCAL_PX * gyro[0] * DT, FOCAL_PX * gyro[1] * DT)
    trans = (1.5, -1.0)
    i1, i2, _, _ = make_pair(_SCENES[0], trans, gyro)
    q, fx, fy = MEAS_PYR(i1, i2)
    assert q > 0
    err_raw = np.hypot(fx - trans[0], fy - trans[1])
    err_der = np.hypot(fx - rot[0] - trans[0], fy - rot[1] - trans[1])
    assert err_der < 0.5 < err_raw, f"де-ротация должна чистить поток: {err_raw:.2f}->{err_der:.2f}"
    print(f"OK derot: 120dps err {err_raw:.2f}px -> {err_der:.2f}px (с пирамидой)")


if __name__ == "__main__":
    rates = [0, 30, 60, 120, 200, 300]   # deg/s: от штатных до резких манёвров
    run(MEAS,     "БЕЗ пирамиды (search_size=4)", rates)
    run(MEAS_PYR, "С пирамидой",                  rates)
    print()
    _selfcheck()
