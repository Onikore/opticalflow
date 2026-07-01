r"""
dronecan_test.py — стенд №2: корректность выхода в DroneCAN.

То, что реально уходит в PX4/ArduPilot EKF2 — сообщение
com.hex.equipment.flow.Measurement (Data Type ID 20000). Ключевое: оно
несёт НЕ угловую скорость rad/s, а ИНТЕГРАЛ потока в радианах за интервал:

    float32   integration_interval        # dt, сек
    float32[2] rate_gyro_integral         # интеграл гиро, рад (для де-ротации)
    float32[2] flow_integral              # интеграл потока, рад
    uint8      quality                    # 0..255

flow_integral = atan2(flow_px, focal_px) — угловое смещение текстуры за
кадр. EKF делит на interval (получает rate) и вычитает rate_gyro_integral.

Проверяем сквозной round-trip в физике: известная скорость камеры v (м/с) на
высоте h -> ожидаемый пиксельный поток -> измеряем алгоритмом -> пакуем в
поля DroneCAN -> распаковываем как EKF -> должно вернуться v с верным знаком.
Ловит ошибки единиц (rad vs rad/s), знака (инверсия) и масштаба (focal).

ЗНАК (критично, см. fallback.md): PX4Flow возвращает смещение ТЕКСТУРЫ
(камера +x -> текстура -x -> flow_px = -). На сенсоре знак НЕ инвертируем;
инверсию делает потребитель. Здесь decode это и воспроизводит.

Точный порядок/имена полей DSDL сверить с dronecan/DSDL при разводке
libcanard — здесь проверяется физика конверсии, а не байтовый лейаут.
"""
import glob
from functools import partial

import numpy as np
from scipy.ndimage import shift as nd_shift

from px4flow_improved import compute_flow_improved as C

WIN, PAD = 64, 16
DT = 1.0 / 30.0
FOV_DEG = 60.0
FOCAL_PX = WIN / (2 * np.tan(np.radians(FOV_DEG) / 2))
MEAS = partial(C, use_median=True, use_pyramid=True)

_SCENES = [np.load(p).astype(np.float64) for p in sorted(glob.glob("real_frames/*.npy"))]


def pack_dronecan(flow_px_x, flow_px_y, gyro_rad_x, gyro_rad_y, quality, dt):
    """Пиксельный поток -> поля com.hex.equipment.flow.Measurement.
    БЕЗ инверсии знака (оригинальная конвенция PX4Flow)."""
    return {
        "integration_interval": dt,
        "flow_integral": [np.arctan2(flow_px_x, FOCAL_PX),
                          np.arctan2(flow_px_y, FOCAL_PX)],
        "rate_gyro_integral": [gyro_rad_x, gyro_rad_y],
        "quality": quality,
    }


def decode_ground_velocity(msg, height):
    """Как EKF: rate = (flow_integral - gyro_integral)/interval; v = -rate*h.
    Минус — потому что flow это смещение текстуры (против движения камеры)."""
    dt = msg["integration_interval"]
    fx, fy = msg["flow_integral"]
    gx, gy = msg["rate_gyro_integral"]
    rate_x = (fx - gx) / dt
    rate_y = (fy - gy) / dt
    return -rate_x * height, -rate_y * height


def make_pair(scene, tex_shift):
    full = scene[:WIN + 2 * PAD, :WIN + 2 * PAD]
    sh = nd_shift(full, shift=(tex_shift[1], tex_shift[0]), order=1, mode="reflect")
    i1 = np.clip(full[PAD:PAD + WIN, PAD:PAD + WIN], 0, 255).astype(np.uint8)
    i2 = np.clip(sh[PAD:PAD + WIN, PAD:PAD + WIN], 0, 255).astype(np.uint8)
    return i1, i2


def run():
    print(f"f={FOCAL_PX:.1f}px, dt={DT*1000:.1f}ms, окно {WIN}px\n")
    print(f"{'v_true m/s':>18} {'h м':>5} {'v_decoded m/s':>20} {'|err|':>7} {'знак':>6}")
    cases = [
        ((0.5, 0.0), 1.0), ((1.0, 0.0), 1.0), ((-0.8, 0.0), 1.0),
        ((0.0, 1.0), 1.0), ((0.0, -1.2), 0.75),
        ((0.7, -0.5), 0.5), ((-0.6, 0.9), 1.5),
    ]
    max_err = 0.0
    for (vx, vy), h in cases:
        # камера +v -> текстура смещается на -focal*(v/h)*dt пикселей
        tex = (-FOCAL_PX * (vx / h) * DT, -FOCAL_PX * (vy / h) * DT)
        dvx, dvy, n = [], [], 0
        for scene in _SCENES:
            i1, i2 = make_pair(scene, tex)
            q, fpx, fpy = MEAS(i1, i2)
            if q == 0:
                continue
            n += 1
            msg = pack_dronecan(fpx, fpy, 0.0, 0.0, q, DT)
            gx, gy = decode_ground_velocity(msg, h)
            dvx.append(gx); dvy.append(gy)
        mvx, mvy = np.mean(dvx), np.mean(dvy)
        err = np.hypot(mvx - vx, mvy - vy)
        max_err = max(max_err, err)
        sign_ok = (np.sign(mvx) == np.sign(vx) or abs(vx) < 1e-6) and \
                  (np.sign(mvy) == np.sign(vy) or abs(vy) < 1e-6)
        print(f"({vx:+.2f},{vy:+.2f})".rjust(18) + f"{h:>5.2f} "
              f"({mvx:+.3f},{mvy:+.3f})".rjust(20) + f"{err:>7.3f} "
              f"{'OK' if sign_ok else 'ПЛОХО':>6}")

    # де-ротация через поле rate_gyro_integral: чистое вращение -> v≈0.
    # gyro_integral должен иметь ТОТ ЖЕ знак, что flow_integral от вращения
    # (это и есть калибровка оси/знака гиро на железе — легко ошибиться).
    print("\nПроверка де-ротации через rate_gyro_integral (чистое вращение, v=0):")
    gyro = np.radians(90)                      # 90 dps pitch
    tex = (-FOCAL_PX * gyro * DT, 0.0)         # вращение двигает текстуру
    i1, i2 = make_pair(_SCENES[0], tex)
    q, fpx, fpy = MEAS(i1, i2)
    gyro_int = [np.arctan2(tex[0], FOCAL_PX), np.arctan2(tex[1], FOCAL_PX)]
    no_gyro = decode_ground_velocity(pack_dronecan(fpx, fpy, 0, 0, q, DT), 1.0)
    with_gyro = decode_ground_velocity(
        pack_dronecan(fpx, fpy, gyro_int[0], gyro_int[1], q, DT), 1.0)
    print(f"  без гиро в сообщении: v_decoded=({no_gyro[0]:+.3f},{no_gyro[1]:+.3f}) — вращение принято за скорость")
    print(f"  с rate_gyro_integral: v_decoded=({with_gyro[0]:+.3f},{with_gyro[1]:+.3f}) м/с (должно ~0)")
    assert max_err < 0.15, f"round-trip должен восстанавливать скорость: max_err={max_err:.3f}"
    assert abs(with_gyro[0]) < 0.15, "де-ротация в сообщении должна гасить вращение"
    print(f"\nOK: max ошибка скорости {max_err:.3f} м/с (растёт с высотой — физика), "
          f"знаки верны, де-ротация гасит вращение")


if __name__ == "__main__":
    run()
