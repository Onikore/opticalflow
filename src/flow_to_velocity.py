"""
flow_to_velocity.py — интеграционный слой: пиксельный поток -> скорость центра дрона.

Собирает коррекции, которые НЕ входят в блок-матчинг (`compute_flow_improved`),
но нужны на выходе сенсора для EKF/DroneCAN:

  1. flow_px -> угловой поток `atan2(flow_px, focal)` -> скорость камеры (× высота)
  2. де-ротация по гироскопу: вычесть вращательную часть (`f·ω·dt` в пикселях либо
     ω·dt в радианах) — вращение корпуса подделывает трансляцию (стенд derotate_test)
  3. lever-arm: камера не в центре дрона -> `v_центр = v_камеры − ω × r`
     (стенд lever-arm, EXPERIMENTS.md)

Плюс упаковка в DroneCAN `com.hex.equipment.flow.Measurement` (интеграл в рад,
см. dronecan_test.py).

ЗНАКИ/ОСИ гиро и вектор r КАЛИБРУЮТСЯ на собранном дроне. Здесь эталонная физика;
маппинг подставляется параметрами (gyro уже в осях, согласованных с камерой:
gyro=(ωx,ωy,ωz), где ωx,ωy — наклон, ωz — рыскание).

Знак потока: PX4Flow возвращает смещение ТЕКСТУРЫ (камера +x -> текстура -x).
Скорость камеры = -flow_rate·h. НЕ инвертировать на сенсоре (EKF ждёт эту конвенцию).
"""
import numpy as np


def derotate(flow_px, gyro_xy, dt, focal_px):
    """Убрать вращательную часть потока (наклон pitch/roll подделывает трансляцию).
    flow_px=(fx,fy) — измеренный поток; gyro_xy=(ωx,ωy) рад/с — угловая скорость наклона.
    Вращение сдвигает текстуру на f·ω·dt px; вычитаем."""
    rx = focal_px * gyro_xy[0] * dt
    ry = focal_px * gyro_xy[1] * dt
    return flow_px[0] - rx, flow_px[1] - ry


def flow_to_camera_velocity(flow_px, height, dt, focal_px):
    """Пиксельный поток -> скорость КАМЕРЫ в осях камеры (м/с).
    Угловая скорость текстуры = atan2(flow_px, focal)/dt; скорость камеры =
    -rate·h (текстура едет против камеры, конвенция PX4Flow)."""
    if dt <= 0:
        return 0.0, 0.0
    rate_x = np.arctan2(flow_px[0], focal_px) / dt
    rate_y = np.arctan2(flow_px[1], focal_px) / dt
    return -rate_x * height, -rate_y * height


def lever_arm_correct(v_cam, gyro, r_cam):
    """v_центр = v_камеры − ω × r. Убирает паразитную скорость смещённой камеры
    при вращении. gyro=(ωx,ωy,ωz) рад/с, r_cam=(rx,ry,rz) м (смещение камеры от
    центра вращения дрона, в осях камеры). Возвращает горизонтальные (x,y)."""
    wx, wy, wz = gyro
    rx, ry, rz = r_cam
    # ω × r, горизонтальные компоненты
    lever_x = wy * rz - wz * ry
    lever_y = wz * rx - wx * rz
    return v_cam[0] - lever_x, v_cam[1] - lever_y


def flow_to_velocity(flow_px, height, gyro, dt, focal_px, r_cam=(0.0, 0.0, 0.0),
                     do_derotate=True, do_lever_arm=True):
    """Полный конвейер: поток -> скорость ЦЕНТРА дрона в осях камеры (м/с).
    Порядок: де-ротация (в px) -> скорость камеры (×h) -> lever-arm (−ω×r)."""
    fp = derotate(flow_px, gyro[:2], dt, focal_px) if do_derotate else flow_px
    v_cam = flow_to_camera_velocity(fp, height, dt, focal_px)
    if do_lever_arm and any(r_cam):
        return lever_arm_correct(v_cam, gyro, r_cam)
    return v_cam


def to_dronecan_fields(flow_px, gyro, dt, focal_px, quality):
    """Поля com.hex.equipment.flow.Measurement (интеграл в рад, БЕЗ инверсии знака).
    EKF сам делит на interval и вычитает rate_gyro_integral. Оси/знак гиро —
    калибровка (см. dronecan_test.py: знак rate_gyro_integral = знаку flow_integral
    от вращения)."""
    return {
        "integration_interval": float(dt),
        "flow_integral": [float(np.arctan2(flow_px[0], focal_px)),
                          float(np.arctan2(flow_px[1], focal_px))],
        "rate_gyro_integral": [float(gyro[0] * dt), float(gyro[1] * dt)],
        "quality": int(quality),
    }


if __name__ == "__main__":
    focal, dt, h = 55.0, 1.0 / 30.0, 2.0

    # 1) де-ротация: чистое вращение pitch -> поток, после де-ротации ~0 скорости
    w = np.radians(60)  # 60 dps pitch
    fpx = (focal * w * dt, 0.0)  # вращение сдвинуло текстуру
    vx, vy = flow_to_velocity(fpx, h, (w, 0, 0), dt, focal)
    assert abs(vx) < 0.05 and abs(vy) < 0.05, f"де-ротация должна гасить вращение: {vx:.3f}"
    print(f"OK derotate: pure 60dps pitch -> v=({vx:+.3f},{vy:+.3f}) м/с ~0")

    # 2) lever-arm: чистое рыскание + смещение -> паразит, коррекция гасит
    wz = 1.0
    r = (0.2, 0.0, 0.0)
    v_par = (-wz * r[1], wz * r[0])  # паразитная скорость камеры от рыскания
    vc = lever_arm_correct(v_par, (0, 0, wz), r)
    assert abs(vc[0]) < 1e-6 and abs(vc[1]) < 1e-6, f"lever-arm должен гасить рыскание: {vc}"
    print(f"OK lever-arm: yaw 1рад/с + r=0.2м -> паразит {v_par[1]:+.2f} м/с, коррекция -> {vc[1]:+.4f}")

    # 3) чистая трансляция проходит без искажения
    v = 1.5  # м/с
    fpx = (-focal * (v / h) * dt, 0.0)  # камера +v -> текстура -
    vx, vy = flow_to_velocity(fpx, h, (0, 0, 0), dt, focal)
    assert abs(vx - v) < 0.05, f"трансляция должна пройти: {vx:.3f} vs {v}"
    print(f"OK translation: +1.5 м/с -> v=({vx:+.3f},{vy:+.3f})")
