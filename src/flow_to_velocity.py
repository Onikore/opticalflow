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
from dataclasses import dataclass, field


@dataclass
class SensorConfig:
    """Все калибровки сенсора в одном месте. Что мерить на собранном дроне:

    focal_px      — фокус в пикселях НА РАБОЧЕМ разрешении потока
                    (из calib: fx * W_flow / W_native; наш баг: 551.8*96/480≈110).
    gyro_map      — какая ось гиро соответствует осям потока (x,y,z-индексы в
                    массиве IMU) — определяется монтажом камеры vs IMU.
    gyro_sign     — знаки к осям после маппинга. Калибровка на столе: чистое
                    вращение -> decoded v должно быть ~0 (см. dronecan_test.py);
                    неверный знак УДВАИВАЕТ вращение вместо гашения.
    r_cam         — смещение камеры от центра вращения дрона, м, в осях камеры
                    (lever-arm: v_центр = v_кам − ω×r). Линейкой по месту.
    h_min/h_max   — валидный диапазон дальномера, м (вне -> quality=0).
    """
    focal_px: float = 110.0
    gyro_map: tuple = (0, 1, 2)
    gyro_sign: tuple = (1.0, 1.0, 1.0)
    r_cam: tuple = (0.0, 0.0, 0.0)
    h_min: float = 0.3
    h_max: float = 12.0

    def map_gyro(self, gyro_raw):
        """Сырой вектор IMU -> (ωx,ωy,ωz) в осях потока."""
        return tuple(self.gyro_sign[i] * gyro_raw[self.gyro_map[i]] for i in range(3))

    def height_valid(self, h):
        return self.h_min <= h <= self.h_max


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


def sensor_output(flow_px, quality, height, gyro_raw, dt, cfg: SensorConfig):
    """Полный выход сенсора по конфигу: маппинг гиро, гейт высоты, де-ротация,
    lever-arm. Возвращает (vx, vy, quality) — quality=0 если высота невалидна."""
    if not cfg.height_valid(height) or dt <= 0:
        return 0.0, 0.0, 0
    gyro = cfg.map_gyro(gyro_raw)
    vx, vy = flow_to_velocity(flow_px, height, gyro, dt, cfg.focal_px, cfg.r_cam)
    return vx, vy, int(quality)


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

    # 4) SensorConfig: гейт высоты + маппинг гиро со знаком
    cfg = SensorConfig(focal_px=focal, gyro_map=(1, 0, 2), gyro_sign=(-1, 1, 1))
    _, _, q = sensor_output(fpx, 200, 0.1, (0, 0, 0), dt, cfg)   # h ниже h_min
    assert q == 0, "невалидная высота должна давать quality=0"
    g = cfg.map_gyro((0.5, 0.2, 0.1))
    assert g == (-0.2, 0.5, 0.1), f"маппинг гиро неверен: {g}"
    print(f"OK config: h-гейт (q=0 при h<h_min), gyro map (0.5,0.2,0.1)->{g}")
