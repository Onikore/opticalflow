"""
flow_camera_test.py

Главный тестовый скрипт. Захватывает кадры с веб-камеры (или видеофайла),
гоняет через алгоритм px4flow_fast.compute_flow_fast и показывает:

  - текущий кадр в grayscale
  - стрелку направления потока (flow vector)
  - quality, flow_x/flow_y в пикселях
  - угловую скорость в рад/с (как уходило бы в DroneCAN сообщение
    com.hex.equipment.flow.Measurement)
  - реальный FPS обработки

Это прямой эквивалент того, что будет считать STM32F7 + ATK-OV7725 MD,
только тут вместо DCMI+FIFO используется обычная веб-камера через OpenCV,
а вместо отправки в DroneCAN — печать в консоль/визуализация.

Управление:
  q / ESC  — выход
  r        — сброс (img_old = None, начать заново)
  s        — сохранить текущую пару кадров в test_frames/ (для regress-теста)

Запуск:
  python3 flow_camera_test.py                  # веб-камера 0
  python3 flow_camera_test.py --source 1        # веб-камера с другим индексом
  python3 flow_camera_test.py --source video.mp4  # тест на видеофайле
"""

import argparse
import math
import time
import os

import cv2

from _paths import DATA
import numpy as np

from px4flow_fast import compute_flow_fast


# ============ Параметры эмулируемой камеры/линзы ============
# Эти параметры нужно будет откалибровать под реальную ATK-OV7725 MD
# с конкретным объективом. Сейчас — приблизительные значения для
# объектива ~90° FOV по диагонали на разрешении FLOW_IMG_SIZE.
FLOW_IMG_SIZE = 64          # PX4Flow по умолчанию работает с блоком 64x64
FOCAL_LENGTH_PX = 64.0       # фокусное расстояние в пикселях (подбирается)

SEARCH_SIZE = 4
FLOW_FEATURE_THRESHOLD = 30
FLOW_VALUE_THRESHOLD = 3000


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Конвертирует кадр с камеры в grayscale 64x64 — имитация того, что
    реально придёт с ATK-OV7725 MD после конфигурации QVGA + кропа
    центральной области, либо после programmatic downscale на STM32.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    # Берём центральный квадрат, чтобь не было искажений от неравномерного
    # сжатия по осям (имитация кропа в прошивке STM32)
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = gray[y0:y0 + side, x0:x0 + side]

    resized = cv2.resize(cropped, (FLOW_IMG_SIZE, FLOW_IMG_SIZE),
                          interpolation=cv2.INTER_AREA)
    return resized


def flow_to_dronecan_fields(flow_x_px: float, flow_y_px: float, dt_s: float):
    """
    Конвертация пиксельного потока в угловую скорость (рад/с) —
    именно то, что упаковывается в DroneCAN сообщение
    com.hex.equipment.flow.Measurement (integrated_x / integrated_y).

    Соответствует OpticalFlowPX4::calcFlow:
        flow_x = atan2(flow_x, focal_length_x)
    """
    if dt_s <= 0:
        return 0.0, 0.0

    angular_x = math.atan2(flow_x_px, FOCAL_LENGTH_PX) / dt_s
    angular_y = math.atan2(flow_y_px, FOCAL_LENGTH_PX) / dt_s
    return angular_x, angular_y


def draw_overlay(display_frame: np.ndarray, flow_x_px: float, flow_y_px: float,
                  quality: int, ang_x: float, ang_y: float, fps: float):
    """Рисует UI поверх увеличенного кадра для наглядности."""
    h, w = display_frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Масштабируем стрелку потока для наглядности (пиксели в кадре
    # FLOW_IMG_SIZE x FLOW_IMG_SIZE слишком малы для display_frame)
    scale = w / FLOW_IMG_SIZE
    arrow_scale = 8.0  # дополнительное визуальное усиление

    end_x = int(cx + flow_x_px * scale * arrow_scale)
    end_y = int(cy + flow_y_px * scale * arrow_scale)

    color = (0, 255, 0) if quality > 50 else (0, 165, 255) if quality > 0 else (0, 0, 255)

    cv2.arrowedLine(display_frame, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)
    cv2.circle(display_frame, (cx, cy), 4, (255, 255, 255), -1)

    lines = [
        f"quality: {quality}",
        f"flow px: x={flow_x_px:+.2f} y={flow_y_px:+.2f}",
        f"angular rad/s: x={ang_x:+.3f} y={ang_y:+.3f}",
        f"fps: {fps:.1f}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(display_frame, line, (10, 25 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display_frame, line, (10, 25 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return display_frame


def main():
    parser = argparse.ArgumentParser(description="Тест optical flow алгоритма PX4Flow на ПК")
    parser.add_argument("--source", default="0",
                         help="Индекс камеры (0,1,...) или путь к видеофайлу")
    parser.add_argument("--display-size", type=int, default=480,
                         help="Размер окна вывода в пикселях")
    args = parser.parse_args()

    # Источник: число -> индекс камеры, иначе путь к файлу
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть источник видео: {source}")
        return

    print("Управление: q/ESC — выход, r — сброс, s — сохранить пару кадров")

    tf = DATA / "test_frames"; os.makedirs(tf, exist_ok=True)

    img_old = None
    t_prev = time.perf_counter()
    fps_smooth = 0.0
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("Конец видео или ошибка чтения кадра")
            break

        gray64 = preprocess_frame(frame_bgr)

        t_now = time.perf_counter()
        dt_s = t_now - t_prev
        t_prev = t_now

        quality, flow_x_px, flow_y_px = 0, 0.0, 0.0

        if img_old is not None:
            quality, flow_x_px, flow_y_px = compute_flow_fast(
                img_old, gray64,
                search_size=SEARCH_SIZE,
                flow_feature_threshold=FLOW_FEATURE_THRESHOLD,
                flow_value_threshold=FLOW_VALUE_THRESHOLD,
                invert_sign=True,  # для интуитивной стрелки на экране;
                                    # в реальной DroneCAN публикации на STM32
                                    # invert_sign НЕ используется (см. px4flow_fast.py)
            )

        ang_x, ang_y = flow_to_dronecan_fields(flow_x_px, flow_y_px, dt_s)

        # FPS обработки (экспоненциальное сглаживание)
        inst_fps = 1.0 / dt_s if dt_s > 0 else 0.0
        fps_smooth = inst_fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * inst_fps)

        # Печатаем в консоль — это симулирует то, что STM32 публикует
        # в DroneCAN com.hex.equipment.flow.Measurement каждый кадр
        if frame_idx % 5 == 0:  # не спамим консоль на каждый кадр
            print(f"[frame {frame_idx:5d}] quality={quality:3d}  "
                  f"flow_px=({flow_x_px:+.2f},{flow_y_px:+.2f})  "
                  f"angular_rad_s=({ang_x:+.4f},{ang_y:+.4f})  "
                  f"dt={dt_s*1000:.1f}ms  fps={fps_smooth:.1f}")

        # Визуализация
        display = cv2.resize(gray64, (args.display_size, args.display_size),
                              interpolation=cv2.INTER_NEAREST)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        display = draw_overlay(display, flow_x_px, flow_y_px, quality,
                                ang_x, ang_y, fps_smooth)

        cv2.imshow("PX4Flow algorithm test (q=quit, r=reset, s=save)", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q или ESC
            break
        elif key == ord('r'):
            img_old = None
            print(">>> Сброс эталонного кадра")
        elif key == ord('s') and img_old is not None:
            ts = int(time.time() * 1000)
            np.save(str(tf / f"frame_{ts}_prev.npy"), img_old)
            np.save(str(tf / f"frame_{ts}_curr.npy"), gray64)
            print(f">>> Сохранена пара кадров test_frames/frame_{ts}_*.npy")

        img_old = gray64
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
