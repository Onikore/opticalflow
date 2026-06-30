"""
synthetic_test.py

Тест алгоритма на полностью синтетических данных с ИЗВЕСТНОЙ "истинной"
скоростью движения камеры. Полезно для:

  1. Проверки что алгоритм вообще работает корректно (без камеры под рукой)
  2. Измерения погрешности (известный ground truth vs то что насчитал
     алгоритм)
  3. Замера производительности — сколько кадров/сек реально получится
     на чистом Python (с быстрой векторизованной версией) — это даёт
     ориентир, насколько алгоритм тяжёлый, прежде чем переносить на
     STM32H743 (480 МГц, без NumPy, всё на чистых циклах с DSP-интринсиками)

Генерируем большую текстурированную "сцену" (шум + узоры), затем
"летаем" над ней виртуальной камерой 64x64 с заданной скоростью пиксель/кадр,
вырезая окно из сцены на каждом кадре.
"""

import time
import numpy as np

from px4flow_fast import compute_flow_fast


def make_scene(size=512, seed=42):
    """Текстурированная 'сцена', по которой будем виртуально летать."""
    rng = np.random.default_rng(seed)
    scene = rng.integers(40, 215, size=(size, size), dtype=np.uint8)

    # Добавляем немного структуры (полосы, пятна), чтобы не был чистый шум —
    # ближе к реальной текстуре земли/травы/асфальта
    for _ in range(40):
        cx, cy = rng.integers(0, size, size=2)
        r = rng.integers(5, 25)
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        scene[mask] = rng.integers(40, 215)

    return scene


def extract_window(scene: np.ndarray, cx: float, cy: float, win: int = 64) -> np.ndarray:
    """Вырезает окно win x win из сцены с центром в (cx, cy)."""
    h, w = scene.shape
    x0 = int(round(cx - win / 2))
    y0 = int(round(cy - win / 2))

    x0 = max(0, min(w - win, x0))
    y0 = max(0, min(h - win, y0))

    return scene[y0:y0 + win, x0:x0 + win].copy()


def run_synthetic_test(true_vx=2.0, true_vy=1.0, n_frames=60, win=64):
    """
    true_vx, true_vy — истинная скорость камеры в пикселях/кадр.
    Алгоритм должен вернуть примерно такие же значения flow_x/flow_y.
    """
    scene = make_scene(size=512)

    cx, cy = 256.0, 256.0  # стартовая позиция окна камеры в сцене

    img_old = extract_window(scene, cx, cy, win)

    errors_x = []
    errors_y = []
    qualities = []
    timings = []

    print(f"=== Синтетический тест: истинная скорость vx={true_vx}, vy={true_vy} px/кадр ===\n")

    for frame in range(1, n_frames + 1):
        cx += true_vx
        cy += true_vy

        img_new = extract_window(scene, cx, cy, win)

        t0 = time.perf_counter()
        quality, flow_x, flow_y = compute_flow_fast(img_old, img_new)
        dt = time.perf_counter() - t0

        timings.append(dt)
        qualities.append(quality)

        if quality > 0:
            err_x = flow_x - true_vx
            err_y = flow_y - true_vy
            errors_x.append(err_x)
            errors_y.append(err_y)

            print(f"frame {frame:3d}: quality={quality:3d}  "
                  f"flow=({flow_x:+.2f},{flow_y:+.2f})  "
                  f"true=({true_vx:+.2f},{true_vy:+.2f})  "
                  f"err=({err_x:+.3f},{err_y:+.3f})  "
                  f"time={dt*1000:.2f}ms")
        else:
            print(f"frame {frame:3d}: quality=0 (низкая текстура/превышен threshold)")

        img_old = img_new

    print("\n=== Итоги ===")
    if errors_x:
        print(f"Средняя ошибка flow_x: {np.mean(errors_x):+.4f} px "
              f"(std={np.std(errors_x):.4f})")
        print(f"Средняя ошибка flow_y: {np.mean(errors_y):+.4f} px "
              f"(std={np.std(errors_y):.4f})")
        print(f"Средний quality: {np.mean(qualities):.1f}/255")
    else:
        print("⚠ Ни один кадр не дал валидный quality > 0")

    print(f"\nВремя обработки на кадр: "
          f"avg={np.mean(timings)*1000:.2f}ms  "
          f"min={np.min(timings)*1000:.2f}ms  "
          f"max={np.max(timings)*1000:.2f}ms")
    print(f"Эквивалентный FPS (чистый Python, для справки): "
          f"{1.0/np.mean(timings):.1f} fps")
    print("\nПримечание: на STM32H743 @ 480MHz с C-кодом и CMSIS DSP "
          "интринсиками (__SADD8/__USAD8) ожидается ускорение в 50-150x "
          "относительно Python — это будет 1-3 мс на кадр, т.е. 300-1000+ fps "
          "теоретически (реально лимитируется частотой кадров с камеры).")


def run_subpixel_accuracy_test():
    """
    Дополнительный тест: проверяем что алгоритм адекватно реагирует
    на разные скорости движения (включая дробные значения, которые
    эмулируются плавным движением окна).
    """
    print("\n\n=== Тест точности на разных скоростях ===\n")
    test_speeds = [
        (0.5, 0.0), (1.0, 0.0), (2.0, 0.0), (3.5, 0.0),
        (0.0, 1.0), (0.0, 2.5),
        (1.5, 1.5), (-2.0, 1.0), (3.0, -2.0),
    ]

    for vx, vy in test_speeds:
        scene = make_scene(size=512, seed=7)
        cx, cy = 256.0, 256.0
        img_old = extract_window(scene, cx, cy)

        flows_x, flows_y, quals = [], [], []
        for _ in range(10):
            cx += vx
            cy += vy
            img_new = extract_window(scene, cx, cy)
            q, fx, fy = compute_flow_fast(img_old, img_new)
            if q > 0:
                flows_x.append(fx)
                flows_y.append(fy)
                quals.append(q)
            img_old = img_new

        if flows_x:
            mean_fx = np.mean(flows_x)
            mean_fy = np.mean(flows_y)
            print(f"true=({vx:+.1f},{vy:+.1f})  "
                  f"measured=({mean_fx:+.2f},{mean_fy:+.2f})  "
                  f"quality_avg={np.mean(quals):.0f}  "
                  f"n_valid={len(flows_x)}/10")
        else:
            print(f"true=({vx:+.1f},{vy:+.1f})  ⚠ нет валидных измерений "
                  "(слишком быстрое движение для search_size=4?)")


if __name__ == "__main__":
    run_synthetic_test(true_vx=2.0, true_vy=1.0, n_frames=30)
    run_subpixel_accuracy_test()
