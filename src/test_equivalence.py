"""
test_equivalence.py

Проверяет, что px4flow_fast.compute_flow_fast даёт ИДЕНТИЧНЫЙ результат
эталонной построчной реализации px4flow_algo.compute_flow.

Это важно: если расхождение есть, то быстрая версия (которой мы будем
пользоваться для реал-тайм теста на ПК) не является надёжным
прототипом для будущего STM32-порта.
"""

import numpy as np
import time

from px4flow_algo import compute_flow as compute_flow_ref
from px4flow_fast import compute_flow_fast


def make_test_images(size=64, seed=0):
    rng = np.random.default_rng(seed)
    # Текстурированное изображение (шум + плавный градиент, чтобы были
    # как богатые текстурой участки, так и гладкие)
    base = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    gx, gy = np.meshgrid(np.linspace(0, 60, size), np.linspace(0, 40, size))
    gradient = (gx + gy).astype(np.uint8)

    img1 = (base * 0.5 + gradient * 0.5).astype(np.uint8)

    # Сдвигаем изображение на (dx, dy) для второго кадра, эмулируя движение
    dx, dy = 2, -1
    img2 = np.roll(img1, shift=(dy, dx), axis=(0, 1))

    return img1, img2


def main():
    img1, img2 = make_test_images(size=64)

    print("=== Эталонная (построчная) реализация ===")
    t0 = time.perf_counter()
    q_ref, fx_ref, fy_ref = compute_flow_ref(img1, img2)
    t_ref = time.perf_counter() - t0
    print(f"quality={q_ref}, flow_x={fx_ref:.4f}, flow_y={fy_ref:.4f}, "
          f"time={t_ref*1000:.2f} ms")

    print("\n=== Быстрая (векторизованная) реализация ===")
    t0 = time.perf_counter()
    q_fast, fx_fast, fy_fast = compute_flow_fast(img1, img2)
    t_fast = time.perf_counter() - t0
    print(f"quality={q_fast}, flow_x={fx_fast:.4f}, flow_y={fy_fast:.4f}, "
          f"time={t_fast*1000:.2f} ms")

    print(f"\nУскорение: {t_ref / t_fast:.1f}x")

    print("\n=== Сравнение ===")
    ok = True
    if q_ref != q_fast:
        print(f"❌ quality расходится: {q_ref} vs {q_fast}")
        ok = False
    if abs(fx_ref - fx_fast) > 1e-9:
        print(f"❌ flow_x расходится: {fx_ref} vs {fx_fast}")
        ok = False
    if abs(fy_ref - fy_fast) > 1e-9:
        print(f"❌ flow_y расходится: {fy_ref} vs {fy_fast}")
        ok = False

    if ok:
        print("✅ Результаты идентичны — быстрая версия эквивалентна эталонной")
    else:
        print("⚠ ЕСТЬ РАСХОЖДЕНИЯ — нужно искать баг в векторизации")

    # Дополнительный прогон с несколькими сдвигами/seed для большей уверенности
    print("\n=== Дополнительные прогоны (разные сдвиги/seed) ===")
    all_ok = ok
    for seed in range(5):
        for shift in [(0, 0), (1, 0), (0, 1), (-2, 3), (3, -2)]:
            rng = np.random.default_rng(seed)
            base = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
            gx, gy = np.meshgrid(np.linspace(0, 60, 64), np.linspace(0, 40, 64))
            gradient = (gx + gy).astype(np.uint8)
            a = (base * 0.5 + gradient * 0.5).astype(np.uint8)
            b = np.roll(a, shift=shift, axis=(0, 1))

            r1 = compute_flow_ref(a, b)
            r2 = compute_flow_fast(a, b)

            match = (r1[0] == r2[0] and
                     abs(r1[1] - r2[1]) < 1e-9 and
                     abs(r1[2] - r2[2]) < 1e-9)
            status = "OK" if match else "MISMATCH"
            if not match:
                all_ok = False
            print(f"seed={seed} shift={shift}: ref={r1} fast={r2} [{status}]")

    print("\n" + ("✅ ВСЕ ПРОГОНЫ СОВПАЛИ" if all_ok else "❌ ЕСТЬ РАСХОЖДЕНИЯ"))


if __name__ == "__main__":
    main()
