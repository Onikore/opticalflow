"""
video_test.py — проверка на РЕАЛЬНЫХ кадрах (не синтетика).

Два теста:
  (A) Панорамирование окна 64x64 по реальному кадру с ИЗВЕСТНОЙ скоростью
      (real_frames/*.npy, извлечены из видео). Реальная текстура + точный
      ground truth -> честный RMSE оригинал vs улучшенный на реальных пикселях.
  (B) Последовательные реальные кадры видео vs эталон OpenCV Farneback —
      сквозная проверка, что поток совпадает по направлению с независимым
      алгоритмом на настоящей съёмке.
"""

import glob

from _paths import REAL_FRAMES, VIDEOS
from functools import partial

import numpy as np
import cv2

import benchmark as B
from px4flow_fast import compute_flow_fast
from px4flow_improved import compute_flow_improved

IMPROVED = partial(compute_flow_improved, use_median=True, use_parabolic=True)

# --- (A) панорамирование по реальной текстуре, ground truth известен ---
_SCENES = [np.load(p).astype(np.float64) for p in sorted(REAL_FRAMES.glob("*.npy"))]


def _real_scene(size, seed):
    """Подмена benchmark.make_scene: реальный кадр вместо синтетики."""
    s = _SCENES[seed % len(_SCENES)]
    # небольшой сдвиг кропа по seed для разнообразия
    off = (seed * 7) % (s.shape[0] - size) if s.shape[0] > size else 0
    crop = s[off:off + size, off:off + size]
    if crop.shape != (size, size):  # если реальная сцена меньше — ресайз
        crop = cv2.resize(s, (size, size)).astype(np.float64)
    return crop


def test_pan():
    B.make_scene = _real_scene  # переопределяем источник текстуры
    print("######## (A) РЕАЛЬНАЯ текстура, панорамирование с известной скоростью ########")
    base = B.bench(compute_flow_fast, "ОРИГИНАЛ на реальных кадрах")
    impr = B.bench(IMPROVED, "УЛУЧШЕННЫЙ (медиана+парабола)")
    B.compare(base, impr, "оригинал", "улучш")


# --- (B) последовательные реальные кадры vs Farneback ---
def _prep(frame, scale=96, crop=64):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (scale, scale))
    o = (scale - crop) // 2
    return g[o:o + crop, o:o + crop]


def test_video(path, n=120):
    print(f"\n######## (B) Последовательные кадры vs Farneback ########")
    print(f"видео: {path.split('/')[-1]}")
    v = cv2.VideoCapture(path)
    prev = None
    ours, ref, quals = [], [], []
    while len(ours) < n:
        ok, fr = v.read()
        if not ok:
            break
        g = _prep(fr)
        if prev is not None:
            q, fx, fy = IMPROVED(prev, g)
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            rx, ry = float(fl[..., 0].mean()), float(fl[..., 1].mean())
            if q > 0:
                ours.append((fx, fy)); ref.append((rx, ry)); quals.append(q)
        prev = g
    if len(ours) < 5:
        print(f"  мало валидных кадров ({len(ours)}) — движение вне диапазона/слабая текстура")
        return
    o = np.array(ours); r = np.array(ref)
    # корреляция нашего потока с эталоном по обеим осям
    cx = np.corrcoef(o[:, 0], r[:, 0])[0, 1]
    cy = np.corrcoef(o[:, 1], r[:, 1])[0, 1]
    # совпадение знака направления (когда движение заметное)
    big = np.hypot(r[:, 0], r[:, 1]) > 0.15
    if big.sum():
        sign_ok = np.mean((np.sign(o[big, 0]) == np.sign(r[big, 0])) &
                          (np.sign(o[big, 1]) == np.sign(r[big, 1])))
    else:
        sign_ok = float("nan")
    print(f"  валидных кадров: {len(ours)}/{n}, средний quality {np.mean(quals):.0f}/255")
    print(f"  корреляция с Farneback:  X={cx:+.2f}  Y={cy:+.2f}  (1.0 = идеально)")
    print(f"  совпадение знака направления на заметном движении: {sign_ok*100:.0f}%")


def test_pan_vs_farneback():
    """Реальная текстура + реальное движение: сверяем НАШ поток и с истиной,
    и с эталоном Farneback одновременно (то, чего не дали статичные клипы)."""
    from scipy.ndimage import shift as nd_shift
    print("\n######## (B') Реальная текстура + реальное движение: мы vs истина vs Farneback ########")
    rng = np.random.default_rng(0)
    truth, ours, ref = [], [], []
    for scene in _SCENES:
        for _ in range(20):
            tdx, tdy = rng.uniform(-3.5, 3.5, 2)
            full = scene[:104, :104]
            sh = nd_shift(full, shift=(tdy, tdx), order=1, mode="reflect")
            i1 = full[12:76, 12:76].astype(np.uint8)
            i2 = sh[12:76, 12:76].astype(np.uint8)
            q, fx, fy = IMPROVED(i1, i2)
            if q == 0:
                continue
            fl = cv2.calcOpticalFlowFarneback(i1, i2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            truth.append((tdx, tdy)); ours.append((fx, fy))
            ref.append((float(fl[..., 0].mean()), float(fl[..., 1].mean())))
    t, o, r = np.array(truth), np.array(ours), np.array(ref)
    rmse = lambda a: float(np.sqrt(np.mean((a[:, 0] - t[:, 0])**2 + (a[:, 1] - t[:, 1])**2)))
    print(f"  кадров: {len(t)}")
    print(f"  RMSE к истине:   наш={rmse(o):.3f}px   Farneback={rmse(r):.3f}px")
    print(f"  корреляция наш↔Farneback:  X={np.corrcoef(o[:,0],r[:,0])[0,1]:+.3f}  "
          f"Y={np.corrcoef(o[:,1],r[:,1])[0,1]:+.3f}")


if __name__ == "__main__":
    test_pan()
    test_pan_vs_farneback()
    test_video(str(VIDEOS / "IMG_1252.MOV"))
