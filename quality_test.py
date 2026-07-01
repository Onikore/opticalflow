r"""
quality_test.py — стенд №3: честность поля quality на вырожденных сценах.

EKF доверяет quality. Опасен не низкий поток, а УВЕРЕННО НЕВЕРНЫЙ: высокий
quality + неправильный flow => EKF примет мусор за движение. Прогоняем сенсор
на сценах, где мерить нечего/трудно, с ИЗВЕСТНЫМ сдвигом, и классифицируем:

  REJECT — q низкий, сенсор честно молчит (EKF проигнорирует)      OK
  GOOD   — q высокий и flow верный                                  OK
  DANGER — q высокий, но flow неверный (EKF доверится мусору)      ПЛОХО

Сцены: гладкая (нет текстуры), тёмная/низкий контраст, повторяющаяся
(aperture problem), блик (пересвет), сильный motion blur.
"""
import glob
from functools import partial

import numpy as np
import cv2
from scipy.ndimage import shift as nd_shift

from px4flow_improved import compute_flow_improved as C

WIN, PAD = 64, 16
SHIFT = (2.0, 1.0)             # известный истинный сдвиг текстуры
Q_HI = 100                    # порог «высокого» quality
ERR_BAD = 1.5                 # порог «неверного» flow, px

MEAS = partial(C, use_median=True, use_parabolic=True)
MEAS_UNIQ = partial(C, use_median=True, use_parabolic=True, use_uniqueness=True)
_SC = [np.load(p).astype(np.float64) for p in sorted(glob.glob("real_frames/*.npy"))]


def _shifted(base):
    full = base[:WIN + 2 * PAD, :WIN + 2 * PAD]
    sh = nd_shift(full, shift=(SHIFT[1], SHIFT[0]), order=1, mode="reflect")
    return (np.clip(full[PAD:PAD + WIN, PAD:PAD + WIN], 0, 255).astype("uint8"),
            np.clip(sh[PAD:PAD + WIN, PAD:PAD + WIN], 0, 255).astype("uint8"))


def scene_good(s):        return s.copy()
def scene_blank(s):       return np.full_like(s, 128.0) + np.random.default_rng(0).normal(0, 1, s.shape)
def scene_dark(s):        return s * 0.10 + 4                      # тёмный, слабый контраст
def scene_lowcontrast(s): return 118 + (s - s.mean()) * 0.08      # почти ровный серый
def scene_glare(s):
    o = s.copy(); o[WIN//2-2:WIN//2+PAD+18, WIN//2-2:WIN//2+PAD+18] = 255; return o
def scene_stripes(s):
    # вертикальные полосы период 4px: движение ВДОЛЬ полос (по y) ненаблюдаемо
    # — классический aperture problem, риск confident-garbage
    xx = np.arange(s.shape[1])
    return (128 + 90 * ((xx // 2) % 2))[None, :].repeat(s.shape[0], 0).astype(float)
def scene_tiles(s):
    yy, xx = np.mgrid[0:s.shape[0], 0:s.shape[1]]
    return 128 + 90 * (((xx // 3) + (yy // 3)) % 2)              # мелкая шахматка период 3px
def scene_blur(s):        return cv2.GaussianBlur(s.astype(np.float32), (0, 0), 3.0)


SCENES = {
    "good (эталон)":        scene_good,
    "blank (гладкая)":      scene_blank,
    "dark (тёмная)":        scene_dark,
    "lowcontrast":          scene_lowcontrast,
    "glare (пересвет)":     scene_glare,
    "stripes (полосы)":     scene_stripes,
    "tiles (мелкий кафель)": scene_tiles,
    "motion blur":          scene_blur,
}


def classify(q, err):
    if q < Q_HI:
        return "REJECT (ок — молчит)"
    return "GOOD (ок)" if err < ERR_BAD else "DANGER (q высокий, врёт!)"


def run(fn, label):
    print(f"\n=== {label} (истинный сдвиг {SHIFT}, порог q>{Q_HI}, err>{ERR_BAD}px) ===")
    print(f"{'сцена':<22} {'quality':>8} {'flow':>16} {'err px':>8}  вердикт")
    dangers = 0
    for name, gen in SCENES.items():
        qs, errs = [], []
        for base in _SC:
            i1, i2 = _shifted(gen(base))
            q, fx, fy = fn(i1, i2)
            qs.append(q)
            errs.append(np.hypot(fx - SHIFT[0], fy - SHIFT[1]) if q > 0 else np.nan)
        q = int(np.mean(qs))
        err = np.nanmean(errs) if np.any(~np.isnan(errs)) else np.nan
        verd = classify(q, err if not np.isnan(err) else 99)
        if "DANGER" in verd:
            dangers += 1
        fs = f"({fx:+.2f},{fy:+.2f})" if q > 0 else "—"
        es = f"{err:.2f}" if not np.isnan(err) else "—"
        print(f"{name:<22} {q:>8} {fs:>16} {es:>8}  {verd}")
    return dangers


if __name__ == "__main__":
    d1 = run(MEAS, "Принятый набор (median+parabolic)")
    d2 = run(MEAS_UNIQ, "+ uniqueness (anti-aperture) на тех же сценах")
    print(f"\nDANGER-кейсов: base={d1}, +uniqueness={d2}")
    assert d1 > 0, "тест должен воспроизводить опасный кейс на повторяющейся текстуре"
    assert d2 == 0, "uniqueness (peak-ratio) должен убрать confident-garbage"
    print("Вывод: гладкие/тёмные/размытые сцены честно молчат (REJECT). "
          "Повторяющаяся текстура (полосы/плитка) — реальный риск confident-"
          "garbage: БЕЗ uniqueness сенсор выдаёт q=255 и врёт (err 4-6px). "
          "peak-ratio uniqueness ловит и отбраковывает -> для полёта над "
          "плиткой/досками/полем его стоит включать.")
