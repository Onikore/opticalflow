r"""
envelope_test.py — стенд №5: карта рабочего диапазона (почему на низкой
высоте нельзя резко манёврить, и как пирамида это лечит).

Сенсор меряет не больше ceiling_px пикселей за кадр (окно поиска). В рад/с:
    ceiling_rate = ceiling_px / (focal · dt)
Полный поток = v/h + ω_наклона < ceiling_rate. На низкой высоте v/h велика
даже при малой v -> на наклон (ω) остаётся мало -> дрон почти не может
крениться. На большой высоте v/h мала -> большой запас.

Замеряем реальный потолок (px/кадр, где ошибка ещё <0.5px) для search=4
(как сток H-Flow) и для пирамиды, строим карту.
"""
import glob
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import shift as nd_shift

from px4flow_improved import compute_flow_improved as C

WIN, PAD = 64, 16
DT = 1.0 / 30.0
FOV_DEG = 60.0
FOCAL_PX = WIN / (2 * np.tan(np.radians(FOV_DEG) / 2))
_SC = [np.load(p).astype(np.float64) for p in sorted(glob.glob("real_frames/*.npy"))]


def _pair(scene, mag, ang):
    tx, ty = mag * np.cos(ang), mag * np.sin(ang)
    full = scene[:WIN + 2 * PAD, :WIN + 2 * PAD]
    sh = nd_shift(full, shift=(ty, tx), order=1, mode="reflect")
    return (np.clip(full[PAD:PAD + WIN, PAD:PAD + WIN], 0, 255).astype("uint8"),
            np.clip(sh[PAD:PAD + WIN, PAD:PAD + WIN], 0, 255).astype("uint8"), tx, ty)


def measure_ceiling(fn):
    """Макс сдвиг px/кадр, при котором ошибка ещё < 0.5 px."""
    prev = 0.0
    for mag in np.arange(1, 12, 0.5):
        errs = []
        for ang in (0, 0.6, 1.2, -0.9):
            for sc in _SC:
                i1, i2, tx, ty = _pair(sc, mag, ang)
                q, fx, fy = fn(i1, i2)
                if q > 0:
                    errs.append(np.hypot(fx - tx, fy - ty))
        if not errs or np.mean(errs) > 0.5:
            return prev
        prev = mag
    return prev


def main(out="envelope.png"):
    configs = {
        "search=4 (сток H-Flow)": (partial(C, use_median=True, use_parabolic=True), "tab:red"),
        "пирамида (наша)":        (partial(C, use_median=True, use_pyramid=True),   "tab:green"),
    }
    ceil_px = {name: measure_ceiling(fn) for name, (fn, _) in configs.items()}
    ceil_rate = {n: c / (FOCAL_PX * DT) for n, c in ceil_px.items()}   # рад/с
    for n in configs:
        print(f"{n:<24} потолок {ceil_px[n]:.1f} px/кадр = {ceil_rate[n]:.2f} рад/с "
              f"= {np.degrees(ceil_rate[n]):.0f} °/с")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Рабочий диапазон optical flow (f={FOCAL_PX:.0f}px, {int(1/DT)}fps, окно {WIN}px)",
                 fontsize=12)

    # A: макс горизонтальная скорость vs высота (при малом наклоне)
    h = np.linspace(0.2, 3.0, 100)
    for name, (_, col) in configs.items():
        ax1.plot(h, ceil_rate[name] * h, col, lw=2.2, label=name)
    ax1.set_xlabel("высота h, м"); ax1.set_ylabel("макс скорость v, м/с")
    ax1.set_title("Потолок скорости растёт с высотой (v = rate·h)")
    ax1.grid(alpha=0.3); ax1.legend()
    ax1.annotate("на низкой высоте\nскорость жёстко ограничена",
                 xy=(0.4, ceil_rate["search=4 (сток H-Flow)"] * 0.4),
                 xytext=(0.9, 1.0), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="gray"))

    # B: бюджет наклона — сколько ω остаётся после набора скорости, по высотам
    for hh, ls in [(0.5, "-"), (1.0, "--"), (2.0, ":")]:
        for name, (_, col) in configs.items():
            v = np.linspace(0, ceil_rate[name] * hh, 50)
            w_deg = np.degrees(ceil_rate[name] - v / hh)
            ax2.plot(v, w_deg, col, ls=ls, lw=1.8,
                     label=f"{name.split()[0]}, h={hh}м")
    ax2.set_xlabel("горизонтальная скорость v, м/с")
    ax2.set_ylabel("остаточный бюджет наклона ω, °/с")
    ax2.set_title("Чем ниже и быстрее — тем меньше можно крениться")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=7, ncol=2)
    ax2.set_ylim(0, None)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=110)
    print(f"готово -> {out}")

    assert ceil_px["пирамида (наша)"] > ceil_px["search=4 (сток H-Flow)"], \
        "пирамида должна расширять потолок"
    print("OK: пирамида шире по диапазону скорости/манёвра на всех высотах")


if __name__ == "__main__":
    main()
