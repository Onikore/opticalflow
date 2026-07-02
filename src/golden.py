"""
golden.py — golden-набор для приёмки C-порта.

Реальные пары кадров из полёта (кэш бага) по режимам: круиз / зависание /
максимальное вращение / случайные. Для каждой пары — эталонный выход
Python-референса (полётный набор флагов). C-порт обязан воспроизвести.

Использование:
    python3 src/golden.py generate   # пересоздать data/test_frames/golden.npz
    python3 src/golden.py check      # Python сам себя (точное совпадение)
    python3 src/golden.py check-c out.csv   # сверить вывод C (допуск 0.05 px)

Формат для C: кадры также выгружаются в data/test_frames/golden_frames.bin
(N пар подряд: i1[W*W] uint8, i2[W*W] uint8) + golden_expected.csv (q,fx,fy).
"""
import sys
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from px4flow_improved import compute_flow_improved
from _paths import DATA

GOLDEN = DATA / "test_frames" / "golden.npz"
FRAMES_BIN = DATA / "test_frames" / "golden_frames.bin"
EXPECTED_CSV = DATA / "test_frames" / "golden_expected.csv"

# полётный набор флагов — то, что реализует C-порт
FLAGS = dict(use_median=True, use_pyramid=True, use_tri_subpix=True,
             use_mad=True, use_boundary_reject=True)
TOL_PX = 0.05      # допуск для C (float Python vs int/Q8 C; fixed-point даёт <0.002)


def reference(i1, i2):
    return compute_flow_improved(i1, i2, **FLAGS)


def generate(n_per_regime=5):
    d = np.load(DATA / "bag_flight" / "bag_cache.npz")
    imgs, it = d["imgs"], d["img_t"]
    h = np.interp(it, d["rf_t"], d["rf_h"])
    gyro = d["gyro"]
    tilt = np.hypot(np.interp(it, d["imu_t"], gyro[:, 0]),
                    np.interp(it, d["imu_t"], gyro[:, 1]))
    lp_t, lp = d["lp_t"], d["lp_xy"]
    vx = np.gradient(np.interp(it, lp_t, lp[:, 0]), it)
    vy = np.gradient(np.interp(it, lp_t, lp[:, 1]), it)
    speed = np.hypot(vx, vy)

    valid = np.where((h > 0.5) & (np.diff(it, prepend=it[0]) > 0.05))[0]
    valid = valid[valid > 0]

    def top(metric, n, reverse=True):
        order = valid[np.argsort(metric[valid])]
        return list(order[-n:] if reverse else order[:n])

    rng = np.random.default_rng(42)
    idx = (top(speed, n_per_regime)                      # круиз (быстрое движение)
           + top(speed, n_per_regime, reverse=False)     # зависание
           + top(tilt, n_per_regime)                     # максимальное вращение
           + list(rng.choice(valid, n_per_regime, replace=False)))  # случайные
    idx = sorted(set(idx))

    i1s, i2s, qs, fxs, fys, tags = [], [], [], [], [], []
    for i in idx:
        i1, i2 = imgs[i - 1], imgs[i]
        q, fx, fy = reference(i1, i2)
        i1s.append(i1); i2s.append(i2)
        qs.append(q); fxs.append(fx); fys.append(fy)
        tags.append(f"t={it[i]-it[0]:.1f}s h={h[i]:.1f} v={speed[i]:.2f} tilt={np.degrees(tilt[i]):.0f}dps")

    np.savez_compressed(GOLDEN, imgs1=np.array(i1s, np.uint8), imgs2=np.array(i2s, np.uint8),
                        q=np.array(qs), fx=np.array(fxs), fy=np.array(fys),
                        tags=np.array(tags), flags=str(FLAGS), W=imgs.shape[1])
    # плоские файлы для C
    with open(FRAMES_BIN, "wb") as f:
        for a, b in zip(i1s, i2s):
            f.write(a.tobytes()); f.write(b.tobytes())
    with open(EXPECTED_CSV, "w") as f:
        f.write("idx,q,fx,fy\n")
        for k, (q, fx, fy) in enumerate(zip(qs, fxs, fys)):
            f.write(f"{k},{q},{fx:.6f},{fy:.6f}\n")
    print(f"golden: {len(idx)} пар -> {GOLDEN.name}, {FRAMES_BIN.name}, {EXPECTED_CSV.name}")
    for t, q, fx, fy in zip(tags, qs, fxs, fys):
        print(f"  {t:<44} q={q:>3} flow=({fx:+.3f},{fy:+.3f})")


def check():
    g = np.load(GOLDEN, allow_pickle=True)
    bad = 0
    for k in range(len(g["q"])):
        q, fx, fy = reference(g["imgs1"][k], g["imgs2"][k])
        if q != g["q"][k] or abs(fx - g["fx"][k]) > 1e-9 or abs(fy - g["fy"][k]) > 1e-9:
            bad += 1
            print(f"  MISMATCH #{k}: got q={q} f=({fx:.6f},{fy:.6f}) "
                  f"want q={g['q'][k]} f=({g['fx'][k]:.6f},{g['fy'][k]:.6f})")
    print(f"golden check (python): {'OK' if bad == 0 else 'FAIL'} "
          f"({len(g['q']) - bad}/{len(g['q'])})")
    return bad == 0


def check_c(csv_path):
    g = np.load(GOLDEN, allow_pickle=True)
    rows = [l.strip().split(",") for l in open(csv_path) if not l.startswith("idx")]
    bad = 0
    for k, q, fx, fy in ((int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in rows):
        dq = abs(q - int(g["q"][k]))
        df = max(abs(fx - g["fx"][k]), abs(fy - g["fy"][k]))
        if (int(g["q"][k]) > 0) != (q > 0) or (q > 0 and df > TOL_PX):
            bad += 1
            print(f"  MISMATCH #{k}: C q={q} f=({fx:.4f},{fy:.4f}) "
                  f"ref q={int(g['q'][k])} f=({g['fx'][k]:.4f},{g['fy'][k]:.4f}) dF={df:.4f}")
    print(f"golden check (C, допуск {TOL_PX}px): {'OK' if bad == 0 else 'FAIL'} "
          f"({len(rows) - bad}/{len(rows)})")
    return bad == 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "generate":
        generate()
    elif cmd == "check":
        sys.exit(0 if check() else 1)
    elif cmd == "check-c":
        sys.exit(0 if check_c(sys.argv[2]) else 1)
