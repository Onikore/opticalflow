r"""
rosbag_path.py — прогон нашего optical flow на реальном полёте из rosbag (mcap).

Один раз парсит баг в кэш (data/bag_flight/bag1_cam_cache.npz), дальше гоняет
алгоритм из кэша (не перечитывая гигабайты). Строит траекторию, сверяет с
бортовой оценкой через Procrustes, рисует и рендерит видео в results/nav/.

Использование:
    python3 nav/rosbag_path.py                       # баг-1 из кэша (быстро)
    python3 nav/rosbag_path.py --bag2                # баг-2 из кэша
    python3 nav/rosbag_path.py --video               # + видео полёта
    python3 nav/rosbag_path.py --gates               # + сравнение гейтов
    python3 nav/rosbag_path.py --extract <bag_dir>   # перепарсить баг в кэш

Камера: /cam_usb/image/compressed (640x480 JPEG ~105 Гц) — ЧИСТАЯ камера.
НЕ /odometry/image — это дебаг-топик с нарисованными жёлтыми маркерами фич
(загрязнял все ранние замеры, см. docs/EXPERIMENTS.md, аудит).
Дедупликация dt<1мс + прореживание каждый 8-й кадр -> ~12.5 Гц.

ВАЖНО: в багах НЕТ независимого GPS (кастомный OF инжектится как GPS), поэтому
сверка с local_position — кросс-валидация двух OF по форме, а не сверка с землёй.
Масштаб нашего пути заякорен реально (calib focal + rangefinder). FINDINGS.md §3.
"""
import sys
from functools import partial

import numpy as np

_p = __import__("pathlib").Path(__file__).resolve()
sys.path.insert(0, str(_p.parent))          # nav/
sys.path.insert(0, str(_p.parent.parent / "src"))   # ядро optical flow
from px4flow_improved import compute_flow_improved as C
from _paths import DATA, RESULTS_NAV as RESULTS

CACHE1 = DATA / "bag_flight" / "bag1_cam_cache.npz"
CACHE2 = DATA / "bag_flight" / "bag2_cam_cache.npz"
FOCAL_FULL = 551.8          # fx из /cam_usb/camera_info на 640 px
CAM_TOPIC = "/cam_usb/image/compressed"
TOPICS = [CAM_TOPIC, "/odometry/rangefinder", "/mavros/imu/data",
          "/odometry/pose", "/mavros/local_position/pose"]
SUBSAMPLE = 8               # 105 Гц -> ~12.5 Гц

# полётный набор после аудита на чистой камере (MAD отключён — вредит на
# слаботекстурной земле; threshold=15 — лучшая калибровка, см. EXPERIMENTS.md)
FLIGHT_FLAGS = dict(use_median=True, use_pyramid=True, use_tri_subpix=True,
                    use_zsad_fine=True, use_boundary_reject=True,
                    flow_feature_threshold=15)


def extract(bag_dir, cache, W=96):
    """Парсит mcap-баг в кэш. Устойчив к обрезанному mcap (запись прервана)."""
    import glob
    import cv2
    from mcap.reader import NonSeekingReader   # переживает обрезанный mcap
    from mcap_ros2.decoder import DecoderFactory
    path = sorted(glob.glob(f"{bag_dir}/*.mcap"))[0]   # map.mcap раньше rosbag_*_0.mcap
    print(f"читаю {path}")

    def crop_resize(g):
        return cv2.resize(g[:, 80:560], (W, W))

    def yaw_q(x, y, z, w):
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    d = {k: [] for k in ("img_t", "imgs", "rf_t", "rf_h", "imu_t", "gyro",
                         "yaw", "ref_t", "ref_xy", "lp_t", "lp_xy")}
    last_t, n_raw = -1.0, 0
    with open(path, "rb") as f:
        reader = NonSeekingReader(f, decoder_factories=[DecoderFactory()])
        it = reader.iter_decoded_messages(topics=TOPICS)
        while True:
            try:
                schema, chan, msg, ros = next(it)
            except StopIteration:
                break
            except Exception as e:          # обрезанный mcap — берём что есть
                print(f"  (обрыв mcap: {type(e).__name__}, взято что было)")
                break
            tp, ts = chan.topic, msg.log_time * 1e-9
            if tp == CAM_TOPIC:
                if ts - last_t < 1e-3:      # дубликаты кадров
                    continue
                last_t = ts; n_raw += 1
                if n_raw % SUBSAMPLE:       # прореживание до ~12.5 Гц
                    continue
                g = cv2.imdecode(np.frombuffer(bytes(ros.data), np.uint8),
                                 cv2.IMREAD_GRAYSCALE)
                if g is not None and g.shape == (480, 640):
                    d["img_t"].append(ts); d["imgs"].append(crop_resize(g))
            elif tp == "/odometry/rangefinder":
                d["rf_t"].append(ts); d["rf_h"].append(ros.range)
            elif tp == "/mavros/imu/data":
                g = ros.angular_velocity; q = ros.orientation
                d["imu_t"].append(ts); d["gyro"].append((g.x, g.y, g.z))
                d["yaw"].append(yaw_q(q.x, q.y, q.z, q.w))
            elif tp == "/odometry/pose":
                p = ros.pose.position; d["ref_t"].append(ts); d["ref_xy"].append((p.x, p.y))
            elif tp == "/mavros/local_position/pose":
                p = ros.pose.position; d["lp_t"].append(ts); d["lp_xy"].append((p.x, p.y))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, W=W, **{k: np.array(v) for k, v in d.items()})
    print(f"кэш сохранён -> {cache}  ({len(d['imgs'])} кадров из {n_raw} сырых)")


def _procrustes(P, Q):
    Pm, Qm = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pm.T @ Qm); R = (U @ Vt).T
    s = S.sum() / max((Pm ** 2).sum(), 1e-9)
    A = s * (Pm @ R.T) + Q.mean(0)
    return s, float(np.sqrt(np.mean(np.sum((A - Q) ** 2, 1)))), A


def build_trajectory(cache=CACHE1, flags=None, gyro_gate_px=None):
    """Гоняет наш flow по кадрам, строит путь, выравнивает к эталону.

    gyro_gate_px: если задан — пропускать кадры с вращательным потоком
    f·|ω|·dt больше порога (за потолком пирамиды алгоритм врёт уверенно).
    """
    d = np.load(cache)
    W = int(d["W"]); focal = FOCAL_FULL * W / 480.0
    imgs, it = d["imgs"], d["img_t"]
    h = np.interp(it, d["rf_t"], d["rf_h"])
    yaw = np.interp(it, d["imu_t"], np.unwrap(d["yaw"]))
    wmag = np.interp(it, d["imu_t"], np.hypot(d["gyro"][:, 0], d["gyro"][:, 1]))
    Q = np.column_stack([np.interp(it, d["lp_t"], d["lp_xy"][:, 0]),
                         np.interp(it, d["lp_t"], d["lp_xy"][:, 1])])

    flow = partial(C, **(flags or FLIGHT_FLAGS))
    N = len(imgs); wx = np.zeros(N); wy = np.zeros(N)
    qual = np.zeros(N); fxy = np.zeros((N, 2)); nvalid = 0
    for i in range(1, N):
        dt = it[i] - it[i - 1]
        if dt <= 0 or dt > 1.0 or h[i] < 0.3:
            continue
        if gyro_gate_px is not None and focal * wmag[i] * dt > gyro_gate_px:
            continue
        q, fx, fy = flow(imgs[i - 1], imgs[i])
        qual[i] = q; fxy[i] = (fx, fy)
        if q == 0:
            continue
        nvalid += 1
        dX = fx / focal * h[i]; dY = fy / focal * h[i]     # метры на земле
        c, s = np.cos(-yaw[i]), np.sin(-yaw[i])            # поворот в мировой кадр
        wx[i] = c * dX - s * dY; wy[i] = s * dX + c * dY
    P = np.column_stack([np.cumsum(wx), np.cumsum(wy)])
    scale, resid, A = _procrustes(P, Q)
    print(f"валидных кадров: {nvalid}/{N}, mean q={qual[qual > 0].mean():.0f}")
    print(f"resid={resid:.2f} м, масштаб={scale:.3f}, путь эталона="
          f"{np.sum(np.hypot(*np.diff(Q, axis=0).T)):.0f} м")
    return dict(A=A, Q=Q, resid=resid, scale=scale, imgs=imgs, qual=qual,
                fxy=fxy, h=h, nvalid=nvalid)


def plot(r, out, title):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    A, Q = r["A"], r["Q"]
    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.plot(Q[:, 0], Q[:, 1], '-', color='0.55', lw=3, label='бортовая оценка (ведётся их OF)')
    ax.plot(A[:, 0], A[:, 1], '-', color='tab:green', lw=1.4, label='наш optical flow (выровнен)')
    ax.plot(Q[0, 0], Q[0, 1], 'ko', ms=9, label='старт')
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_xlabel('X, м'); ax.set_ylabel('Y, м')
    ax.set_title(f'{title} — чистая камера /cam_usb\n'
                 f'совпадение формы {r["resid"]:.2f} м, относит. масштаб {r["scale"]:.2f} '
                 f'(GPS-denied, независимой земли нет)')
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"график -> {out}")


def render_video(r, out, fps=12.5):
    """Видео: кадр камеры + стрелка потока + quality + растущая траектория."""
    import cv2
    A, Q, imgs, qual, fxy = r["A"], r["Q"], r["imgs"], r["qual"], r["fxy"]
    S = 480; PW = 480                                  # кадр и панель карты
    lo = np.minimum(A.min(0), Q.min(0)) - 2; hi = np.maximum(A.max(0), Q.max(0)) + 2
    span = max(hi - lo); ctr = (lo + hi) / 2

    def m2px(p):
        xy = (p - ctr) / span * (PW - 40) + PW / 2
        return int(xy[0]), int(PW - xy[1])

    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (S + PW, S))
    canvas_map = np.full((S, PW, 3), 30, np.uint8)
    for i in range(1, len(Q)):
        cv2.line(canvas_map, m2px(Q[i - 1]), m2px(Q[i]), (90, 90, 90), 2)
    for i in range(1, len(imgs)):
        fr = cv2.cvtColor(cv2.resize(imgs[i], (S, S), interpolation=cv2.INTER_NEAREST),
                          cv2.COLOR_GRAY2BGR)
        q = qual[i]; fx, fy = fxy[i]
        col = (0, 220, 0) if q > 100 else ((0, 200, 255) if q > 0 else (0, 0, 255))
        cv2.arrowedLine(fr, (S // 2, S // 2),
                        (int(S / 2 + fx * 40), int(S / 2 + fy * 40)), col, 3, tipLength=.3)
        cv2.putText(fr, f"q={q:.0f} h={r['h'][i]:.1f}m flow=({fx:+.2f},{fy:+.2f})px",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        cv2.line(canvas_map, m2px(r["A"][i - 1]), m2px(r["A"][i]), (0, 200, 0), 1)
        m = canvas_map.copy()
        cv2.circle(m, m2px(r["A"][i]), 6, (0, 255, 255), -1)
        cv2.putText(m, "gray=onboard  green=ours", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (200, 200, 200), 1)
        vw.write(np.hstack([fr, m]))
    vw.release()
    print(f"видео -> {out}")


def gate_compare(cache=CACHE1, out=None):
    """Без защиты vs гиро-гейт (f·|ω|·dt < 7px, потолок пирамиды ~9.5px)."""
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = out or (RESULTS / "bag_gate_compare.png")
    r0 = build_trajectory(cache)
    r1 = build_trajectory(cache, gyro_gate_px=7.0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, r, name in [(axes[0], r0, "без защиты"),
                        (axes[1], r1, "гиро-гейт f·|ω|·dt < 7px")]:
        ax.plot(r["Q"][:, 0], r["Q"][:, 1], '-', color='0.55', lw=3)
        ax.plot(r["A"][:, 0], r["A"][:, 1], '-', color='tab:green', lw=1.2)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_title(f'{name}: resid {r["resid"]:.2f} м')
    fig.suptitle('Гейты на чистой камере (/cam_usb), баг-1', fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"график -> {out}")
    return r0["resid"], r1["resid"]


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--extract":
        cache = CACHE2 if "bag_2026" in args[1] or "18_02" in args[1] else CACHE1
        extract(args[1], cache); sys.exit(0)
    bag2 = "--bag2" in args
    cache = CACHE2 if bag2 else CACHE1
    if not cache.exists():
        sys.exit(f"нет кэша {cache}: python3 nav/rosbag_path.py --extract <bag_dir>")
    if "--gates" in args:
        gate_compare(cache); sys.exit(0)
    tag = "bag2" if bag2 else "bag1"
    r = build_trajectory(cache)
    plot(r, RESULTS / (f"{tag}_run.png" if bag2 else "bag_trajectory.png"),
         f"Траектория нашим OF, {'баг-2' if bag2 else 'баг-1'}")
    if "--video" in args:
        render_video(r, RESULTS / (f"{tag}_run.mp4" if bag2 else "final_run.mp4"))
