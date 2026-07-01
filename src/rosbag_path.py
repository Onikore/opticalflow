r"""
rosbag_path.py — прогон нашего optical flow на реальном полёте из rosbag (mcap).

Один раз парсит баг в кэш (data/bag_flight/bag_cache.npz), дальше гоняет алгоритм
из кэша (не перечитывая 1.16 ГБ). Строит траекторию, сверяет с бортовой оценкой
через Procrustes, рисует в results/.

Использование:
    python3 src/rosbag_path.py                      # из кэша (быстро)
    python3 src/rosbag_path.py /path/to/rosbag_dir  # перепарсить баг в кэш

Топики (downward VO-камера + телеметрия):
    /odometry/image/compressed  640x480 JPEG  -> flow
    /odometry/rangefinder       высота (масштаб)
    /mavros/imu/data            гиро + ориентация (yaw)
    /mavros/local_position/pose эталон (в этом баге ведётся их OF -> кросс-валидация)

ВАЖНО: в исходном баге НЕ было независимого GPS (кастомный OF инжектится как GPS),
поэтому сверка с local_position — это кросс-валидация двух OF по форме, а не сверка
с землёй. Масштаб нашего пути заякорен реально (calib focal + rangefinder). См.
docs/FINDINGS.md §3.
"""
import sys
from functools import partial

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from px4flow_improved import compute_flow_improved as C
from _paths import DATA, RESULTS

CACHE = DATA / "bag_flight" / "bag_cache.npz"
FOCAL_FULL = 551.8          # fx из /cam_usb/camera_info на 640 px
IMG_TOPICS = ["/odometry/image/compressed", "/odometry/rangefinder",
              "/mavros/imu/data", "/odometry/pose", "/mavros/local_position/pose"]


def extract(bag_dir, W=96):
    """Парсит mcap-баг в кэш. Требует mcap + mcap_ros2 (без установки ROS)."""
    import glob
    import cv2
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    path = glob.glob(f"{bag_dir}/*.mcap")[0]

    def crop_resize(g):
        return cv2.resize(g[:, 80:560], (W, W))

    def yaw_q(x, y, z, w):
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    d = {k: [] for k in ("img_t", "imgs", "rf_t", "rf_h", "imu_t", "gyro",
                         "yaw", "ref_t", "ref_xy", "lp_t", "lp_xy")}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, chan, msg, ros in reader.iter_decoded_messages(topics=IMG_TOPICS):
            tp, ts = chan.topic, msg.log_time * 1e-9
            if tp == "/odometry/image/compressed":
                g = cv2.imdecode(np.frombuffer(bytes(ros.data), np.uint8), cv2.IMREAD_GRAYSCALE)
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
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, W=W, **{k: np.array(v) for k, v in d.items()})
    print(f"кэш сохранён -> {CACHE}  ({len(d['imgs'])} кадров)")


def _procrustes(P, Q):
    Pm, Qm = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pm.T @ Qm); R = (U @ Vt).T
    s = S.sum() / max((Pm ** 2).sum(), 1e-9)
    A = s * (Pm @ R.T) + Q.mean(0)
    return s, float(np.sqrt(np.mean(np.sum((A - Q) ** 2, 1)))), A


def build_trajectory(ref="lp"):
    """Гоняет наш flow по кадрам, строит путь, выравнивает к эталону."""
    d = np.load(CACHE)
    W = int(d["W"]); focal = FOCAL_FULL * W / 480.0
    imgs, it = d["imgs"], d["img_t"]
    h = np.interp(it, d["rf_t"], d["rf_h"])
    yaw = np.interp(it, d["imu_t"], np.unwrap(d["yaw"]))
    rt, rxy = (d["lp_t"], d["lp_xy"]) if ref == "lp" else (d["ref_t"], d["ref_xy"])
    Q = np.column_stack([np.interp(it, rt, rxy[:, 0]), np.interp(it, rt, rxy[:, 1])])

    flow = partial(C, use_median=True, use_pyramid=True, use_boundary_reject=True)
    N = len(imgs); wx = np.zeros(N); wy = np.zeros(N); nvalid = 0
    for i in range(1, N):
        dt = it[i] - it[i - 1]
        if dt <= 0 or dt > 1.0 or h[i] < 0.3:
            continue
        q, fx, fy = flow(imgs[i - 1], imgs[i])
        if q == 0:
            continue
        nvalid += 1
        dX = fx / focal * h[i]; dY = fy / focal * h[i]     # метры на земле
        c, s = np.cos(-yaw[i]), np.sin(-yaw[i])            # поворот в мировой кадр
        wx[i] = c * dX - s * dY; wy[i] = s * dX + c * dY
    P = np.column_stack([np.cumsum(wx), np.cumsum(wy)])
    scale, resid, A = _procrustes(P, Q)
    print(f"валидных кадров: {nvalid}/{N}")
    print(f"vs {ref}: resid={resid:.2f} м, масштаб={scale:.3f}, путь эталона="
          f"{np.sum(np.hypot(*np.diff(Q, axis=0).T)):.0f} м")
    return A, Q, resid, scale


def plot(A, Q, resid, scale, out=None):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = out or (RESULTS / "bag_trajectory.png")
    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.plot(Q[:, 0], Q[:, 1], '-', color='0.55', lw=3, label='бортовая оценка (ведётся их OF)')
    ax.plot(A[:, 0], A[:, 1], '-', color='tab:green', lw=1.4, label='наш optical flow (выровнен)')
    ax.plot(Q[0, 0], Q[0, 1], 'ko', ms=9, label='старт')
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_xlabel('X, м'); ax.set_ylabel('Y, м')
    ax.set_title(f'Траектория нашим OF vs бортовая оценка\n'
                 f'совпадение формы {resid:.1f} м, относит. масштаб {scale:.2f} '
                 f'(GPS-denied, независимой земли нет)')
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"график -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract(sys.argv[1])
    if not CACHE.exists():
        sys.exit("нет кэша: запусти с путём к багу — python3 src/rosbag_path.py <bag_dir>")
    A, Q, resid, scale = build_trajectory("lp")
    plot(A, Q, resid, scale)
