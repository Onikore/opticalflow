"""
algo_compare.py — наш блок-матчер vs DIS-Flow vs Lucas-Kanade на реальном полёте.

Все три считают средний поток кадр-к-кадру на ЧИСТОЙ камере (/cam_usb,
bag1_cam_cache.npz), интегрируются в траекторию одинаковым пайплайном
(высота × фокус × yaw) и сверяются с бортовой оценкой через Procrustes.

    python3 src/algo_compare.py            ->  results/flow/algo_trajectories.png
                                               results/flow/bag_vs_optflow.png
"""
import sys
from functools import partial

import cv2
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from px4flow_improved import compute_flow_improved as C
from _paths import DATA, RESULTS_FLOW as RESULTS

CACHE = DATA / "bag_flight" / "bag1_cam_cache.npz"
FOCAL_FULL = 551.8


def flow_ours(a, b):
    return C(a, b, use_median=True, use_pyramid=True, use_tri_subpix=True,
             use_zsad_fine=True, use_boundary_reject=True, flow_feature_threshold=15)


_dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

def flow_dis(a, b):
    f = _dis.calc(a, b, None)
    return 255, float(np.median(f[..., 0])), float(np.median(f[..., 1]))


def flow_lk(a, b):
    p0 = cv2.goodFeaturesToTrack(a, 40, 0.05, 6)
    if p0 is None or len(p0) < 5:
        return 0, 0.0, 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(15, 15), maxLevel=2)
    v = (p1 - p0)[st.ravel() == 1]
    if len(v) < 5:
        return 0, 0.0, 0.0
    return 255, float(np.median(v[:, 0, 0])), float(np.median(v[:, 0, 1]))


def _procrustes(P, Q):
    Pm, Qm = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pm.T @ Qm); R = (U @ Vt).T
    s = S.sum() / max((Pm ** 2).sum(), 1e-9)
    A = s * (Pm @ R.T) + Q.mean(0)
    return s, float(np.sqrt(np.mean(np.sum((A - Q) ** 2, 1)))), A


def run():
    d = np.load(CACHE)
    W = int(d["W"]); focal = FOCAL_FULL * W / 480.0
    imgs, it = d["imgs"], d["img_t"]
    h = np.interp(it, d["rf_t"], d["rf_h"])
    yaw = np.interp(it, d["imu_t"], np.unwrap(d["yaw"]))
    Q = np.column_stack([np.interp(it, d["lp_t"], d["lp_xy"][:, 0]),
                         np.interp(it, d["lp_t"], d["lp_xy"][:, 1])])
    algos = [("ours (SAD+флаги)", flow_ours), ("DIS-Flow (medium)", flow_dis),
             ("Lucas-Kanade", flow_lk)]
    out = []
    for name, fn in algos:
        N = len(imgs); wx = np.zeros(N); wy = np.zeros(N); nv = 0
        for i in range(1, N):
            dt = it[i] - it[i - 1]
            if dt <= 0 or dt > 1.0 or h[i] < 0.3:
                continue
            q, fx, fy = fn(imgs[i - 1], imgs[i])
            if q == 0:
                continue
            nv += 1
            dX = fx / focal * h[i]; dY = fy / focal * h[i]
            c, s = np.cos(-yaw[i]), np.sin(-yaw[i])
            wx[i] = c * dX - s * dY; wy[i] = s * dX + c * dY
        P = np.column_stack([np.cumsum(wx), np.cumsum(wy)])
        scale, resid, A = _procrustes(P, Q)
        print(f"{name:<18} resid={resid:.2f} м  масштаб={scale:.3f}  валидных {nv}/{N}")
        out.append((name, A, resid, scale))

    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (name, A, resid, scale) in zip(axes, out):
        ax.plot(Q[:, 0], Q[:, 1], '-', color='0.6', lw=2.5)
        ax.plot(A[:, 0], A[:, 1], '-', lw=1.2)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_title(f'{name}\nresid {resid:.2f} м, scale {scale:.2f}')
    fig.suptitle('Траектории на чистой камере (/cam_usb), баг-1, 109 м пути', fontsize=13)
    fig.tight_layout(); fig.savefig(RESULTS / "algo_trajectories.png", dpi=110)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [o[0] for o in out]; resids = [o[2] for o in out]
    ax.bar(names, resids, color=['tab:green', 'tab:blue', 'tab:orange'])
    for i, r in enumerate(resids):
        ax.text(i, r + .02, f'{r:.2f}', ha='center')
    ax.set_ylabel('resid к бортовой оценке, м'); ax.grid(alpha=.3, axis='y')
    ax.set_title('Наш блок-матчер vs OpenCV (чистая камера, баг-1)')
    fig.tight_layout(); fig.savefig(RESULTS / "bag_vs_optflow.png", dpi=120)
    print(f"графики -> {RESULTS}/algo_trajectories.png, bag_vs_optflow.png")


if __name__ == "__main__":
    run()
