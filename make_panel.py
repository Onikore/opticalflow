"""
make_panel.py — рендер сравнительной панели: одно тестовое видео,
4 алгоритма в ряд, под каждым своя панель. Пишет mp4 (headless-friendly).

Колонки:
  1. Оригинал (SAD, среднее)         — px4flow_fast
  2. + медиана
  3. + медиана + парабола (ПРИНЯТО)
  4. + census (eps=10)

На каждой панели: рабочий кадр (то, что видит алгоритм), мгновенная
стрелка потока (направление движения камеры), накопленная траектория
(показывает прямую линию и боковой дрейф), quality и пройденный путь в px.
"""
import sys
from functools import partial

import cv2
import numpy as np

from px4flow_fast import compute_flow_fast
from px4flow_improved import compute_flow_improved as C

W = 96          # рабочее разрешение потока
PANEL = 300     # размер панели для показа
HEAD = 34       # высота заголовка
TRAJ_SCALE = 1.0

VARIANTS = [
    ("Original (SAD, mean)", compute_flow_fast),
    ("+ median", partial(C, use_median=True)),
    ("+ median + parabolic", partial(C, use_median=True, use_parabolic=True)),
    ("+ census (eps=10)", partial(C, use_median=True, use_parabolic=True,
                                  use_census=True, census_eps=10,
                                  census_value_threshold=320)),
]


def load_square_gray(path, w):
    v = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = v.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        s = min(g.shape)
        y0, x0 = (g.shape[0] - s) // 2, (g.shape[1] - s) // 2
        out.append(cv2.resize(g[y0:y0 + s, x0:x0 + s], (w, w)))
    return out


def draw_panel(name, gray, fx, fy, q, traj_pts):
    """Одна колонка: заголовок + кадр с оверлеем."""
    disp = cv2.cvtColor(cv2.resize(gray, (PANEL, PANEL),
                                   interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    cx = cy = PANEL // 2

    # накопленная траектория (камера движется против текстуры -> минус)
    if len(traj_pts) > 1:
        pts = np.array(traj_pts, np.int32).reshape(-1, 1, 2)
        cv2.polylines(disp, [pts], False, (255, 180, 0), 2, cv2.LINE_AA)
    cv2.circle(disp, (cx, cy), 3, (255, 255, 255), -1)

    # мгновенная стрелка (инверсия знака -> «куда едет камера»)
    color = (0, 255, 0) if q > 50 else (0, 165, 255) if q > 0 else (0, 0, 255)
    ex = int(cx - fx * (PANEL / W) * 6.0)
    ey = int(cy - fy * (PANEL / W) * 6.0)
    cv2.arrowedLine(disp, (cx, cy), (ex, ey), color, 2, tipLength=0.3)

    # метрики
    mag = np.hypot(*traj_pts[-1] - np.array([cx, cy])) if traj_pts else 0
    for i, t in enumerate([f"q={q}", f"path={mag:.0f}px"]):
        org = (8, PANEL - 12 - i * 22)
        cv2.putText(disp, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # заголовок
    head = np.full((HEAD, PANEL, 3), 40, np.uint8)
    cv2.putText(head, name, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, disp])


def main(path, out="algo_panel.mp4"):
    frames = load_square_gray(path, W)
    print(f"кадров: {len(frames)}")

    n = len(VARIANTS)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                         (PANEL * n, PANEL + HEAD))

    acc = [np.zeros(2) for _ in VARIANTS]          # накопленный поток
    traj = [[np.array([PANEL // 2, PANEL // 2])] for _ in VARIANTS]

    prev = None
    for fr in frames:
        cols = []
        for k, (name, fn) in enumerate(VARIANTS):
            fx = fy = 0.0
            q = 0
            if prev is not None:
                q, fx, fy = fn(prev, fr)
                if q > 0:
                    acc[k] += (fx, fy)
                    p = np.array([PANEL // 2, PANEL // 2]) - acc[k] * TRAJ_SCALE
                    traj[k].append(np.clip(p, 0, PANEL - 1).astype(int))
            cols.append(draw_panel(name, fr, fx, fy, q, traj[k]))
        vw.write(np.hstack(cols))
        prev = fr

    vw.release()
    print(f"готово -> {out}")
    for k, (name, _) in enumerate(VARIANTS):
        print(f"  {name:<26} интеграл потока |{np.hypot(*acc[k]):.1f}|px")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "IMG_1252.MOV"
    main(src)
