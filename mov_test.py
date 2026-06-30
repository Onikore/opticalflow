"""
mov_test.py — прогон на реальном снятом видео IMG_1248.MOV.
Геометрия: высота над полом h=0.75м, пройдено D≈0.50м по прямой.

Цель: проверить что flow-сенсор (наш улучшенный алгоритм) восстанавливает
это движение. Сверка с эталоном Farneback + пересчёт пикселей в метры.

Честная геометрия: берём КВАДРАТНЫЙ центр-кроп нативного кадра (без
анаморфного сжатия 720x1280->WxW), затем изотропный даунскейл до W.
m/px = 2*h*tan(FOV/2)/W;  расстояние = Σflow_px * m/px.
FOV не знаем точно -> показываем оценку для типичных значений телефона
и восстанавливаем подразумеваемый FOV из известных h и D.
"""
import numpy as np
import cv2
from functools import partial
from px4flow_improved import compute_flow_improved
from px4flow_fast import compute_flow_fast

IMPROVED = partial(compute_flow_improved, use_median=True, use_parabolic=True)
H = 0.75   # высота, м
D = 0.50   # пройдено, м


def load_square(path, W):
    v = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = v.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        s = min(g.shape)               # квадратный центр-кроп
        y0 = (g.shape[0] - s) // 2
        x0 = (g.shape[1] - s) // 2
        g = g[y0:y0 + s, x0:x0 + s]
        out.append(cv2.resize(g, (W, W)))
    return out


def integrate(fn, frames):
    tot = np.zeros(2)
    traj = [tot.copy()]
    quals, perframe = [], []
    for a, b in zip(frames, frames[1:]):
        q, fx, fy = fn(a, b)
        if q > 0:
            tot += (fx, fy)
            quals.append(q)
            perframe.append((fx, fy))
        traj.append(tot.copy())
    return tot, np.array(traj), np.array(perframe), quals


def farneback_integrate(frames):
    tot = np.zeros(2)
    pf = []
    for a, b in zip(frames, frames[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        d = np.array([fl[..., 0].mean(), fl[..., 1].mean()])
        tot += d
        pf.append(d)
    return tot, np.array(pf)


def report(W):
    frames = load_square("IMG_1248.MOV", W)
    print(f"\n================ W={W}px, кадров={len(frames)} ================")

    tot_o, _, _, _ = integrate(compute_flow_fast, frames)
    tot_i, traj, pf, quals = integrate(IMPROVED, frames)
    tot_f, pf_f = farneback_integrate(frames)

    mag_i = np.hypot(*tot_i)
    print(f"интеграл потока (px):  оригинал={np.hypot(*tot_o):6.1f}  "
          f"улучшенный={mag_i:6.1f}  Farneback={np.hypot(*tot_f):6.1f}")
    print(f"  улучшенный вектор=({tot_i[0]:+.1f},{tot_i[1]:+.1f})  "
          f"-> боковой дрейф {abs(tot_i[1])/mag_i*100:.1f}% (прямая линия => мало)")
    print(f"  средний quality={np.mean(quals):.0f}/255, валидных кадров={len(quals)}/{len(frames)-1}")

    # покадровая корреляция с эталоном
    n = min(len(pf), len(pf_f))
    cx = np.corrcoef(pf[:n, 0], pf_f[:n, 0])[0, 1]
    print(f"  корреляция с Farneback по X: {cx:+.3f}")

    # --- физика: расстояние при типичных FOV телефона ---
    print(f"  расстояние из потока (h={H}м):")
    for fov_deg in (50, 55, 60, 65):
        mpp = 2 * H * np.tan(np.radians(fov_deg) / 2) / W
        dist = mag_i * mpp
        print(f"     FOV={fov_deg}°: {dist*100:5.1f} см   (истина {D*100:.0f} см)")

    # обратная задача: какой FOV даёт ровно D=0.50м
    fov_impl = 2 * np.degrees(np.arctan(D * W / (2 * H * mag_i)))
    print(f"  => подразумеваемый горизонтальный FOV для D={D*100:.0f}см: {fov_impl:.1f}°"
          f"  (типично для телефона ~55-65°)")
    return traj


if __name__ == "__main__":
    for W in (64, 96):
        traj = report(W)
    # сохраним траекторию для наглядности
    import json
    np.save("mov_trajectory.npy", traj)
    print("\nтраектория сохранена -> mov_trajectory.npy")
