"""
map_localize.py — АБСОЛЮТНАЯ локализация по подложке (map-matching), отдельный слой.

Назначение: миссии в координатах ГНСС (QGC/Mission Planner) без ГНСС. Вид вниз
матчится к геопривязанной карте (спутниковый ортофотоплан / своя мозаика) ->
абсолютная позиция -> отдать в FC тем же путём, каким уже инжектится GPS
(MAVLink GPS_INPUT / mavros gps_input), т.е. миссии в WGS84 работают как есть.

НЕ трогает optical flow: одометрия остаётся прайором, этот слой лишь обнуляет
её дрейф абсолютной поправкой, когда матч уверенный.

Демо на реальном баге (mosaic == "спутник"): карта строится из ПЕРВОЙ части
полёта по бортовым позам, кадры ПОСЛЕДНЕЙ части локализуются абсолютно с
зашумлённым прайором (имитация дрейфа одометрии).

Запуск:  python3 src/map_localize.py           # демо на баге-1
"""
import sys
import numpy as np
import cv2

_p = __import__("pathlib").Path(__file__).resolve()
sys.path.insert(0, str(_p.parent))          # nav/
sys.path.insert(0, str(_p.parent.parent / "src"))   # ядро optical flow
from _paths import DATA, RESULTS_NAV as RESULTS

GSD = 0.15          # м/пиксель карты (реальные ортоподложки: 0.1-0.6)
CONF_THR = 0.20     # порог уверенности (калиброван по лоскутной демо-мозаике;
                    # на настоящем ортофотоплане перекалибровать по своим данным)


def _hp(img, k=21):
    """High-pass (вычесть локальное среднее) — кросс-сенсорная нормализация:
    убирает разницу яркости/цвета спутник vs камера, оставляет структуру
    (дороги, границы полей, здания). Ядро крупнее, чем в потоке (структуры крупные)."""
    g = img.astype(np.float32)
    return np.clip(g - cv2.boxFilter(g, -1, (k, k)) + 128, 0, 255).astype(np.uint8)


class GeoMap:
    """Карта: растр север-вверх + привязка (origin в локальных метрах, GSD).
    Для реального применения растр = тайл ортофотоплана, origin = его геопривязка
    (ENU от опорной WGS84-точки); здесь строится мозаикой из кадров."""

    def __init__(self, x0, y0, w_m, h_m, gsd=GSD):
        self.x0, self.y0, self.gsd = x0, y0, gsd
        self.img = np.zeros((int(h_m / gsd), int(w_m / gsd)), np.uint8)
        self.mask = np.zeros_like(self.img)

    @classmethod
    def from_image(cls, path, x0, y0, gsd, highpass=True):
        """Подложка из готового снимка (спутниковый тайл/скрин SAS.Planet/GeoTIFF-
        экспорт): файл + мировые координаты ЛЕВОГО-НИЖНЕГО угла + м/px.
        highpass=True — кросс-сенсорный препроцесс (вычесть локальное среднее):
        гасит радиометрическую разницу спутник≠наша камера (сезон/солнце/сенсор);
        тот же препроцесс применить к живым кадрам (use_highpass в потоке или
        localize(..., highpass=True))."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        g = cls(x0, y0, img.shape[1] * gsd, img.shape[0] * gsd, gsd)
        g.img = _hp(img) if highpass else img
        g.mask = np.full_like(g.img, 255)
        g.hp = highpass
        return g

    def world2map(self, x, y):
        return (x - self.x0) / self.gsd, (y - self.y0) / self.gsd

    def map2world(self, mx, my):
        return self.x0 + mx * self.gsd, self.y0 + my * self.gsd

    def _affine(self, frame_w, yaw, h_agl, focal_px, cx_map, cy_map):
        """cam px -> map px: масштаб (h/f)/gsd и поворот R(-yaw) (как в одометрии)."""
        s = (h_agl / focal_px) / self.gsd
        c, sn = np.cos(-yaw), np.sin(-yaw)
        A = s * np.array([[c, -sn], [sn, c]])
        ctr = A @ np.array([frame_w / 2, frame_w / 2])
        return np.hstack([A, [[cx_map - ctr[0]], [cy_map - ctr[1]]]])

    def paste(self, frame, x, y, yaw, h_agl, focal_px):
        mx, my = self.world2map(x, y)
        M = self._affine(frame.shape[1], yaw, h_agl, focal_px, mx, my)
        warped = cv2.warpAffine(frame, M, self.img.shape[::-1],
                                flags=cv2.INTER_LINEAR)
        wmask = cv2.warpAffine(np.full_like(frame, 255), M, self.img.shape[::-1])
        sel = wmask > 128
        self.img[sel] = warped[sel]
        self.mask[sel] = 255

    def make_template(self, frame, yaw, h_agl, focal_px):
        """Кадр -> север-вверх шаблон в масштабе карты (центральный вписанный
        квадрат повёрнутого кадра — без маски, годен для TM_CCOEFF_NORMED)."""
        W = frame.shape[1]
        s = (h_agl / focal_px) / self.gsd
        side_cam = W / (abs(np.cos(yaw)) + abs(np.sin(yaw)))   # вписанный квадрат
        side_map = max(int(side_cam * s) - 2, 16)
        canvas = int(np.ceil(W * s * 1.5))
        M = self._affine(W, yaw, h_agl, focal_px, canvas / 2, canvas / 2)
        warped = cv2.warpAffine(frame, M, (canvas, canvas), flags=cv2.INTER_LINEAR)
        o = (canvas - side_map) // 2
        return warped[o:o + side_map, o:o + side_map]

    def localize(self, frame, yaw, h_agl, focal_px, prior_xy, search_m=12.0):
        """Абсолютная позиция по карте вокруг прайора (одометрии).
        Возвращает (x, y, confidence) либо None (нет уверенного матча)."""
        if getattr(self, "hp", False):
            frame = _hp(frame)          # кросс-сенсорная нормализация как у карты
        tpl = self.make_template(frame, yaw, h_agl, focal_px)
        ts = tpl.shape[0]
        pmx, pmy = self.world2map(*prior_xy)
        r = int(search_m / self.gsd)
        x0 = int(pmx - ts / 2 - r); y0 = int(pmy - ts / 2 - r)
        x1 = x0 + ts + 2 * r; y1 = y0 + ts + 2 * r
        Hm, Wm = self.img.shape
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(Wm, x1), min(Hm, y1)
        if x1 - x0 < ts + 4 or y1 - y0 < ts + 4:
            return None
        # валидность: заполнен ли футпринт шаблона у прайора (не всё окно —
        # карта-коридор всегда с дырами по краям)
        fx0 = int(np.clip(pmx - ts / 2, 0, Wm - 1)); fy0 = int(np.clip(pmy - ts / 2, 0, Hm - 1))
        foot = self.mask[fy0:fy0 + ts, fx0:fx0 + ts]
        if foot.size == 0 or foot.mean() < 150:      # <60% футпринта в карте
            return None
        mwin = self.mask[y0:y1, x0:x1]
        win = np.where(mwin > 0, self.img[y0:y1, x0:x1], 128).astype(np.uint8)
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, conf, _, loc = cv2.minMaxLoc(res)
        if conf < CONF_THR:
            return None
        # доминантность пика (anti-aperture на самоподобной траве/поле):
        # второй пик вне окрестности 1 м должен быть заметно ниже
        r1 = max(int(1.0 / self.gsd), 2)
        res2 = res.copy()
        yl, xl = loc[1], loc[0]
        res2[max(0, yl - r1):yl + r1 + 1, max(0, xl - r1):xl + r1 + 1] = -1
        if res2.max() > 0.85 * conf:
            return None
        mx = x0 + loc[0] + ts / 2
        my = y0 + loc[1] + ts / 2
        x, y = self.map2world(mx, my)
        return x, y, conf


def enu_to_wgs84(lat0, lon0, x_east, y_north):
    """Локальные метры (ENU от опорной точки) -> WGS84. Для GPS_INPUT в FC."""
    R = 6378137.0
    lat = lat0 + np.degrees(y_north / R)
    lon = lon0 + np.degrees(x_east / (R * np.cos(np.radians(lat0))))
    return lat, lon


# ------------------------- демо на реальном полёте -------------------------

def demo():
    d = np.load(DATA / "bag_flight" / "bag1_cam_cache.npz")  # ЧИСТАЯ камера /cam_usb
    W = int(d["W"]); focal = 551.8 * W / 480.0
    imgs, it = d["imgs"], d["img_t"]
    h = np.interp(it, d["rf_t"], d["rf_h"])
    yaw = np.interp(it, d["imu_t"], np.unwrap(d["yaw"]))
    Q = np.column_stack([np.interp(it, d["lp_t"], d["lp_xy"][:, 0]),
                         np.interp(it, d["lp_t"], d["lp_xy"][:, 1])])
    t = it - it[0]; T = t[-1]

    # карта из первых 60% полёта (роль "спутниковой подложки")
    gm = GeoMap(Q[:, 0].min() - 12, Q[:, 1].min() - 12,
                Q[:, 0].max() - Q[:, 0].min() + 24,
                Q[:, 1].max() - Q[:, 1].min() + 24)
    n_map = 0
    for i in range(0, len(imgs), 4):
        if t[i] > 0.6 * T or h[i] < 5:
            continue
        gm.paste(imgs[i], Q[i, 0], Q[i, 1], yaw[i], h[i], focal)
        n_map += 1
    cov = 100 * (gm.mask > 0).mean()
    print(f"карта: {gm.img.shape[1]}x{gm.img.shape[0]} px @{gm.gsd} м/px, "
          f"{n_map} кадров, покрытие {cov:.0f}%")
    cv2.imwrite(str(RESULTS / "map_mosaic.png"), gm.img)

    # самопроверка знаков: кадры ИЗ карты должны локализоваться в ноль
    self_err = []
    for i in range(0, len(imgs), 40):
        if t[i] > 0.6 * T or h[i] < 5:
            continue
        r = gm.localize(imgs[i], yaw[i], h[i], focal, Q[i], search_m=6)
        if r:
            self_err.append(np.hypot(r[0] - Q[i, 0], r[1] - Q[i, 1]))
    print(f"self-test (кадры из карты): медиана {np.median(self_err):.2f} м "
          f"(n={len(self_err)}) — должно быть ~GSD")

    # локализация кадров ПОСЛЕ 65% с зашумлённым прайором (дрейф одометрии ±6 м)
    rng = np.random.default_rng(1)
    errs, confs, n_try = [], [], 0
    for i in range(0, len(imgs), 8):
        if t[i] < 0.65 * T or h[i] < 5:
            continue
        n_try += 1
        prior = Q[i] + rng.uniform(-6, 6, 2)
        r = gm.localize(imgs[i], yaw[i], h[i], focal, prior, search_m=12)
        if r is None:
            continue
        errs.append(np.hypot(r[0] - Q[i, 0], r[1] - Q[i, 1]))
        confs.append(r[2])
    errs = np.array(errs)
    print(f"\nлокализация возврата (прайор ±6 м, поиск 12 м): "
          f"{len(errs)}/{n_try} уверенных ({100*len(errs)/max(n_try,1):.0f}%)")
    if len(errs):
        print(f"  ошибка абсолютной позиции: медиана {np.median(errs):.2f} м, "
              f"p90 {np.percentile(errs, 90):.2f} м, conf медиана {np.median(confs):.2f}")
    print("\nинтеграция: (x,y)->enu_to_wgs84->MAVLink GPS_INPUT (как ваш OF уже "
          "инжектит) => миссии QGC в WGS84 работают без изменений")
    return errs


if __name__ == "__main__":
    demo()
