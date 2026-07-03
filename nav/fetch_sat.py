"""
fetch_sat.py — скачивание спутниковой подложки (XYZ-тайлы) для map_localize.

Скачивает тайлы вокруг точки, сшивает в PNG, пишет мета (углы WGS84, GSD,
ENU-привязку относительно опорной точки) — готово для GeoMap.from_image().

Запуск (нужен интернет; из песочницы Claude не работает — запускать у себя):
    python3 src/fetch_sat.py                          # район багов, google+esri+bing, z19
    python3 src/fetch_sat.py --lat 56.83257 --lon 60.79319 --radius 300 --zoom 19 \
        --provider google --out data/sat
    python3 src/fetch_sat.py --dry-run                # только посчитать привязку/URL

Провайдеры: google, esri, bing (все — Web Mercator XYZ/quadkey).
Яндекс НЕ включён: у него эллиптический Меркатор — прямое смешение даст сдвиг ~десятки
метров по широте; при необходимости добавлять с отдельной репроекцией.

Использование результата:
    import json, map_localize as ML
    meta = json.load(open("data/sat/google_z19.json"))
    gm = ML.GeoMap.from_image("data/sat/google_z19.png",
                              meta["enu_x0"], meta["enu_y0"], meta["gsd"], highpass=True)
"""
import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import cv2

TILE = 256
UA = {"User-Agent": "Mozilla/5.0 (offline research; contact: local)"}

PROVIDERS = {
    "google": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "esri": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
             "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
    "bing": "http://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1",
}


# ---------- Web Mercator математика ----------
def deg2tile(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def tile2deg(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def gsd_at(lat, z):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** z)


def quadkey(x, y, z):
    q = ""
    for i in range(z, 0, -1):
        d = 0
        m = 1 << (i - 1)
        if x & m:
            d += 1
        if y & m:
            d += 2
        q += str(d)
    return q


def enu_offset(lat0, lon0, lat, lon):
    """(east, north) метров точки (lat,lon) от опорной (lat0,lon0)."""
    R = 6378137.0
    e = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    n = math.radians(lat - lat0) * R
    return e, n


# ---------- загрузка и сшивка ----------
def fetch(url, retries=2, timeout=10, verbose_fail=False):
    err = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                buf = np.frombuffer(r.read(), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception as e:
            err = e
            time.sleep(0.5 + k)
    if verbose_fail:
        print(f"  FAIL {url}: {err}", flush=True)
    return None


def grab(provider, lat, lon, radius_m, zoom, out_dir, ref_lat, ref_lon, dry=False):
    xf, yf = deg2tile(lat, lon, zoom)
    r_tiles = int(math.ceil(radius_m / (gsd_at(lat, zoom) * TILE))) + 1
    x0t, y0t = int(xf) - r_tiles, int(yf) - r_tiles
    x1t, y1t = int(xf) + r_tiles, int(yf) + r_tiles
    nx, ny = x1t - x0t + 1, y1t - y0t + 1

    # привязка мозаики: NW угол тайла (x0t,y0t), SE угол тайла (x1t+1,y1t+1)
    lat_nw, lon_nw = tile2deg(x0t, y0t, zoom)
    lat_se, lon_se = tile2deg(x1t + 1, y1t + 1, zoom)
    gsd = gsd_at(lat, zoom)
    # ENU углов относительно опорной точки (для GeoMap: origin = ЛЕВЫЙ-НИЖНИЙ)
    e_w, n_n = enu_offset(ref_lat, ref_lon, lat_nw, lon_nw)
    e_e, n_s = enu_offset(ref_lat, ref_lon, lat_se, lon_se)

    print(f"[{provider}] z={zoom} тайлы {nx}x{ny} ({nx*ny} шт), GSD={gsd:.3f} м/px")
    print(f"  NW {lat_nw:.6f},{lon_nw:.6f}  SE {lat_se:.6f},{lon_se:.6f}")
    print(f"  ENU от опоры: x0(запад)={e_w:.1f} y0(юг)={n_s:.1f} "
          f"x1={e_e:.1f} y1={n_n:.1f}")
    if dry:
        u = PROVIDERS[provider]
        ex = (u.format(q=quadkey(x0t, y0t, zoom)) if provider == "bing"
              else u.format(x=x0t, y=y0t, z=zoom))
        print(f"  пример URL: {ex}")
        return

    # предпроверка сети: один центральный тайл, с немедленным вердиктом
    u = PROVIDERS[provider]
    cx, cy = int(xf), int(yf)
    test_url = (u.format(q=quadkey(cx, cy, zoom)) if provider == "bing"
                else u.format(x=cx, y=cy, z=zoom))
    print(f"  предпроверка: {test_url}", flush=True)
    t0 = time.time()
    probe = fetch(test_url, verbose_fail=True)
    if probe is None:
        print(f"  !! сервер {provider} недоступен ({time.time()-t0:.0f} с) — "
              f"пропускаю (сеть/блокировка? попробуй другой --provider или VPN)",
              flush=True)
        return
    print(f"  сеть ок ({time.time()-t0:.1f} с/тайл) — качаю {nx*ny} тайлов, "
          f"~{nx*ny*(0.15+time.time()-t0)/60:.1f} мин", flush=True)

    mosaic = np.zeros((ny * TILE, nx * TILE, 3), np.uint8)
    ok = fail = 0
    t0 = time.time()
    for ty in range(y0t, y1t + 1):
        for tx in range(x0t, x1t + 1):
            url = (u.format(q=quadkey(tx, ty, zoom)) if provider == "bing"
                   else u.format(x=tx, y=ty, z=zoom))
            img = fetch(url)
            if img is None:
                fail += 1
            else:
                iy, ix = ty - y0t, tx - x0t
                mosaic[iy * TILE:(iy + 1) * TILE,
                       ix * TILE:(ix + 1) * TILE] = img
                ok += 1
            time.sleep(0.15)          # вежливая пауза
        done = (ty - y0t + 1) * nx
        eta = (time.time() - t0) / done * (nx * ny - done)
        print(f"  ряд {ty-y0t+1}/{ny}: ok {ok}, fail {fail}, "
              f"осталось ~{eta/60:.1f} мин", flush=True)
    print(f"  скачано {ok}/{nx*ny}" + (f" (fail {fail})" if fail else ""))

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{provider}_z{zoom}"
    cv2.imwrite(str(base) + ".png", mosaic)
    cv2.imwrite(str(base) + "_gray.png",
                cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY))
    # NB: строки мозаики идут с СЕВЕРА вниз; GeoMap.world2map растит row с юга.
    # Для GeoMap.from_image отдаём вертикально отражённую (юг внизу -> origin ЛН).
    cv2.imwrite(str(base) + "_gray_flip.png",
                cv2.flip(cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY), 0))
    meta = dict(provider=provider, zoom=zoom, gsd=gsd,
                center_lat=lat, center_lon=lon,
                nw=[lat_nw, lon_nw], se=[lat_se, lon_se],
                ref_lat=ref_lat, ref_lon=ref_lon,
                enu_x0=e_w, enu_y0=n_s, enu_x1=e_e, enu_y1=n_n,
                image_for_geomap=str(base) + "_gray_flip.png",
                note="GeoMap.from_image(image_for_geomap, enu_x0, enu_y0, gsd)")
    json.dump(meta, open(str(base) + ".json", "w"), indent=1)
    print(f"  -> {base}.png / _gray_flip.png / .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=56.83257)   # район багов
    ap.add_argument("--lon", type=float, default=60.79319)
    ap.add_argument("--radius", type=float, default=300.0)
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--provider", default="all",
                    choices=list(PROVIDERS) + ["all"])
    ap.add_argument("--out", default="data/sat")
    ap.add_argument("--ref-lat", type=float, default=None,
                    help="опорная точка ENU (по умолчанию = центр)")
    ap.add_argument("--ref-lon", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    provs = list(PROVIDERS) if a.provider == "all" else [a.provider]
    ref_lat = a.ref_lat if a.ref_lat is not None else a.lat
    ref_lon = a.ref_lon if a.ref_lon is not None else a.lon
    for p in provs:
        grab(p, a.lat, a.lon, a.radius, a.zoom, Path(a.out),
             ref_lat, ref_lon, dry=a.dry_run)
