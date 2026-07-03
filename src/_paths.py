"""Пути репозитория (файлы кода в src/, данные в data/, выходы в results/).
Все пути считаются от корня репо, не от текущего каталога, чтобы скрипты
запускались из любого места: `python3 src/xxx.py`."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REAL_FRAMES = DATA / "real_frames"
VIDEOS = DATA / "videos"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
RESULTS_FLOW = RESULTS / "flow"    # ядро optical flow (алгоритм/сенсорика)
RESULTS_NAV = RESULTS / "nav"      # GNSS-denied навигация (карты/спутник/пути)
RESULTS_FLOW.mkdir(exist_ok=True)
RESULTS_NAV.mkdir(exist_ok=True)
