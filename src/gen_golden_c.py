"""
gen_golden_c.py — упаковка golden-набора в C-заголовок для прошивки STM32.

Кадры (20 пар 96x96) + эталонные ответы кладутся в const-массивы (~360 КБ
во flash, H743 имеет 2 МБ). На плате c/stm32/main_stm32_golden.c гоняет
compute_flow по парам и сверяет с эталоном — приёмка порта БЕЗ камеры.

    python3 src/gen_golden_c.py   ->  c/stm32/golden_data.h
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA, ROOT

W = 96
OUT = ROOT / "c" / "stm32" / "golden_data.h"


def main():
    frames = np.fromfile(DATA / "test_frames" / "golden_frames.bin", np.uint8)
    exp = np.genfromtxt(DATA / "test_frames" / "golden_expected.csv",
                        delimiter=",", names=True)
    n = len(exp)
    assert frames.size == n * 2 * W * W, "bin не сходится с csv"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("/* автогенерировано src/gen_golden_c.py — НЕ править руками */\n")
        f.write("#include <stdint.h>\n")
        f.write(f"#define GOLDEN_N {n}\n#define GOLDEN_W {W}\n")
        f.write(f"static const uint8_t golden_frames[{n}][2][{W*W}] = {{\n")
        pairs = frames.reshape(n, 2, W * W)
        for k in range(n):
            f.write("{")
            for j in range(2):
                f.write("{" + ",".join(map(str, pairs[k, j])) + "},")
            f.write("},\n")
        f.write("};\n")
        f.write("static const struct { int16_t q; float fx, fy; } "
                f"golden_expected[{n}] = {{\n")
        for r in exp:
            f.write(f"  {{{int(r['q'])}, {r['fx']:.6f}f, {r['fy']:.6f}f}},\n")
        f.write("};\n")
    print(f"-> {OUT} ({OUT.stat().st_size // 1024} КБ, {n} пар)")


if __name__ == "__main__":
    main()
