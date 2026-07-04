/*
 * px4flow_ref.c — портируемая C-реализация улучшенного PX4Flow.
 * Полётный набор: median + pyramid(zero-mean coarse) + tri-subpix + MAD + boundary.
 * Эталон: src/px4flow_improved.py (те же флаги). Приёмка: golden-набор
 * (data/test_frames/golden_frames.bin), допуск 0.05 px.
 *
 * Целочисленный тракт: SAD/ZSAD — int (на STM32 -> __USADA8, 4 px/такт);
 * float только в субпикселе и агрегации (H743 имеет FPU; либо Q8 — проверено,
 * расхождение <0.002 px). Память: статические буферы, без malloc.
 *
 * Zero-mean coarse без float: |(a-ā)-(b-b̄)| * 64 = |64a - Sa - 64b + Sb| —
 * целочисленно, argmin совпадает с float-эталоном точно.
 *
 * Сборка (ПК):  gcc -O2 -o px4flow_ref px4flow_ref.c -lm
 * Запуск:       ./px4flow_ref golden_frames.bin > out.csv
 * Сверка:       python3 src/golden.py check-c out.csv
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define W        96          /* рабочее разрешение потока */
#define S        4           /* search_size */
#define T        8           /* TILE_SIZE */
#define NB       5           /* NUM_BLOCKS (сетка NB x NB) */
#define REFINE   2           /* fine-окно пирамиды +-REFINE */
#define PADF     (2*S+REFINE)
#define PADH     S
#define DW       (W/2)
#define PW       (W+2*PADF)  /* padded full */
#define PDW      (DW+2*PADH) /* padded down  */
#define FEATURE_THR  30
#define VALUE_THR    3000
#define MAD_K        2.0f
#define MIN_BLOCKS   10

static uint8_t  p1[PW*PW], p2[PW*PW];       /* padded кадры            */
static uint16_t d1[PDW*PDW], d2[PDW*PDW];   /* padded 2x2-суммы (x4)   */

/* ---- препроцесс ---- */
static void pad_u8(const uint8_t *src, uint8_t *dst, int w, int pad, int pw) {
    for (int y = 0; y < pw; y++) {
        int sy = y - pad; if (sy < 0) sy = 0; if (sy > w-1) sy = w-1;
        for (int x = 0; x < pw; x++) {
            int sx = x - pad; if (sx < 0) sx = 0; if (sx > w-1) sx = w-1;
            dst[y*pw+x] = src[sy*w+sx];
        }
    }
}
static void down_pad(const uint8_t *img, uint16_t *dst) {
    static uint16_t tmp[DW*DW];
    for (int y = 0; y < DW; y++)
        for (int x = 0; x < DW; x++)
            tmp[y*DW+x] = (uint16_t)(img[(2*y)*W+2*x] + img[(2*y)*W+2*x+1]
                                   + img[(2*y+1)*W+2*x] + img[(2*y+1)*W+2*x+1]);
    for (int y = 0; y < PDW; y++) {
        int sy = y - PADH; if (sy < 0) sy = 0; if (sy > DW-1) sy = DW-1;
        for (int x = 0; x < PDW; x++) {
            int sx = x - PADH; if (sx < 0) sx = 0; if (sx > DW-1) sx = DW-1;
            dst[y*PDW+x] = tmp[sy*DW+sx];
        }
    }
}

/* ---- текстурность блока (как PX4Flow::compute_diff) ---- */
static int compute_diff(const uint8_t *img, int ox, int oy) {
    int16_t p[4][4]; int acc = 0;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            p[r][c] = img[(oy+2+r)*W + ox+2+c];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++) acc += abs(p[r][c] - p[r+1][c]);
    for (int c = 0; c < 3; c++)
        for (int r = 0; r < 4; r++) acc += abs(p[r][c] - p[r][c+1]);
    return acc;
}

/* ---- coarse: целочисленный zero-mean SAD на 2x2-суммах ---- */
static uint32_t zsad64_down(const uint16_t *a, const uint16_t *b,
                            int ax, int ay, int bx, int by) {
    int32_t Sa = 0, Sb = 0; uint32_t acc = 0;
    for (int r = 0; r < T; r++)
        for (int c = 0; c < T; c++) {
            Sa += a[(ay+r)*PDW + ax+c];
            Sb += b[(by+r)*PDW + bx+c];
        }
    for (int r = 0; r < T; r++)
        for (int c = 0; c < T; c++) {
            int32_t da = 64*(int32_t)a[(ay+r)*PDW + ax+c] - Sa;
            int32_t db = 64*(int32_t)b[(by+r)*PDW + bx+c] - Sb;
            acc += (uint32_t)abs(da - db);
        }
    return acc;
}

/* ---- fine: SAD uint8. Три пути, бит-в-бит одинаковые:
 *  - PX4FLOW_USE_USADA8:     Cortex-M4/M7 DSP-инструкция (4 px/такт)
 *  - PX4FLOW_EMULATE_USADA8: та же логика в C — проверить SIMD-путь на ПК
 *  - иначе: скалярный референс                                        ---- */
#if defined(PX4FLOW_USE_USADA8)
static inline uint32_t usada8(uint32_t x, uint32_t y, uint32_t acc) {
    uint32_t r;
    __asm__ ("usada8 %0, %1, %2, %3" : "=r"(r) : "r"(x), "r"(y), "r"(acc));
    return r;
}
#elif defined(PX4FLOW_EMULATE_USADA8)
static inline uint32_t usada8(uint32_t x, uint32_t y, uint32_t acc) {
    for (int i = 0; i < 4; i++) {
        int a = (x >> (8*i)) & 0xFF, b = (y >> (8*i)) & 0xFF;
        acc += (uint32_t)abs(a - b);
    }
    return acc;
}
#endif

static uint32_t sad8x8(const uint8_t *a, const uint8_t *b,
                       int ax, int ay, int bx, int by) {
    uint32_t acc = 0;
#if defined(PX4FLOW_USE_USADA8) || defined(PX4FLOW_EMULATE_USADA8)
    for (int r = 0; r < T; r++) {
        const uint8_t *pa = a + (ay+r)*PW + ax, *pb = b + (by+r)*PW + bx;
        uint32_t a0, a1, b0, b1;                /* memcpy = unaligned-safe */
        memcpy(&a0, pa, 4); memcpy(&a1, pa+4, 4);
        memcpy(&b0, pb, 4); memcpy(&b1, pb+4, 4);
        acc = usada8(a0, b0, acc);
        acc = usada8(a1, b1, acc);
    }
#else
    for (int r = 0; r < T; r++)
        for (int c = 0; c < T; c++)
            acc += (uint32_t)abs((int)a[(ay+r)*PW + ax+c] - (int)b[(by+r)*PW + bx+c]);
#endif
    return acc;
}

/* ---- триангулярный (V) субпиксель — правильный для L1 ---- */
static float tri_subpix(const uint32_t *v, int c, int n) {
    if (c <= 0 || c >= n-1) return 0.0f;
    float lo = (float)v[c-1], cc = (float)v[c], hi = (float)v[c+1], d;
    if (lo == hi) return 0.0f;
    if (lo > hi) { if (lo <= cc) return 0.0f; d = 0.5f*(lo-hi)/(lo-cc); }
    else         { if (hi <= cc) return 0.0f; d = 0.5f*(lo-hi)/(hi-cc); }
    if (d > 0.5f) d = 0.5f;
    if (d < -0.5f) d = -0.5f;
    return d;
}

/* ---- медиана как np.median (чётное n -> среднее двух средних) ---- */
static float median_f(float *v, int n) {
    for (int i = 1; i < n; i++) {          /* insertion sort, n<=25 */
        float k = v[i]; int j = i-1;
        while (j >= 0 && v[j] > k) { v[j+1] = v[j]; j--; }
        v[j+1] = k;
    }
    return (n & 1) ? v[n/2] : 0.5f*(v[n/2-1] + v[n/2]);
}

/* ---- главная функция: два кадра WxW -> (quality, flow_x, flow_y) ---- */
int compute_flow(const uint8_t *img1, const uint8_t *img2,
                 float *flow_x, float *flow_y) {
    pad_u8(img1, p1, W, PADF, PW);
    pad_u8(img2, p2, W, PADF, PW);
    down_pad(img1, d1);
    down_pad(img2, d2);

    float fxs[NB*NB], fys[NB*NB];
    int   n = 0;

    /* сетка блоков — как _block_positions */
    const int lo = S + 1, hi = W - (S+1) - T;
    const int step = (hi - lo) / NB + 1;

    for (int oy = lo; oy < hi; oy += step)
    for (int ox = lo; ox < hi; ox += step) {
        if (compute_diff(img1, ox, oy) < FEATURE_THR) continue;

        /* --- coarse на 1/2 разрешении, zero-mean, окно +-S --- */
        uint32_t best = UINT32_MAX; int cby = 0, cbx = 0;
        int ax = ox/2 + PADH, ay = oy/2 + PADH;
        for (int wy = 0; wy <= 2*S; wy++)
        for (int wx = 0; wx <= 2*S; wx++) {
            uint32_t c = zsad64_down(d1, d2, ax, ay, ax + wx - S, ay + wy - S);
            if (c < best) { best = c; cby = wy; cbx = wx; }
        }
        /* boundary_reject: coarse-минимум на краю окна -> ненадёжно */
        if (cby == 0 || cby == 2*S || cbx == 0 || cbx == 2*S) continue;
        int cdx = 2*(cbx - S), cdy = 2*(cby - S);

        /* --- fine +-REFINE вокруг 2x coarse --- */
        uint32_t cost[2*REFINE+1][2*REFINE+1];
        best = UINT32_MAX; int by = 0, bx = 0;
        int fx0 = ox + PADF, fy0 = oy + PADF;
        for (int wy = 0; wy <= 2*REFINE; wy++)
        for (int wx = 0; wx <= 2*REFINE; wx++) {
            uint32_t c = sad8x8(p1, p2, fx0, fy0,
                                fx0 + cdx + wx - REFINE, fy0 + cdy + wy - REFINE);
            cost[wy][wx] = c;
            if (c < best) { best = c; by = wy; bx = wx; }
        }
        if (best >= VALUE_THR) continue;

        /* --- субпиксель по строке/столбцу cost --- */
        uint32_t row[2*REFINE+1], col[2*REFINE+1];
        for (int k = 0; k <= 2*REFINE; k++) { row[k] = cost[by][k]; col[k] = cost[k][bx]; }
        float sx = tri_subpix(row, bx, 2*REFINE+1);
        float sy = tri_subpix(col, by, 2*REFINE+1);

        fxs[n] = (float)(cdx + bx - REFINE) + sx;
        fys[n] = (float)(cdy + by - REFINE) + sy;
        n++;
    }

    if (n <= MIN_BLOCKS) { *flow_x = 0; *flow_y = 0; return 0; }

    /* --- MAD-фильтр + медиана --- */
    float tx[NB*NB], ty[NB*NB], dev[NB*NB], keepx[NB*NB], keepy[NB*NB];
    memcpy(tx, fxs, n*sizeof(float)); memcpy(ty, fys, n*sizeof(float));
    float mx = median_f(tx, n), my = median_f(ty, n);
    for (int i = 0; i < n; i++)
        dev[i] = sqrtf((fxs[i]-mx)*(fxs[i]-mx) + (fys[i]-my)*(fys[i]-my));
    memcpy(tx, dev, n*sizeof(float));
    float mad = median_f(tx, n) + 1e-6f;
    int nk = 0;
    for (int i = 0; i < n; i++)
        if (dev[i] <= MAD_K*mad) { keepx[nk] = fxs[i]; keepy[nk] = fys[i]; nk++; }
    if (nk < 5) { nk = n; memcpy(keepx, fxs, n*sizeof(float)); memcpy(keepy, fys, n*sizeof(float)); }

    *flow_x = median_f(keepx, nk);
    *flow_y = median_f(keepy, nk);
    int q = n * 255 / (NB*NB);
    return q > 255 ? 255 : q;
}

/* ---- приёмочный прогон по golden-набору (ПК). На STM32 этот main не нужен:
 *      компилировать с -DPX4FLOW_NO_MAIN и звать compute_flow() из своего main
 *      (см. c/stm32/main_stm32_golden.c). ---- */
#ifndef PX4FLOW_NO_MAIN
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s golden_frames.bin\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    static uint8_t i1[W*W], i2[W*W];
    printf("idx,q,fx,fy\n");
    for (int k = 0; ; k++) {
        if (fread(i1, 1, W*W, f) != W*W) break;
        if (fread(i2, 1, W*W, f) != W*W) break;
        float fx, fy;
        int q = compute_flow(i1, i2, &fx, &fy);
        printf("%d,%d,%.6f,%.6f\n", k, q, fx, fy);
    }
    fclose(f);
    return 0;
}
#endif /* PX4FLOW_NO_MAIN */
