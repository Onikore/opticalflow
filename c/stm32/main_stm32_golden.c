/*
 * main_stm32_golden.c — приёмка C-порта НА ПЛАТЕ, камера не нужна.
 *
 * Что проверяет на STM32H743:
 *   1. Корректность: 20 golden-пар реального полёта из flash -> compute_flow,
 *      сверка с эталоном (допуск 0.05 px) — тот же критерий, что на ПК.
 *   2. Скорость: такты DWT->CYCCNT на кадр (min/медиана/max), мкс @ 480 МГц.
 *      Отдельно виден эффект -DPX4FLOW_USE_USADA8 (DSP SAD, 4 px/такт).
 *   3. Память: всё статическое — .bss/.data видны в map-файле, malloc нет.
 *
 * Интеграция в проект CubeIDE/Makefile (3 файла, без HAL-зависимостей):
 *   - добавить px4flow_ref.c с -DPX4FLOW_NO_MAIN (и -DPX4FLOW_USE_USADA8)
 *   - добавить этот файл + golden_data.h (python3 src/gen_golden_c.py)
 *   - из main() после инициализации клоков/UART вызвать run_golden_test()
 *   - printf ретаргетить на UART (стандартный _write -> HAL_UART_Transmit)
 *
 * Проверка логики харнесса на ПК до прошивки:
 *   gcc -O2 -DHOST -DPX4FLOW_EMULATE_USADA8 c/stm32/main_stm32_golden.c \
 *       -x c c/px4flow_ref.c -DPX4FLOW_NO_MAIN -lm && ./a.out
 */
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "golden_data.h"

int compute_flow(const uint8_t *img1, const uint8_t *img2,
                 float *flow_x, float *flow_y);

#define TOL_PX 0.05f

/* ---- счётчик тактов ---- */
#ifdef HOST
#include <time.h>
static void cyc_init(void) {}
static uint32_t cyc_now(void) { return (uint32_t)clock(); }
#define CPU_MHZ 1  /* на ПК мкс не считаем, только логика */
#else
/* DWT без CMSIS-заголовков — только адреса регистров ARMv7-M */
#define DEMCR   (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
static void cyc_init(void) {
    DEMCR |= 1u << 24;          /* TRCENA */
    DWT_CYCCNT = 0;
    DWT_CTRL |= 1u;             /* CYCCNTENA */
}
static uint32_t cyc_now(void) { return DWT_CYCCNT; }
#ifndef CPU_MHZ
#define CPU_MHZ 480             /* H743 @ 480 МГц; поправить под свой клок */
#endif
#endif

static uint32_t cycles[GOLDEN_N];

static uint32_t med_u32(uint32_t *v, int n) {
    for (int i = 1; i < n; i++) {
        uint32_t k = v[i]; int j = i - 1;
        while (j >= 0 && v[j] > k) { v[j+1] = v[j]; j--; }
        v[j+1] = k;
    }
    return v[n/2];
}

int run_golden_test(void) {
    cyc_init();
    int fail = 0;
    uint32_t worst = 0, bestc = 0xFFFFFFFFu;
    printf("golden on-target: %d пар, допуск %.2f px\r\n", GOLDEN_N, (double)TOL_PX);
    for (int k = 0; k < GOLDEN_N; k++) {
        float fx, fy;
        uint32_t t0 = cyc_now();
        int q = compute_flow(golden_frames[k][0], golden_frames[k][1], &fx, &fy);
        uint32_t dt = cyc_now() - t0;
        cycles[k] = dt;
        if (dt > worst) worst = dt;
        if (dt < bestc) bestc = dt;
        float ex = golden_expected[k].fx, ey = golden_expected[k].fy;
        int eq = golden_expected[k].q;
        int ok = (q == eq) &&
                 (eq == 0 || (fabsf(fx - ex) <= TOL_PX && fabsf(fy - ey) <= TOL_PX));
        if (!ok) fail++;
        printf("%2d: q=%3d(%3d) fx=%+.3f(%+.3f) fy=%+.3f(%+.3f) %lu тактов %s\r\n",
               k, q, eq, (double)fx, (double)ex, (double)fy, (double)ey,
               (unsigned long)dt, ok ? "OK" : "FAIL");
    }
    uint32_t med = med_u32(cycles, GOLDEN_N);
    printf("---\r\nтакты/кадр: min %lu, медиана %lu, max %lu",
           (unsigned long)bestc, (unsigned long)med, (unsigned long)worst);
#ifndef HOST
    printf("  (= %lu / %lu / %lu мкс @ %d МГц)",
           (unsigned long)(bestc / CPU_MHZ), (unsigned long)(med / CPU_MHZ),
           (unsigned long)(worst / CPU_MHZ), CPU_MHZ);
#endif
    printf("\r\nитог: %s (%d/%d)\r\n",
           fail ? "FAIL" : "ВСЕ ПРОШЛИ", GOLDEN_N - fail, GOLDEN_N);
    return fail;
}

#ifdef HOST
int main(void) { return run_golden_test(); }
#endif
