# Приёмка порта на STM32 БЕЗ камеры

> **Первый раз в CubeIDE?** Пошаговый гайд: [CUBEIDE_SETUP.md](CUBEIDE_SETUP.md).

Плата уже позволяет проверить весь алгоритм: golden-набор (20 пар кадров
реального полёта) зашивается во flash, плата считает поток и сверяет с
эталоном. Камера нужна только для следующего этапа (захват DCMI/FIFO).

## Что проверяется на плате

1. **Корректность** — те же 20/20 пар с допуском 0.05 px, что и приёмка на ПК.
2. **Скорость** — такты DWT->CYCCNT на кадр (min/медиана/max → мкс @480 МГц);
   бюджет из доков «<1 мс/кадр с USADA8» проверяется измерением.
3. **SIMD** — `-DPX4FLOW_USE_USADA8` включает DSP-инструкцию (4 px/такт) в
   fine-SAD; корректность пути доказана на ПК эмуляцией (бит-в-бит с референсом).
4. **Память** — всё статическое: алгоритм ≈44 КБ .bss (влезает в DTCM 128К),
   golden-данные ≈360 КБ во flash (из 2 МБ), malloc отсутствует.

## Шаги

```bash
python3 src/gen_golden_c.py        # только если нужно пересоздать golden_data.h
                                   # (готовый уже лежит в репо рядом)
```

В проект CubeIDE / Makefile добавить 3 файла (HAL не нужен):

- `c/px4flow_ref.c` с дефайнами `-DPX4FLOW_NO_MAIN -DPX4FLOW_USE_USADA8`
- `c/stm32/main_stm32_golden.c` (+ `golden_data.h` рядом)
- в своём `main()` после клоков/UART: `extern int run_golden_test(void);
  run_golden_test();`
- `printf` ретаргетить на UART (обычный `_write()` → `HAL_UART_Transmit`);
  если клок не 480 МГц — собрать с `-DCPU_MHZ=<ваш>`.

Рекомендуемые флаги: `-O2 -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard`.
Для замера эффекта SIMD прошить дважды: с `PX4FLOW_USE_USADA8` и без.

## Проверка на ПК до прошивки (логика харнесса + SIMD-путь)

```bash
gcc -O2 -DHOST -DPX4FLOW_EMULATE_USADA8 -DPX4FLOW_NO_MAIN \
    c/stm32/main_stm32_golden.c c/px4flow_ref.c -lm -o /tmp/stm_host && /tmp/stm_host
```

Ожидаемый вывод на плате — 20 строк `OK`, затем:

```
такты/кадр: min .., медиана .., max ..  (= .. / .. / .. мкс @ 480 МГц)
итог: ВСЕ ПРОШЛИ (20/20)
```

Дальше (когда приедет камера): SCCB-конфиг OV7725 (**AEC/AGC OFF**),
чтение FIFO AL422B (битбэнг → TIM+DMA), даунскейл в 96×96 → этот же
`compute_flow`, знаковый тест гиро на столе, FDCAN/DroneCAN.
