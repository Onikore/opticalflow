# CubeIDE с нуля: прогон golden-теста на плате (первый запуск)

Пошаговая настройка STM32CubeIDE (Windows) для приёмки алгоритма на плате
БЕЗ камеры. Контекст: плата STM32H743 приехала, камеры ещё нет; тест гоняет
20 golden-пар кадров реального полёта из flash и меряет такты. Файлы:
`c/px4flow_ref.c`, `c/stm32/main_stm32_golden.c`, `c/stm32/golden_data.h`
(лежит в репо готовый; пересоздать — `python3 src/gen_golden_c.py`).

## 1. Создать проект

НЕ «Empty Project» (он без клоков и HAL). Правильно:

**File → New → STM32 Project** → откроется Target Selection (грузится ~минуту).
Вкладка **MCU/MPU Selector** → в поиск Commercial Part Number вбить маркировку
с большого чипа платы (типа `STM32H743VIT6`) → выбрать справа → **Next** →
Name `opticalflow`, Language **C**, Binary **Executable**, Project Type
**STM32Cube** → **Finish** → «Initialize all peripherals...» → **Yes**.

## 2. Конфигурация .ioc (графика с чипом)

- **System Core → SYS** → Debug: **Serial Wire** (иначе после прошивки плата
  перестанет отвечать отладчику!)
- **System Core → RCC** → HSE: **Crystal/Ceramic Resonator**
- **System Core → CORTEX_M7** → ICache **Enabled**, DCache **Enabled**
- **Connectivity → USART1** → Mode **Asynchronous** (115200 8N1 по умолчанию;
  пины обычно PA9=TX, PA10=RX — подсветятся на чипе)

## 3. Клоки

Вкладка **Clock Configuration** → в поле **HCLK** вписать `480`, Enter,
согласиться на авто-подбор. Если ошибка — проверить Input frequency (кварц):
на китайских платах (DevEBox/WeAct) обычно **25 МГц**, на Nucleo 8. Если 480
не выходит — 400, тогда в шаге 5 добавить дефайн `CPU_MHZ=400`.

## 4. Генерация + файлы

**Ctrl+S** → generate Yes. Перетащить мышкой в **Core/Src** три файла
(диалог → Copy files): `px4flow_ref.c`, `main_stm32_golden.c`, `golden_data.h`.

## 5. Настройки сборки

Правый клик по проекту → **Properties → C/C++ Build → Settings**:

- **MCU/MPU GCC Compiler → Preprocessor** → Define symbols → добавить:
  `PX4FLOW_NO_MAIN` и `PX4FLOW_USE_USADA8` (+ `CPU_MHZ=400`, если клок 400)
- **MCU/MPU GCC Compiler → Optimization** → **-O2** (дефолтный -O0 даёт
  тайминги в ~5 раз хуже)

## 6. main.c — только внутри USER CODE блоков

CubeIDE перегенерирует main.c при каждом изменении .ioc и СТИРАЕТ всё вне пар
`/* USER CODE BEGIN/END */`.

В `/* USER CODE BEGIN 0 */`:

```c
int _write(int fd, char *buf, int len) {
    HAL_UART_Transmit(&huart1, (uint8_t*)buf, len, HAL_MAX_DELAY);
    return len;
}
extern int run_golden_test(void);
```

В `/* USER CODE BEGIN 2 */` (внутри main, после инициализаций):

```c
run_golden_test();
```

## 7. Сборка и прошивка

Молоток (Ctrl+B) → `Build Finished. 0 errors`. ST-Link к SWDIO/SWCLK/GND/3.3V,
зелёный Run (▶). Первый раз: «ST-LINK firmware upgrade» — согласиться; Debug
Configuration — OK.

## 8. Вывод

USB-UART: RX переходника → PA9 платы, GND→GND. Терминал 115200 (в CubeIDE:
Window → Show View → Terminal → Serial Terminal, COM-порт переходника).
Reset на плате. Ожидается:

```
golden on-target: 20 пар, допуск 0.05 px
 0: q=255(255) fx=+0.625(+0.625) fy=+2.347(+2.347) ... OK
...
такты/кадр: min .., медиана .., max ..  (= .. / .. / .. мкс @ 480 МГц)
итог: ВСЕ ПРОШЛИ (20/20)
```

Интерпретация: 20/20 = порт на камне считает как валидированный Python;
медиана тактов /480 = мкс на кадр (бюджет: <1 мс/кадр с USADA8). Для замера
эффекта SIMD прошить второй раз БЕЗ `PX4FLOW_USE_USADA8` и сравнить медианы.

## Типичные грабли

1. Не видит ST-Link → драйвер ST-Link (ставится с CubeIDE) / другой USB-порт.
2. Мусор в терминале → скорость не 115200 или клок реально не 480 (см. шаг 3).
3. Ничего не печатает → TX/RX перепутаны; проверить, что USART1 включён в .ioc.
4. `undefined reference to run_golden_test` → файлы не в Core/Src или не
   .c-расширение.
5. FAIL по всем парам → забыт -O2? нет — оптимизация на результат не влияет,
   FAIL = реальная проблема, прислать вывод в чат.

## Что дальше (после зелёного прогона)

Записать в docs/EXPERIMENTS.md: медиану/мин/макс тактов с USADA8 и без,
клок, плату. Затем: SCCB-конфиг OV7725 (AEC/AGC OFF), чтение FIFO AL422B
(битбэнг → TIM+DMA), даунскейл 96×96 → тот же compute_flow, знаковый тест
гиро, FDCAN/DroneCAN (см. README раздел C-порта и docs/HARDWARE_LIMITS.md).
Маркировку платы записать сюда же (влияет на кварц/пины/способ прошивки).
