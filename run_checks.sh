#!/usr/bin/env bash
# Все self-check'и репозитория одной командой. Выход !=0 при любом провале.
set -e
cd "$(dirname "$0")"
echo "== px4flow_improved (все флаги) ==" && python3 src/px4flow_improved.py
echo "== эквивалентность fast==algo ==" && python3 src/test_equivalence.py | tail -1
echo "== flow_to_velocity (физика выхода) ==" && python3 src/flow_to_velocity.py
echo "== dronecan round-trip ==" && python3 src/dronecan_test.py | tail -1
echo "== де-ротация ==" && python3 src/derotate_test.py | tail -1
echo "== quality на слепых сценах ==" && python3 src/quality_test.py | tail -1
echo "== golden-набор (python) ==" && python3 src/golden.py check
if command -v gcc >/dev/null; then
  echo "== C-референс vs golden =="
  gcc -O2 -Wall -o /tmp/px4flow_ref c/px4flow_ref.c -lm
  /tmp/px4flow_ref data/test_frames/golden_frames.bin > /tmp/c_out.csv
  python3 src/golden.py check-c /tmp/c_out.csv
fi
echo "ВСЕ ПРОВЕРКИ ПРОШЛИ"
