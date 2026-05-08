#!/usr/bin/env bash
set -euo pipefail

day="${1:-01}"
case "$day" in
  00) script="scripts/day00_deep_learning_whole_picture.py" ;;
  01) script="scripts/day01_free_fall.py" ;;
  02) script="scripts/day02_oscillator.py" ;;
  03) script="scripts/day03_linear_regression_numpy.py" ;;
  04) script="scripts/day04_torch_tensor_autograd.py" ;;
  05) script="scripts/day05_fit_sine_torch.py" ;;
  06) script="scripts/day06_damped_oscillator_learning.py" ;;
  07) script="scripts/day07_review_and_play.py" ;;
  *) echo "Usage: ./run_day.sh 00..07"; exit 1 ;;
esac

python "$script"
