#!/usr/bin/env bash
# Сценарий «генератор --- критик» для запуска в tmux на сервере курса.
# Тонкая обёртка над детерминированным Python-портом (одинаковые артефакты).
#
#   bash scenario/run_scenario.sh                         # обычный запуск
#   tmux new -d -s hw3 'bash scenario/run_scenario.sh'    # выживает обрыв связи
#   tmux attach -t hw3                                    # вернуться
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
exec "$PY" scenario/run_scenario.py "$@"
