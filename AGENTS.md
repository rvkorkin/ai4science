# AGENTS.md — контракт среды и запуск верификатора (ДЗ-2)

Проект: раннее предсказание frac-hit по LF-DAS (TrafficLight, SeqCNNGRU).
Автор: Roman Korkin. Курс ШАД «ИИ-ассистенты для исследователя. AI4Science».

> Полный контракт среды агента (ДЗ-1) — в `AGENT.md`. Этот файл — точка входа для
> проверяющего ДЗ-2: **как одной командой запустить верификатор**.

---

## 1. Проверка (verification) — одна команда

```bash
# окружение: Python 3.13, numpy/scipy/sklearn/hypothesis/pytest
python -m pytest -q checks
```

Запускается из корня репозитория на чистой копии. Ожидаемый вывод:
`14 passed, 6 skipped` (6 skipped — leakage-тесты, которым нужны закрытые полевые
данные; они проходят при `TL_CORE=<project>/core`, см. §3).

## 2. Численный эксперимент (K4) — цена look-ahead-утечки

```bash
python checks/experiment.py smoke     # быстрый прогон (2 seed)
python checks/experiment.py           # full: 5 seed → checks/results/experiment_metrics.json
```

Полностью синтетический и детерминированный — закрытые данные не нужны.
Вердикт печатается в конце (`ВЕРДИКТ: утечка завышает AUC = True`).

## 3. Полевая проверка данных (опционально, нужны закрытые данные)

```bash
set FIELD_DATA_DIR=<...>\field_data_enriched     # 626 npz (вне репозитория)
set TL_CORE=<project>\work\trafficlight\core      # split.json + лейблы
python checks/run_experiment.py                   # валидатор по всему корпусу + аудит split
python -m pytest -q checks                         # + 6 leakage-тестов → 20 passed
```

## 4. Формальная проверка (приложение, Lean 4)

`checks/CausalPreprocess.lean` проверяется облачным checker SciLib (MCP `lean_check`,
`scilibai.ru/grag`, GRC21): `success: true`, 0 `sorry`/`admit`/`axiom`. Это **приложение**
к вычислительному направлению (на оценку выставлено вычислительное).

## 5. Среда

- Python-окружение с GPU: `C:\Users\rkorkin\rkorkin\shad_env` (torch 2.6+cu124,
  numpy 2.5, scipy 1.18, sklearn 1.9, NVIDIA RTX 4000 Ada).
- Метка сдачи: `git tag hw2`. Воспроизведение — см. `REPORT.md` §7.
