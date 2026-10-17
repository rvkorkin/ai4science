---
name: trafficlight
description: Frac-hit prediction (DAS CNN+GRU) — Roman's TrafficLight project. Live GRE/YEL/RED app, training/eval, model experiments.
triggers: TrafficLight, traffic light, frac hit, GRE YEL RED, DAS CNN GRU, HFTS2, CrownQuest, SeqCNNGRU, wedge fusion, nn_seq_pipeline, shell.py
combines_with: knowledge-graph (для статей по frac hit), python (для анализа)
---

# TrafficLight workflow

## Расположение
Working copy: `work/trafficlight/` (относительно корня AI4Science).
Original production: `D:\work\TrafficLight` (НЕ трогать напрямую — работаем на копии).

## Python environment
Этот проект требует **отдельный** Python:
- `C:\Users\rkorkin\rkorkin\shad_env\Scripts\python.exe` (3.13 с numpy/torch/PyQt5)
- НЕ `python` на PATH и НЕ `.venv\Scripts\python.exe` AI4Science (там нет torch/PyQt)

Все команды запускай через полный путь:
```
shell("C:\\Users\\rkorkin\\rkorkin\\shad_env\\Scripts\\python.exe work\\trafficlight\\<script>.py")
```

## Current model
Config C (SeqCNNGRU, c_in=7, hidden=96, physics=on), AUC 0.960.
Thresholds: thr_red=0.553, thr_yel=0.321, persist=3.
Metrics: recall=0.84, FAR=0.14, median red lead=38 min.

## Что в проекте

### deploy/ — production app (не ломать)
- `shell.py` — PyQt5 UI (main entry)
- `nn_seq_pipeline.py` — SeqCNNGRU architecture + preprocess
- `inference.py` — single-snapshot wrapper
- `traffic.py` — GRE/YEL/RED state machine
- `wedge_near.py` — classical wedge (idea for fusion)
- `calibrate_thresholds.py` — threshold selection
- `whichpt.py` — model config self-check
- `nn_seq_model_best.pt` — current weights (config C)
- `traffic_thresholds.json` — thresholds used
- `sources/` — 10 sample npz files for emulator testing
- `incoming/` — folder watcher looks here

### core/ — research code (editable)
- `nn_seq_pipeline.py` — training version
- `preprocess.py`, `windowed_core.py`, `hazard_model.py`
- `split.json` — train/val split
- `field_strongest_hits.json` — ground truth labels
- `WORKLOG.md` — 52KB honest history, читай когда нужны детали прошлых экспериментов
- `ANALYSIS_SUMMARY.md` — current state summary (4 analyses + metrics)

### NOTES.md — краткая выжимка (читать первым)

## Dataset
Full field data: `D:\work\TrafficLight\field_data_enriched\` (2.44 GB, 623 npz, 470 recordings).
Путь в `.env` как `FIELD_DATA_DIR`. Код читает оттуда напрямую, не копируется.

## Типичные задачи

### 1. Проверить что модель жива
```
shell("C:\\Users\\rkorkin\\rkorkin\\shad_env\\Scripts\\python.exe work\\trafficlight\\deploy\\whichpt.py work\\trafficlight\\deploy\\nn_seq_model_best.pt")
```
Должно вывести config C.

### 2. Пересчитать thresholds на знее
```
shell("cd work\\trafficlight\\deploy && C:\\Users\\rkorkin\\rkorkin\\shad_env\\Scripts\\python.exe calibrate_thresholds.py --red 0.553")
```

### 3. Запустить live app с эмулятором (для разработки)
Два отдельных окна cmd нужны — shell и emulator одновременно.
```
shell("cd work\\trafficlight\\deploy && C:\\Users\\rkorkin\\rkorkin\\shad_env\\Scripts\\python.exe shell.py")
shell("cd work\\trafficlight\\deploy && C:\\Users\\rkorkin\\rkorkin\\shad_env\\Scripts\\python.exe emulator.py")
```

### 4. Что-то изменить в модели
- НИКОГДА не правь оригинал `D:\work\TrafficLight` — работаем в `work/trafficlight/`
- Перед правкой `nn_seq_pipeline.py` — git commit копии, если проект под git
- После изменений — retrain + сравнение AUC с 0.960 baseline
- Causal normalization (WARMUP_MIN=30) трогать нельзя — это защита от data leak

## Правила

- Всегда полный путь `C:\Users\rkorkin\rkorkin\shad_env\Scripts\python.exe` (бэкслэши удваивать в shell команде)
- Читать WORKLOG.md при вопросах "что уже пробовали X" — там честный лог
- Для вопросов по live app — сначала `deploy/README.md`
- При изменении production модели — `whichpt.py` на старую И новую, сравнить configs
- При добавлении данных нового месторождения — сначала convert_* скрипт нужен (в оригинале есть, в копии не брали — попросить добавить если понадобится)

## Что НЕ в этой копии (намеренно)
- GBM baseline (sweep_windows_gbm.py и tramvay) — устаревший, хуже CNN+GRU
- CNN per-crop (nn_pipeline.py в корне) — withdrawn из-за data leak
- Конверторы (convert_petrohunt, convert_vineyard, enrich_*) — не нужны пока нет новых источников
- Experimental diag scripts (_diag_*, _wedge_*, _hit_pick*) — черновики
- Business reports (business_plot, lead_report, baselines_compare) — одноразовые отчёты
- NeuralNet/, NeuralNet2/, Vista/, PRC/, mil_train_pkg/, base_model/ — архив

Если что-то из этого нужно — попросить "скопировать X из оригинала", не делать guess work.
