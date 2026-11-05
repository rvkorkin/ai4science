# AGENT.md — ИИ-ассистент для исследователя

**Автор:** Roman Korkin
**Курс:** ШАД «ИИ-ассистенты для исследователя» (AI4Science / Khalov), 2026
**Контекст:** контракт среды агента, обновлён к 8-й неделе курса (ДЗ-2 «Проверка» сдано).

> ДЗ-1 «Среда и контекст» описывало эту среду; здесь она актуализирована и дополнена
> разделом про верификатор ДЗ-2 (`verify/`) и расширение про неопределённость разметки.

---

## 1. Автор и домен

Principal Physicist. PhD по теоретической физике (кандидат физ.-мат. наук).

**Технические домены:**
- Fiber optic sensing (DAS/DSS)
- Well logging tools
- Hydraulic fracturing monitoring
- Geophysical inversion

**Основная задача курса:** построить модель раннего предсказания frac hits
(приближающийся прорыв трещины ГРП в мониторинговую скважину) по данным LF-DAS
strain rate. Это продолжение production-проекта TrafficLight: GRE/YEL/RED
live-мониторинг в PyQt5-приложении, текущая модель — SeqCNNGRU config C.

**Метрики модели** (honest pairloo OOF, leave-one-GROUP-out, group = treatment event):
- Window-level AUC = 0.960 (config C, deploy)
- File-level recall @ FAR ≤ 0.30 = 0.935, median lead = 38 min
- Known limitation: deployed = одна сохранённая модель (fold2 seed2) → data leak
  в оценке deployed, правильный fix — ensemble из 9 моделей либо retrain + hold-out.
  Задокументировано в `WORKLOG.md`.

---

## 2. Среда

### Harness
AI4Science / Goose 1.50 поверх Yandex AI Studio (OpenAI-совместимый endpoint).
Запуск: `bin\ai4science.bat`, `/help`, `/model <алиас>`, `/plan`, `/status`.
Бюджет: 3M токенов в сутки, сброс в полночь локального времени.

### Python environments
- **`.venv`** (Python 3.12.7) — harness, llama-index, Yandex API
- **`shad_env`** (Python 3.13.15) — torch 2.6+cu124, PyQt5, bge-m3 на GPU (NVIDIA RTX 4000 Ada)
- `C:\Users\rkorkin\AI4Science\AI4Science\.venv\Scripts\python.exe` — для Goose
- `C:\Users\rkorkin\rkorkin\shad_env\Scripts\python.exe` — для TrafficLight, graph_rag и верификатора

### Модели (Yandex AI Studio)
| Алиас | Slug | Назначение |
|---|---|---|
| qwen | qwen3.6-35b-a3b/latest | default |
| qwen235 | qwen3-235b-a22b-fp8/latest | /plan, сложные вопросы |
| gptoss | gpt-oss-120b/latest | fallback |
| deepseek | — | альтернатива |

### MCP-инструменты (17)
`token_budget`, `load_artifact`, `list_skills`, `load_skill`,
`lean_check`, `lean_health`, `lean_search_*`,
`web_search`, `web_fetch`, `pdf_download`, `pdf_info`, `pdf_read`, `pdf_search`.

---

## 3. Локальный Knowledge Graph

- **Папка:** `graph_storage/` (LlamaIndex PropertyGraphStore)
- **Embed-модель:** `BAAI/bge-m3` (multilingual, 1024-dim), на GPU через `shad_env`
- **Содержимое:** 208 text chunks из 9 PDF, с metadata (file_name, page_label)
- **Retrieval:** vector RAG (numpy cosine), top-5 чанков → синтез Qwen'ом с цитатами `[file p.N]`
- **Запуск:** `scripts\graph_rag.bat "query"` (+ `--retrieve-only --top-k N`, `--rebuild`)

---

## 4. Проект TrafficLight (безопасная копия)

Оригинал: `D:\work\TrafficLight` (production). Копия для AI4Science:
`C:\Users\rkorkin\AI4Science\AI4Science\work\trafficlight\`.
- `deploy/` — live app (shell.py, nn_seq_pipeline, inference, веса, пороги, `wedge_near.py`)
- `core/` — training code (nn_seq_pipeline, `preprocess.py`, windowed_core, split.json)
- `verify/` — **верификатор ДЗ-2** (property-тесты + валидатор данных + расширение)
- Данные: `FIELD_DATA_DIR=D:\work\TrafficLight\field_data_enriched` (626 npz) через `.env`.

---

## 5. Что сделано к 8-й неделе

### ДЗ-1 «Среда и контекст»
- Harness + Yandex API; vector RAG по 9 PDF (208 чанков × bge-m3 на GPU)
- Два профильных skill (knowledge-graph, trafficlight), `.goosehints`
- Безопасная копия TrafficLight для экспериментов

### ДЗ-2 «Проверка» (этот репозиторий)
- **Property-тесты причинности** предобработки (`decimate_causal`, `strain_rate_*`) на
  Hypothesis — отсутствие look-ahead-утечки доказано рандомизированным возмущением будущего.
- **Аудит split** на утечки (train∩val, покрытие лейблами, tpk-диапазон, well-группы).
- **Валидатор данных** — схема + физические инварианты 623 npz (0 ошибок).
- Численный эксперимент: `20 passed` (pytest; +2 property-теста добавлены агентом Goose),
  626/626 файлов валидны (0 ошибок).
- **Расширение:** неопределённость разметки → консенсус оценщиков (клин/активность/ручная) →
  soft-labels → опт-ин проводка в обучение; A/B на GPU показал рост детекции (recall@FAR≤0.10
  0.0→0.73, AUC 0.881→0.915) и near-field точности прихода.

---

## 6. Планы (недели 9–12)
1. **Статья (неделя 10):** учебные тезисы 2–4 стр. (аннотация, литобзор, методология,
   результаты, ограничения, ≤10 источников) — написать при помощи агента.
2. Строгий leave-one-WELL-out + несколько seed'ов для доверительных интервалов детекции.
3. Синтетические данные (Discontinuous Displacement Model + честная физика + полевой шум),
   уменьшение domain gap, вливание в train; уточнение качества лейблов (время и координата хита).
4. Расширение графа (+~15 PDF по DAS/fracturing), ребилд на GPU.
