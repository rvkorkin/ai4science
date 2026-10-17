# AGENT.md — ИИ-ассистент для исследователя

**Автор:** Roman Korkin  
**Курс:** ШАД «ИИ-ассистенты для исследователя» (AI4Science / Khalov), 2026  
**Домашнее задание 1:** Среда и контекст

---

## 1. Автор и домен

Principal Physicist.
PhD по теоретической физике (кандидат физ.-мат. наук).

**Технические домены:**
- Fiber optic sensing (DAS/DSS)
- Well logging tools
- Hydraulic fracturing monitoring
- Geophysical inversion

**Основная задача курса:** построить модель раннего предсказания frac hits
(приближающийся прорыв трещины ГРП в мониторинговую скважину) по данным LF-DAS
strain rate. Это продолжение production-проекта TrafficLight: GRE/YEL/RED
live-мониторинг в PyQt5-приложении, текущая модель — SeqCNNGRU config C.

**Метрики модели** (все измерены на leave-one-GROUP-out cross-validation,
где group = treatment event — чтобы данные одной скважины не утекали
между train и val):
- Window-level AUC = 0.960 (fold-level evaluation)
- File-level recall @ FAR ≤ 0.30 = 0.935, median lead = 38 min
- После threshold calibration на knee операционной точки (на 470 recordings
  `field_data_enriched`): recall = 0.84 / FAR = 0.14 / median red lead = 38 min

Важное: цифра AUC 0.960 — pooled OOF по 9 моделям (3 fold × 3 seed),
в deploy ушла последняя сохранённая модель (fold2 seed2). Это означает
data leak в оценке deployed модели — она видела при обучении файлы
из других fold'ов, на которых её метрики замерены OOF. Правильный fix —
либо ensemble из 9 моделей, либо отдельный retrain на всех данных +
hold-out set. Задокументировано как known limitation в WORKLOG.md.
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
- `C:\Users\rkorkin\rkorkin\shad_env\Scripts\python.exe` — для TrafficLight и graph_rag

### Модели (Yandex AI Studio)
| Алиас | Slug | Назначение |
|---|---|---|
| qwen | qwen3.6-35b-a3b/latest | default |
| qwen235 | qwen3-235b-a22b-fp8/latest | /plan, сложные вопросы |
| gptoss | gpt-oss-120b/latest | fallback при проблемах с Qwen |
| deepseek | — | альтернатива |

### MCP-инструменты (17)
`token_budget`, `load_artifact`, `list_skills`, `load_skill`,
`lean_check`, `lean_health`, `lean_search_engines/loogle/leansearch/moogle/scilib`,
`web_search`, `web_fetch`, `pdf_download`, `pdf_info`, `pdf_read`, `pdf_search`.

### .goosehints
Короткий системный контекст: кто пользователь, предпочтения по стилю ответов,
активный проект (TrafficLight), указание на локальный knowledge graph.

---

## 3. Локальный Knowledge Graph

### База
- **Папка:** `graph_storage/` (LlamaIndex PropertyGraphStore)
- **Embed-модель:** `BAAI/bge-m3` (multilingual, 1024-dim), на GPU через `shad_env`
- **Содержимое:** 208 text chunks из 9 PDF, с metadata (file_name, page_label)
- **Статьи:**
  1. SLB Report.pdf — LF-DAS fundamentals, лабораторные эксперименты (95 чанков)
  2. sensors-24-03880.pdf — Sensors 2024, LF-DAS отклик двух параллельных трещин (23 чанка)
  3. 2569356.pdf — Peiyao Li et al., SPE Journal 2024, Heart Shape to Fracture Distance (16 чанков)
  4. spe-230644-ms.pdf — SPE 230644 (16 чанков)
  5. urtec-4475249-ms.pdf — URTeC 2024 (16 чанков)
  6. spe-223513-ms.pdf — SPE 223513 (13 чанков)
  7. seg-2022-3750851.pdf — SEG 2022, CNN для DAS classification (12 чанков)
  8. spe-227698-ms.pdf — SPE 227698, HF/LF DAS в glutenite (9 чанков)
  9. arma-2024-0396.pdf — ARMA 2024 (8 чанков)

### Архитектура
- Vector RAG на numpy cosine similarity (SimplePropertyGraphStore не имеет своего vector_query)
- Retrieval: top-5 чанков по эмбеддингам bge-m3
- Синтез ответа: LLM сессии (qwen3.6-35b по умолчанию), промпт включает контекст с цитатами

### Запуск
```cmd
scripts\graph_rag.bat "query на английском"
scripts\graph_rag.bat "query" --retrieve-only --top-k 10
scripts\graph_rag.bat --rebuild        # пересборка при добавлении PDF в articles/
```

Batch-файл использует `shad_env` → bge-m3 на CUDA (ребилд графа ~секунды).

---

## 4. Skills

| Skill | Назначение |
|---|---|
| `knowledge-graph.md` | Поиск по локальной базе 9 PDF с цитатами [file p.N] |
| `trafficlight.md` | Работа с frac-hit prediction проектом в `work/trafficlight/` |
| `literature.md` | Внешний поиск статей (web_search + pdf_download) |
| `latex.md` | LaTeX документы |
| `lean.md` | Формализация теорем в Lean 4 (через scilibai.ru) |
| `markdown.md` | Markdown workflow |
| `python.md` | Python скрипты и анализ данных |
| `debug-loop.md` | Отладка в режиме test-fix-test |

Все skills написаны с frontmatter (`name`, `description`, `triggers`, `combines_with`)
и подхватываются Goose автоматически. Триггер по ключевым словам в сообщении
пользователя или прямой вызов через `load_skill("name")`.

---

## 5. Fork TrafficLight (безопасные эксперименты)

Оригинал проекта: `D:\work\TrafficLight` (production, не трогается).
Копия для AI4Science: `C:\Users\rkorkin\AI4Science\AI4Science\work\trafficlight\`.

**Что скопировано (29 MB, 67 файлов):**
- `deploy/` — live app (shell.py, nn_seq_pipeline, inference, model weights, thresholds)
- `core/` — training code (nn_seq_pipeline, preprocess, windowed_core, hazard_model, split.json)
- `NOTES.md` — краткая выжимка проекта на 1 страницу

**Данные:** 364k записей DAS в `D:\work\TrafficLight\field_data_enriched\` (2.44 GB)
читаются напрямую через переменную `FIELD_DATA_DIR` в `.env` копии, без дублирования.

---

## 6. Примеры запросов

### Пример 1 — обзорный вопрос по корпусу (knowledge-graph)

**Query:** "What does the SLB Report say about low-frequency DAS?"

**Ответ агента (сжато):**
> LF-DAS is a methodology to monitor and characterize multistage hydraulic
> fracturing in horizontal unconventional wells [SLB Report.pdf p.2].
> Deployment inside vs outside the casing changes signal character:
> outside — higher magnitude and precision [SLB Report.pdf p.92].
> Laboratory experiments demonstrate correlation with known fracture geometries
> and separation of two fracture planes [SLB Report.pdf p.74].

Все цитаты валидные, страницы правильные. Задействован vector retrieval →
top-5 чанков → синтез Qwen'ом.

### Пример 2 — метод-специфичный вопрос

**Query:** "What methods are used for DAS strain analysis in the loaded papers?"

**Ответ агента (сжато):**
> - Crosswell far-field strain measurements for fracture azimuth/width/height [2569356.pdf p.7122]
> - Strain rate calculation from strain [SLB Report.pdf p.12]
> - Heart-shaped signal → peak-picking → fracture tip distance [2569356.pdf p.7131]
> - CNN classification of LF-DAS strain rate patterns [seg-2022-3750851.pdf p.5]

Правильно вытаскивает методы из разных источников, не выдумывает.

### Пример 3 — self-reflection по TrafficLight

**Query:** "По TrafficLight: какие методы frac hit detection уже пробовались?"

**Ответ агента (сжато):**
> Читает `work/trafficlight/NOTES.md` и `core/ANALYSIS_SUMMARY.md`:
> - Current production: config C (SeqCNNGRU, c_in=7, hidden=96, physics=on),
>   recall 0.84, FAR 0.14, median red lead = 38 min
> - GBM baseline: recall 0.85 / FAR 0.30 — хуже чем CNN+GRU
> - CNN per-crop: withdrawn из-за future-data leak в per-file нормализации
> - Classical wedge (FWHM collapse): ETA точная в последние 15 min, но
>   standalone detector плохой (FAR 0.81)
> - Not done: wedge-NN fusion (NN владеет RED, wedge даёт minute-level countdown
>   только внутри armed window)

Агент сам находит открытое направление для развития работы в рамках курса.

---

## 7. Что сделано, что нет

### Сделано
- Harness установлен, Yandex API работает, бюджет 3M токенов/день
- Vector RAG по 9 PDF с цитированием (208 чанков × 1024-dim bge-m3)
- Два профильных skill (knowledge-graph, trafficlight) агент сам выбирает по контексту
- Безопасная копия TrafficLight — эксперименты без риска для production
- GPU-ускорение embedding через shad_env (RTX 4000 Ada, 20 GB VRAM)
- `.goosehints` с системным промптом под профиль пользователя

### Не сделано (сознательно)
- **Entity extraction в графе** — пустой (ImplicitPathExtractor даёт только
  SOURCE/NEXT/PREVIOUS, без семантических связей). Полноценный LLM-extractor
  потребует переиндексации и расхода токенов на 208 чанков × каждый.
- **Нет MCP для внутренних SLB-систем** — вся работа на публичных PDF и
  копии проекта. Корпоративные данные намеренно вне AI4Science.
- **bge-m3 не fine-tuned под oil & gas** — multilingual general-purpose модель.
  Доменный adapter поверх embedding мог бы дать +3-5% retrieval accuracy, но
  требует размеченных пар и времени.
- **TrafficLight model не переобучена через агента** — пока только чтение NOTES
  и диагностика. Следующий шаг — передать агенту задачу improve wedge-NN fusion.

### Планы на следующие недели курса
1. Расширить граф: добавить ~15 дополнительных PDF (SPE, URTeC, SEG по
   DAS/fracturing) → ребилд на GPU займёт пару минут
2. Прогнать агента на реальную задачу из TrafficLight: собрать improvement
   proposal по wedge-NN fusion из WORKLOG.md
3. Fine-tune roberta/mpnet на QQP → перенести опыт на доменные пары
   (парафразы терминов из oilfield литературы)
4. Увеличить размер датасетов через полевых специалистов (хотя похоже уже отжал от них все)
5. Доделать существующую модель для генерации синтетических данных основанных на Discontinuous Displacement Model (но с более честной физиков) с добавлением сильного полевого шума. Попробовать помирить их с полевыми данными (уменьшить domain gap), с тем чтобы влить в трейн синтетические данные. Заодно уточнить качество лейблов (точка во времени и по координате где случился фрак хит)
