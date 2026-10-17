# Отчёт по ДЗ1 — «Среда и контекст»

**Курс:** ШАД «ИИ-ассистенты для исследователя. AI4Science»  
**Автор:** Roman Korkin, PhD (теоретическая физика), Principal Physicist, SLB  
**Дата:** 2026-10-17

---

## 1. Задача и домен

**Основная задача курса:** построить модель раннего предсказания frac hits (приближающийся прорыв трещины ГРП в мониторинговую скважину) по данным LF-DAS strain rate.

**Проект TrafficLight:** production live-мониторинг в PyQt5-приложении (GRE/YEL/RED статусы), текущая модель — SeqCNNGRU config C.

**Метрики deployed модели (fold2 seed2, leave-one-GROUP-out CV):**
- Window-level AUC = 0.960 (pooled OOF по 9 моделям: 3 fold × 3 seed)
- File-level recall @ FAR ≤ 0.30 = 0.935, median lead = 38 min
- После threshold calibration: recall = 0.84 / FAR = 0.14 / median red lead = 38 min

**Известная проблема:** AUC 0.960 — data leak в оценке deployed модели (видела файлы из других fold'ов при обучении). Fix: ensemble из 9 моделей или retrain на всех данных + hold-out set. Задокументировано в WORKLOG.md.

**Данные:** 364k записей DAS (`field_data_enriched/`, 2.44 GB).

---

## 2. Архитектура среды

### Harness
Goose 1.50 поверх Yandex AI Studio (OpenAI-совместимый endpoint). Запуск: `bin\ai4science.bat`.

### Python environments

| Env | Python | Назначение |
|---|---|---|
| `.venv` | 3.12.7 | Harness, llama-index, Yandex API |
| `shad_env` | 3.13.15 | torch 2.6+cu124, PyQt5, bge-m3 на GPU |

Модели (Yandex AI Studio): qwen3.6-35b (default), qwen3-235b-a22b (plan), gpt-oss-120b (fallback). Бюджет: 3M токенов/день.

### Vector-RAG по корпусу PDF

- **Хранилище:** `graph_storage/` (LlamaIndex PropertyGraphStore)
- **Embedding:** `BAAI/bge-m3` (1024-dim, multilingual), CUDA через `shad_env`
- **Корпус:** 9 PDF в `articles/`, 208 чанков в RAG
- **Retrieval:** top-5 по cosine similarity на numpy
- **Синтез:** LLM сессии (qwen3.6-35b), temperature=0.2

**Статьи в RAG (208 чанков):**

| PDF | Чанков |
|---|---|
| SLB Report.pdf | 95 |
| sensors-24-03880.pdf | 23 |
| 2569356.pdf | 16 |
| spe-230644-ms.pdf | 16 |
| urtec-4475249-ms.pdf | 16 |
| spe-227698-ms.pdf | 9 |
| arma-2024-0396.pdf | 8 |
| seg-2022-3750851.pdf | 12 |
| spe-223513-ms.pdf | 13 |
| **Total** | **208** |

### Skills

8 файлов с frontmatter (`name`, `description`, `triggers`, `combines_with`):
- `knowledge-graph.md` — поиск по локальной базе 9 PDF с цитатами
- `trafficlight.md` — работа с frac-hit prediction проектом
- `literature.md` — внешний поиск статей
- `latex.md`, `lean.md`, `markdown.md`, `python.md`, `debug-loop.md`

### .goosehints

Системный промпт: профиль пользователя, предпочтения по стилю ответов (русский для общения, английский для кода; концизно с объяснением «почему»; код без комментариев; не галлюцинировать на цифрах), активный проект (TrafficLight), указание на локальный knowledge graph.

---

## 3. Ключевые проектные решения и ПОЧЕМУ

### bge-m3 на GPU (shad_env)
Embedding 208 чанков на CPU занял бы минуты. На RTX 4000 Ada (CUDA) — секунды. Ребилд графа при добавлении PDF занимает пару минут.

### Vector-RAG вместо LLM entity-extraction
`ImplicitPathExtractor` из LlamaIndex даёт только `SOURCE/NEXT/PREVIOUS` — без семантических связей. 139 entity-нод в графе пустые. Полноценный LLM-extractor потребовал бы переиндексации 208 чанков и большого расхода токенов. Vector-RAG с bge-m3 работает сразу и даёт релевантные чанки по cosine similarity.

### Fork TrafficLight ради безопасности production
Оригинал: `D:\work\TrafficLight` (production, не трогается). Копия: `work/trafficlight/` (29 MB, 67 файлов). Данные (`field_data_enriched/`) читаются напрямую через `FIELD_DATA_DIR` в `.env` — без дублирования 2.44 GB.

---

## 4. Что работает и что сознательно не сделано

### Что работает
- Harness установлен, Yandex API работает, бюджет 3M токенов/день
- Vector RAG по 9 PDF с цитированием (208 чанков × 1024-dim bge-m3)
- 8 skills с frontmatter, агент сам выбирает по контексту
- Безопасная копия TrafficLight — эксперименты без риска для production
- GPU-ускорение embedding через shad_env (RTX 4000 Ada, 20 GB VRAM)
- `.goosehints` с системным промптом под профиль пользователя

### Что сознательно не сделано (см. раздел 7 AGENT.md)
- **Entity extraction в графе** — пустой (ImplicitPathExtractor даёт только SOURCE/NEXT/PREVIOUS). Полноценный LLM-extractor потребует переиндексации и расхода токенов на 208 чанков.
- **Нет MCP для внутренних SLB-систем** — вся работа на публичных PDF и копии проекта. Корпоративные данные вне AI4Science.
- **bge-m3 не fine-tuned под oil & gas** — multilingual general-purpose модель. Доменный adapter поверх embedding мог бы дать +3-5% retrieval accuracy, но требует размеченных пар и времени.
- **TrafficLight model не переобучена через агента** — пока только чтение NOTES и диагностика. Следующий шаг — передать агенту задачу improve wedge-NN fusion.

---

## 5. Как воспроизвести

### Запуск RAG

```cmd
REM Retrieval + LLM synthesis
scripts\graph_rag.bat "What is LF-DAS and how is it used for hydraulic fracturing monitoring"

REM Только чанки, без синтеза
scripts\graph_rag.bat "heart-shaped signal fracture tip distance" --retrieve-only --top-k 10

REM Пересборка графа (при добавлении новых PDF в articles/)
scripts\graph_rag.bat --rebuild
```

Batch-файл использует `shad_env` → bge-m3 на CUDA.

### Запуск debug

```cmd
C:\Users\rkorkin\rkorkin\shad_env\Scripts\python.exe src\graph_debug.py
```

Показывает что retrieval возвращает без LLM-синтеза.

### Запуск harness

```cmd
bin\ai4science.bat
/help
/model <алиас>
/plan
/status
```

---

**Приложение:** `AGENT.md` — полный файл документации среды (прилагается отдельно).