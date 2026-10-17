---
name: knowledge-graph
description: Локальная база 9 статей по LF-DAS, hydraulic fracturing, fiber optic sensing — vector RAG через bge-m3 (на GPU через shad_env), ответы с цитатами (file p.page)
triggers: knowledge graph, local corpus, loaded papers, SLB Report, Peiyao Li, heart shape, LF-DAS, DAS strain, fiber optic sensing, fracture monitoring, что в наших статьях, что в локальной базе, graph_rag
combines_with: literature (для расширения вовне), markdown (для записи выжимки), python (для построения графиков по извлечённым данным)
---

# Knowledge Graph — локальный vector RAG

## Когда применяется

Применяй **ДО** `web_search` в следующих случаях:

- Вопрос про методы, термины, авторов или результаты из уже загруженной локальной базы (9 PDF: SLB Report, Sensors 2024, Peiyao Li SPE Journal 2024, SPE 230644, URTeC 2024, SPE 223513, SEG 2022, SPE 227698, ARMA 2024)
- Темы: LF-DAS, DAS strain analysis, heart-shaped signal, fracture propagation monitoring, fiber optic sensing inside/outside casing, perforation clusters, cross-well strain
- Пользователь явно ссылается на «наши статьи», «локальную базу», «загруженный corpus»
- Нужен точный факт с цитатой — страница + источник

## База данных

- Папка: `graph_storage/` (LlamaIndex PropertyGraphStore, уже построен)
- Embed-модель: `BAAI/bge-m3` (1024-dim, multilingual, на GPU через shad_env)
- 208 text chunks из 9 PDF, с metadata (file_name, page_label)
- LLM для синтеза: та же что в сессии AI4Science (через Yandex AI Studio)

## Запуск

Все команды — через обёртку `scripts\graph_rag.bat`, которая зовёт `shad_env\python.exe` (torch+CUDA, bge-m3 на GPU).

### 1. Прямой вопрос — один retrieval + синтез

```
shell("scripts\\graph_rag.bat \"<query на английском — лучше релевантность по bge-m3>\"")
```

Возвращает ответ с цитатами вида `[SLB Report.pdf p.12]`. Используй как есть, не перефразируй цитаты.

### 2. Только релевантные чанки, без LLM-синтеза

```
shell("scripts\\graph_rag.bat \"<query>\" --retrieve-only --top-k 10")
```

Применяй когда:
- Хочешь увидеть первоисточник для дальнейшего анализа
- Нужно больше top_k чем 5 (собрать все упоминания термина)
- Планируешь свой синтез по нескольким запросам

### 3. Многошаговый анализ

Сделай 2-3 целевых запроса, сопоставь источники:

```
shell("scripts\\graph_rag.bat \"heart-shaped signal fracture tip distance\" --retrieve-only")
shell("scripts\\graph_rag.bat \"LF-DAS strain rate waterfall plot\" --retrieve-only")
```

Потом **своим рассуждением** свяжи выдачи. Не проси LLM «сравнить» — лучше дай ей оба контекста в одном промпте.

### 4. Если база не отвечает

Если ответ «context does not contain the answer» или top-score < 0.4 — переключайся на `literature` skill (web_search + pdf_download). Явно скажи: «В локальной базе нет, иду наружу».

## Правила

- Запросы формулируй на английском — bge-m3 multilingual, но корпус англоязычный, релевантность выше
- Цитаты не редактируй — формат `[filename.pdf p.N]` валидный
- Не вызывай `pdf_read` на тех же 9 PDF, если ответ уже есть через RAG
- Для deep-dive — сначала `--retrieve-only`, увидишь какие страницы релевантны, потом `pdf_read(path, pages=...)` именно по ним
- Пересборка графа только при добавлении новых PDF: положи их в `articles/`, запусти `scripts\graph_rag.bat --rebuild` (на GPU — секунды)

## Debug

Если RAG падает:

```
shell("C:\\Users\\rkorkin\\rkorkin\\shad_env\\Scripts\\python.exe src\\graph_debug.py")
```

Показывает что retrieval возвращает без LLM-синтеза.

## Что НЕ делает

- Не ищет во внешних источниках (arxiv, SPE, Google Scholar) — для этого `literature`
- Не умеет semantic reasoning по entities (139 entity-нод в графе пустые, по сути vector RAG)
- Не извлекает таблицы и формулы — только текстовые чанки
