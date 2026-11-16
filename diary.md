# diary.md — как мы пришли к текущему состоянию

Хронологический журнал решений, тупиков и причин. Формат записи:
`## YYYY-MM-DD (неделя N): короткий заголовок` → 3–8 строк.

---

## 2026-11-16 (неделя 10): каркас проекта статьи

- Построен каркас по шаблону курса: `AGENTS.md / STATUS.md / diary.md / ONTOLOGY.yaml /
  GLOSSARY.md / WRITING.md / openspec/ / checks/`.
- Переименовано `verify/` → `checks/` для консистентности с заданием и с ДЗ-2.
- Из `hw2` подтянуты: `METHOD.md`, property-тесты, валидатор, `experiment.py`, `CHECKS.md`.
- Онтология заполнена реальными сущностями проекта (wedge, LF-DAS, strain-rate, hit,
  causal preprocess, soft-labels, SeqCNNGRU, метрики), не заглушками — это то, за что
  в ДЗ-1 били.
- `.gitignore` расширен (python/pytest/hypothesis кэши).

## 2026-11-14 (неделя 9): ДЗ-2 заморожена

- Сдана ДЗ-2 по детальной рубрике: `METHOD.md` раньше результатов, K4-эксперимент
  (5 seed, парная статистика, Холм), пойманная ошибка «до/после», дерево подцелей и
  журнал попыток, Lean 0 sorry, метка `hw2` в origin.
- Ветка `hw2` заморожена, тег `hw2` указывает на ту же цепочку.
- Форма: `forms.yandex.ru/cloud/6adc8e68c45c07d33226f1ae` — отправлена ссылка
  `github.com/rvkorkin/ai4science/tree/hw2`.

## 2026-11-10 (неделя 8): литобзор + Lean

- `LITREVIEW.md`: 9 источников, DOI сверены вручную (в ранней версии гусь выдумал
  `10.2118/123456-PA` для Li 2024 — заменено на реальный `10.2118/223631-PA`,
  SPE J. 29(12):7122–7133).
- Lean 4: теорема `P_causal` (каузальность всей цепочки `bdiff → ewma → decim`)
  доказана, 0 `sorry`. Base case EWMA синхронизирован с numpy (`s = dr.copy()` ↔
  `ewma a x 0 = x 0`).

## 2026-10-31 (неделя 6–7): основной верификатор

- Property-тесты каузальности на Hypothesis (`decimate_causal`, `strain_rate_causal`,
  `strain_rate_from_raw`) + валидатор данных по 626 enriched-npz: 0 hard-fail.
- Поймана и исправлена `label_T_mismatch` в 9 CrownQuest-записях.

## 2026-10-16 (неделя 1–5): среда

- Среда ДЗ-1: vector RAG на bge-m3 по 9 PDF, skills, безопасная копия проекта.
- Отложено: entity-extraction / ontology — догоняем сейчас (неделя 10).
