# Журнал проверок (verification log)

Каждая запись: **что проверяли → команда → наблюдаемый результат → вердикт → дата**.
Цель — чтобы любой прогон верификатора был воспроизводим и прослеживаем. Прогоны
выполнял агент (Goose / Cursor), результаты проверены человеком (R. Korkin).

Окружение: Windows, `PYEXE=C:\Users\rkorkin\rkorkin\shad_env\Scripts\python.exe`
(torch 2.6+cu124, NVIDIA RTX 4000 Ada). Lean — через облачный checker SciLib
(`scilibai.ru/grag`, режим GRC21).

---

## 1. Property-тесты причинности + юнит-тесты валидатора (Hypothesis + pytest)

- **Команда (standalone, без полевых данных):**
  ```cmd
  cd checks & %PYEXE% -m pytest -q
  ```
- **Результат:** `14 passed, 6 skipped in ~2.0s`. Skipped — leakage-тесты, которым
  нужны полевые данные (вне репо).
- **Команда (с данными проекта):**
  ```cmd
  set TL_CORE=<project>\work\trafficlight\core & %PYEXE% -m pytest -q
  ```
- **Результат:** `20 passed in ~2.3s` (добавляются 6 leakage-тестов).
- **Вердикт:** ✅ причинность предобработки (`decimate_causal`, `strain_rate_causal`,
  `strain_rate_from_raw`) и дизъюнктность split подтверждены на сотнях случайных
  примеров (max_examples 60–250). Тесты не вырождены (возмущение входа обязано
  менять выход).
- **Дата:** 2026-11-05 (ре-прогон агентом Goose; набор расширен с 5→7 тестов причинности).

---

## 2. Валидатор данных по всему корпусу (численный эксперимент)

- **Команда:**
  ```cmd
  cd checks & %PYEXE% run_experiment.py
  ```
- **Результат:** 626 / 626 файлов прошли без ошибок (0 hard-fail).
  Предупреждения: `strain_rate_extreme` 327, `dz_out_of_range` 48,
  `no_label_entry` 12, `label_D_mismatch` 1, `label_T_mismatch` **0**.
- **Аудит split:** train/val 430/40, пересечение стемов train/val = 0,
  стемов без лейбла = 0, `tpk` вне `[0,T)` = 0, пересечение well-групп = 20 (ожидаемо).
- **Артефакты:** `results/validation_report.csv`, `results/validation_summary.json`,
  `results/experiment_summary.json`.
- **Вердикт:** ✅ данные схемно и физически валидны; утечек по стемам нет.
- **Дата:** 2026-11-05.

---

## 3. Исправление лейблов CrownQuest (label_T_mismatch 9 → 0)

- **Что:** валидатор поймал 9 записей CrownQuest, где `T` хранился в сырых 10-с
  сэмплах, а npz уже децимирован до минут.
- **Команда:**
  ```cmd
  cd checks & %PYEXE% fix_labels.py
  ```
- **Результат:** 9 записей нормализованы к минутной сетке → `label_T_mismatch` = 0.
  Остаётся `label_D_mismatch` = 1 (отмечена для ручной проверки).
- **Вердикт:** ✅ временные лейблы согласованы с сеткой npz.
- **Дата:** 2026-11-05.

---

## 4. Формальная верификация каузальности (Lean 4)

- **Команда:** MCP `lean_check` на `checks/CausalPreprocess.lean` (checker SciLib GRC21).
- **Результат:** `success: true`; 0 `sorry` / 0 `admit` / 0 `axiom`.
  Доказаны 4 теоремы: `bdiff_causal`, `ewma_causal`, `P_causal`, `P_zero`.
- **Проверка на заглушки:** `grep -i "sorry|admit|axiom|native_decide"` → совпадений нет.
- **Вердикт:** ✅ каузальность предобработки доказана формально для всех входов (∀x),
  а не только на сэмплах. Базовый случай EWMA совпадает с numpy (`ewma a x 0 = x 0`).
- **Дата:** 2026-11-06.

---

## 5. Численный эксперимент: soft-labels vs baseline (A/B)

- **Команда:**
  ```cmd
  set NN_SOFT_LABELS=& %PYEXE% nn_seq_pipeline.py                         REM baseline
  set NN_SOFT_LABELS=..\verify\results\soft_labels_full.csv& %PYEXE% nn_seq_pipeline.py  REM soft
  cd ..\verify & %PYEXE% analyze_ab.py & %PYEXE% compute_detection.py
  ```
- **Условия:** GPU RTX 4000 Ada, 25 эпох, seed 0, 430 train / 40 val.
- **Результат (near-field прихода, val hit n=15):** MAE vs manual 21.8→21.0;
  median vs consensus 18.9→**12.2**. **Детекция (per-file, FAR≤0.10):**
  AUC 0.881→**0.915**, recall 0→**0.733**, precision 0.846.
- **Артефакты:** `results/ab_time_eval.json`, `results/soft_labels_full.csv`,
  `results/consensus_uncertainty.csv`, `results/stg28_wedge.png`.
- **Вердикт:** ✅ soft-labels улучшают детекцию и near-field точность.
  ⚠ Ограничения: 1 seed, маленький val, split не строго leave-pair-out (см. REPORT §6, §8).
- **Дата:** 2026-10-17 (обучение) / сверка 2026-11-05.

---

## 6. Численный эксперимент K4: цена look-ahead-утечки

- **Команда:**
  ```cmd
  cd checks & %PYEXE% experiment.py
  ```
- **Условия:** синтетика, 5 seed (0–4), парное сравнение causal vs leaky нормировки,
  правило принятия предзадано в `METHOD.md §5`.
- **Результат:** AUC causal 0.479 → leaky 0.927, ΔAUC +0.448, 95%CI(t) [+0.376, +0.519],
  d_z 7.76, Холм p 0.0001 (AUC) / 0.0001 (AP). Правило принятия выполнено.
- **Артефакты:** `checks/results/experiment_metrics.json`, `experiment_runs.csv`.
- **Вердикт:** ✅ look-ahead-утечка материально завышает метрику → верификатор каузальности
  обоснован количественно.
- **Дата:** 2026-11-14.

---

## 7. Пойманная ошибка (внесённый баг) — «до/после»

- **Что:** центральная разность `(s[t+1]−s[t−1])/2dt` в `strain_rate_causal` (использует будущее).
- **Команда:** ветка `hw2-injected-bug`, затем `cd checks & %PYEXE% -m pytest -q`.
- **Результат «до»:** `2 failed, 12 passed` (`checks/results/caught_error_before.txt`);
  поймал `test_strain_rate_causal_no_future_leak`. **«После»** (ветка `hw2`):
  `14 passed, 6 skipped` (`checks/results/caught_error_after.txt`).
- **Вердикт:** ✅ property-тест причинности ловит look-ahead.
- **Дата:** 2026-11-14.

---

## Как перепроверить всё разом

```cmd
set PYEXE=C:\Users\rkorkin\rkorkin\shad_env\Scripts\python.exe
cd checks
%PYEXE% -m pytest -q            REM п.1
%PYEXE% run_experiment.py       REM п.2 (нужен FIELD_DATA_DIR в .env)
```
Lean (п.4) проверяется через MCP `lean_check` в среде AI4Science; A/B (п.5) требует
GPU и данных проекта. Результаты зафиксированы под git-меткой `hw2`.
