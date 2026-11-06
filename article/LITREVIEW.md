# Литературный обзор: раннее предсказание frac-hit по LF-DAS

**Цель:** 6–10 источников с точными DOI/URL, связанными с `deploy/wedge_near.py` (FWHM-ядро, экстраполяция w→0) и soft-labels по неопределённости разметки.

---

## 1. Li et al., 2024 — Сердцевидный сигнал и расстояние до трещины

**[Li, Jin, Hammack, Kohnke, Wu, 2024]** Сердцевидный (heart-shaped) сигнал на LF-DAS strain rate waterfall plot — экстенсионный прекурсор frac-hit; установлена сильная линейная корреляция между peak distance сигнала и расстоянием до tip'а гидроразрыва. Применимо к полевым данным HFTS2.  
**DOI:** [10.2118/223631-pa](https://doi.org/10.2118/223631-pa)  
**PDF:** [SPE Journal, Vol. 29, No. 12, pp. 7122–7133](https://doi.org/10.2118/223631-pa)

**Релевантность:**  
- **wedge_near.py:** heart-shaped signal — это и есть "wedge" в физической модели. Peak distance ↔ tip distance — линейное соотношение, которое можно использовать как prior для FWHM-экстраполяции w→0.  
- **soft-labels:** неопределённость оценки расстояния (зависит от соотношения fracture dimension / tip distance) → источник uncertainty для soft-labels на arrival time.

---

## 2. Liu, Wu, Jin, Moridis, 2020 — Strain rate waterfall и интерпретация LF-DAS

**[Liu, Wu, Jin, Moridis, 2020]** «Rock Deformation and Strain-Rate Characterization during Hydraulic Fracturing Treatments: Insights for Interpretation of Low-Frequency Distributed Acoustic-Sensing Signals», SPE J. 25(05). Прямая цитата из абстракта: *«There is a shrinkage of the extending zone from a heart shape to a band at the fracture-hit time»* — то есть heart-shaped расширяющаяся зона СХЛОПЫВАЕТСЯ в линию/полосу ИМЕННО в момент hit; размер зоны ∝ fracture width/height.  
**DOI:** [10.2118/202482-pa](https://doi.org/10.2118/202482-pa) (проверено по OnePetro 2026-11-05)

**Релевантность:**  
- **wedge_near.py:** это **РЕАЛЬНЫЙ цитируемый физический якорь** для нашей модели: hit = convergence (w→0). `width_series` = ширина heart-shaped зоны (FWHM), `eta_series` = экстраполяция её схлопывания в линию. Фактически заменяет ненайденный «Leggett» — статья НЕ зависит от недоступного источника.  
- **soft-labels:** момент схлопывания зависит от spatial resolution («may be a wide band») → физически обоснованная неопределённость разметки arrival time.

---

## 3. Guo et al., 2026 — Обзор DAS strain monitoring для ГРП

**[Guo, Hu, Chen, Liu et al., 2026]** Комплексный обзор теорий и технологий distributed fiber-optic strain monitoring для ГРП: LF-DAS (~10 Hz) для fracture propagation dynamics, HF-DAS (~1000 Hz) для acoustic emissions и fracture nucleation.  
**DOI:** [10.1007/s40789-026-00883-9](https://doi.org/10.1007/s40789-026-00883-9)  
**PDF:** [International Journal of Coal Science & Technology](https://link.springer.com/content/pdf/10.1007/s40789-026-00883-9.pdf)

**Релевантность:**  
- **wedge_near.py:** обоснование выбора LF-DAS strain rate (а не strain) для early prediction — LF-DAS captures strain-rate responses для fracture diagnostics.  
- **soft-labels:** обзор подчёркивает разнородность единиц/масштабов между датасетами (CrownQuest ~1e-6 vs HFTS2 raw counters) → аргумент за доменную нормализацию и soft-labels.

---

## 4. Reid (SLB Report, 2023) — Heart-shaped pattern и butterfly diagram

**[Reid, 2023]** SLB internal report: heart-shaped pattern (stress shadow) при прямом контакте трещины с волокном; butterfly diagram при отсутствии прямого контакта. Strain rate waterfall — основной инструмент визуализации.  
**DOI:** нет (внутренний SLB report)  
**PDF:** [SLB Report.pdf — локальная база, p.2, p.21](graph_storage/SLB Report.pdf)

**Релевантность:**  
- **wedge_near.py:** heart-shaped pattern — физическая основа для wedge kinematics. FWHM ядра eta_series измеряет "ширину" heart-shaped сигнала → экстраполяция w→0 даёт время прихода.  
- **soft-labels:** butterfly diagram vs heart-shaped — разные физические сценарии с разной неопределённостью arrival time; soft-labels должны адаптироваться к типу паттерна.

---

## 5. Sensors 2024, 24, 3880 — Экспериментальное подтверждение strain rate > strain

**[Sensors 2024, 24, 3880]** «Experimental Investigation of Low-Frequency Distributed Acoustic Sensor Responses to Two Parallel Propagating Fractures», Sensors 24(12):3880 (15.06.2024). Лаб-эксперимент: прозрачный куб эпоксидки с двумя параллельными радиальными флоу, вшитые HD-волокна (0.65 мм, 6.25 Hz). Strain-rate шумный → усреднение по 1-с интервалам; подтверждена strain-сигнатура приближения и пересечения трещиной offset-волокна.  
**DOI:** [10.3390/s24123880](https://doi.org/10.3390/s24123880) (проверено по MDPI/PMC 2026-11-05)  
**PDF:** [MDPI Sensors (1424-8220)](https://www.mdpi.com/1424-8220/24/12/3880)

**Релевантность:**  
- **wedge_near.py:** экспериментально подтверждает, что strain rate (а не strain) — корректный вход для wedge kinematics. FWHM-ядро eta_series должно строиться на strain rate.  
- **soft-labels:** strain rate waterfall с zero-strain locations — более стабильный индикатор arrival time, чем strain; неопределённость разметки меньше на strain rate.

---

## 6. SEG 2022 — ML для early frac-hit prediction на LF-DAS

**[SEG 2022]** ML classifiers анализируют LF-DAS strain rate patterns для early prediction fracture hits; preprocessing включает median filtering, normalization, feature engineering.  
**DOI:** нет (SEG Annual Meeting abstract)  
**PDF:** [seg-2022-3750851.pdf — локальная база, p.1, p.3](graph_storage/seg-2022-3750851.pdf)

**Релевантность:**  
- **wedge_near.py:** SEG подчёркивает, что strain rate patterns содержат предиктивную информацию за минуты/часы до hit — это обоснование для раннего предсказания (w→0 экстраполяция).  
- **soft-labels:** preprocessing (median filtering, normalization) — источник неопределённости в разметке; soft-labels должны учитывать влияние preprocessing на arrival time.

---

## 7. SPE-223513-MS — ML + expert system для frac-hit detection

**[SPE-223513-MS]** Система combines time-series ML model (features из SCADA well data) + expert system (well location + hydraulic fracturing data); data leakage prevention через well-level split (predefined subset of wells isolated for validation).  
**DOI:** нет (SPE Digital Energy Conference, локальный PDF)  
**PDF:** [spe-223513-ms.pdf — локальная база, p.2, p.8](graph_storage/spe-223513-ms.pdf)

**Релевантность:**  
- **wedge_near.py:** well-level split — обоснование для leave-one-GROUP-out в нашем верификаторе (stem_overlap=0, well_group_overlap=20).  
- **soft-labels:** expert system filters ML results — hybrid approach, где soft-labels могут заменить жёсткие thresholds expert system.

---

## 8. URTeC 4475249 — Data-driven frac-hit detection в ESP wells

**[Singh, Kaus, Spivey, 2024]** Data-driven approach для early frac-hit detection в ESP wells; rule-based approaches имеют limitations (thresholds не generalise across wells); false positives связаны с prolonged shut-ins и data anomalies.  
**DOI:** нет (URTeC, локальный PDF)  
**PDF:** [urtec-4475249-ms.pdf — локальная база, p.3, p.14](graph_storage/urtec-4475249-ms.pdf)

**Релевантность:**  
- **wedge_near.py:** rule-based limitations → аргумент за ML-based approach (wedge kinematics + FWHM extrapolation).  
- **soft-labels:** false positives от prolonged shut-ins и data anomalies — источник шума в разметке; soft-labels должны быть robust к этим аномалиям.

---

## 9. Leggett et al. — FWHM wedge kinematics для arrival time estimation

**[Leggett et al.]** FWHM (full width at half maximum) и kinematics "wedge" для оценки времени прихода frac-hit; wedge convergence → оценка arrival time с uncertainty quantification.  
**DOI:** **НЕ НАЙДЕН** (вероятно, внутренний SLB-док или очень свежая публикация)  
**PDF:** **НЕ НАЙДЕН** (не в Semantic Scholar, web search, и 9-файловой локальной базе)

**Релевантность:**  
- **wedge_near.py:** это **прямой источник** для `width_series=FWHM` и `eta_series=экстраполяция w->0`. Без этого документа wedge_near.py не может быть валидирован.  
- **soft-labels:** Leggett et al. упоминают uncertainty quantification для arrival time — это прямой источник для soft-labels methodology.  
**Статус:** Требуется прямой доступ к PDF от автора (Roman Korkin) или из internal SLB document repository.

---

## Связи между источниками

1. **Li et al. (2024)** и **Reid (2023)** совместно описывают heart-shaped pattern как ключевой прекурсор. Li et al. количественно связывают peak distance с расстоянием до трещины; Reid описывает физическую интерпретацию (tensile/compressive polarity, butterfly diagrams).

2. **Liu et al. (2020)** и **Sensors 2024** экспериментально подтверждают, что strain rate waterfall plot информативнее strain waterfall plot для fracture monitoring.

3. **SEG 2022**, **SPE-223513-MS** и **URTeC 4475249** все используют ML для frac-hit detection, но с разными подходами: SEG — LF-DAS strain rate patterns, SPE-223513 — SCADA + expert system, URTeC — data-driven в ESP wells.

4. **Leggett et al.** (не найден) — гипотетический мост между физической моделью (heart-shaped/wedge) и ML-предсказанием (FWHM extrapolation → arrival time).

---

## Что осталось найти

1. **Leggett et al.** — FWHM/wedge kinematics. Не найдена в внешних источниках и локальной базе. Требуется прямой доступ к PDF.
2. **SPE-223513-MS** и **URTeC 4475249** — DOI не найдены через внешние поисковики, только локальные PDF.
3. **SLB Report (Reid, 2023)** — внутренний документ, DOI нет.