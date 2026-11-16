# -*- coding: utf-8 -*-
"""Численный эксперимент ДЗ-2: цена look-ahead-утечки в предобработке.

ТЕЗИС (проверяемое утверждение).
    Нормировка окна по статистикам ВСЕЙ последовательности (mean/std, включая
    будущие сэмплы) — это look-ahead-утечка: она завышает метрику раннего
    обнаружения frac-hit относительно КАУЗАЛЬНОЙ нормировки (только прошлое),
    хотя в пред-онсетном окне физически нет информации о будущем хите.

    Это ровно тот класс ошибки, который ловят property-тесты каузальности
    (checks/test_causality.py) и доказывает Lean-теорема P_causal: выход в
    момент i зависит только от входа до i. Эксперимент измеряет ЦЕНУ нарушения
    этого инварианта в баллах AUC/AP.

ПЛАН (шаги глубины с условиями перехода):
    smoke : 2 seed, N=160  — быстрый прогон, условие перехода: оба AUC конечны,
                              причинный ~0.5, leaky > причинного → печать PASSED.
    full  : 5 seed, N=400  — основной прогон с повторами и интервалами.

ПРАВИЛО ПРИНЯТИЯ (записано в METHOD.md ДО первого прогона):
    Leaky завышает AUC материально, если ОДНОВРЕМЕННО:
      (1) парная средняя разность Δ = AUC_leaky − AUC_causal по ≥5 seed
          имеет 95%-интервал (парный t), НЕ содержащий 0;
      (2) |Δ| превышает практический порог δ = 0.05 (полбалла детекции);
      (3) парный t-критерий значим после поправки Холма по {AUC, AP}.
    Сравнение ПАРНОЕ (один и тот же seed → одни и те же данные для обоих
    конвейеров), поэтому сравниваются варианты, а не реализации шума.

ВОСПРОИЗВОДИМОСТЬ: всё детерминировано от seed; результаты в
    checks/results/experiment_metrics.json и checks/results/experiment_runs.csv.

Запуск:
    python experiment.py          # full (5 seed)
    python experiment.py smoke     # быстрый прогон (2 seed)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

RESULTS = Path(__file__).resolve().parent / "results"

# практический порог «значимо в баллах детекции» (предзадан в METHOD.md)
DELTA = 0.05
SEEDS_FULL = [0, 1, 2, 3, 4]
SEEDS_SMOKE = [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
#  Синтетика: DAS-подобные последовательности strain-rate
# ─────────────────────────────────────────────────────────────────────────────
def make_dataset(seed: int, n: int, T: int = 120):
    """n последовательностей длины T. Половина — «hit» (медленный рост strain-rate
    после онсета t0), половина — «no-hit» (только шум). Возвращает сырой strain-rate
    (n, T) и метки (n,). Онсет t0 ∈ [40, 80] — ПОЗЖЕ раннего окна [5, 20]."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1
    rng.shuffle(y)

    X = np.zeros((n, T), dtype=np.float64)
    t = np.arange(T)
    for i in range(n):
        noise_sd = rng.uniform(0.7, 1.8)             # свой уровень шума у скважины
        base = rng.normal(0.0, noise_sd, size=T)      # фоновый шум
        drift = rng.normal(0.0, 0.3)                  # межскважинный сдвиг базлайна
        X[i] = base + drift
        if y[i] == 1:
            t0 = rng.integers(40, 81)                 # онсет хита (после раннего окна)
            amp = rng.uniform(1.2, 2.6)               # амплитуда прихода
            ramp = amp * np.clip((t - t0) / 25.0, 0.0, 1.0)  # медленный рост
            X[i] += ramp
    return X, y


def early_features(X: np.ndarray, mode: str, win=(5, 20)):
    """Фичи из РАННЕГО окна [win] под двумя режимами нормировки.

    causal : нормировка статистиками ТОЛЬКО прошлого (расширяющееся mean/std
             до правого края окна) — реальный онлайн-режим, без утечки.
    leaky  : нормировка mean/std ВСЕЙ последовательности (включая будущее после
             онсета) — классическая look-ahead-утечка per-file нормализации.
    """
    n, T = X.shape
    a, b = win
    feats = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        seq = X[i]
        if mode == "leaky":
            mu, sd = seq.mean(), seq.std() + 1e-9           # ВСЯ последовательность
        elif mode == "causal":
            past = seq[: b + 1]                             # только [0 .. правый край окна]
            mu, sd = past.mean(), past.std() + 1e-9
        else:
            raise ValueError(mode)
        z = (seq[a : b + 1] - mu) / sd
        feats[i, 0] = z.mean()
        feats[i, 1] = z.max()
    return feats


def one_run(seed: int, n: int):
    """Один seed: строим данные, обучаем логрег на causal- и на leaky-фичах,
    возвращаем AUC и AP на одном и том же тест-сплите для обоих конвейеров."""
    X, y = make_dataset(seed, n)
    rng = np.random.default_rng(1000 + seed)
    idx = rng.permutation(n)
    ntr = int(0.7 * n)
    tr, te = idx[:ntr], idx[ntr:]

    out = {}
    for mode in ("causal", "leaky"):
        F = early_features(X, mode)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(F[tr], y[tr])
        p = clf.predict_proba(F[te])[:, 1]
        out[f"auc_{mode}"] = float(roc_auc_score(y[te], p))
        out[f"ap_{mode}"] = float(average_precision_score(y[te], p))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Статистика повторов
# ─────────────────────────────────────────────────────────────────────────────
def paired_stats(a: np.ndarray, b: np.ndarray):
    """Парная статистика Δ = b − a: среднее, sd, 95% t-интервал, p (парный t),
    размер эффекта Cohen d_z, парный бутстреп-интервал."""
    d = b - a
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tcrit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    ci = (mean - tcrit * se, mean + tcrit * se) if n > 1 else (float("nan"),) * 2
    tstat, p = stats.ttest_rel(b, a) if n > 1 else (float("nan"), float("nan"))
    dz = mean / sd if sd > 0 else float("inf")
    rng = np.random.default_rng(12345)
    boot = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(10000)])
    bci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return {
        "mean": mean, "sd": sd, "ci95_t": [float(ci[0]), float(ci[1])],
        "p_paired_t": float(p), "cohen_dz": float(dz), "ci95_boot": list(bci),
        "a_mean": float(a.mean()), "b_mean": float(b.mean()),
    }


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Поправка Холма (step-down) на словарь {имя: p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def run(stage: str = "full"):
    seeds = SEEDS_SMOKE if stage == "smoke" else SEEDS_FULL
    n = 160 if stage == "smoke" else 400
    runs = [dict(seed=s, **one_run(s, n)) for s in seeds]

    auc_c = np.array([r["auc_causal"] for r in runs])
    auc_l = np.array([r["auc_leaky"] for r in runs])
    ap_c = np.array([r["ap_causal"] for r in runs])
    ap_l = np.array([r["ap_leaky"] for r in runs])

    st_auc = paired_stats(auc_c, auc_l)
    st_ap = paired_stats(ap_c, ap_l)
    holm_p = holm({"AUC": st_auc["p_paired_t"], "AP": st_ap["p_paired_t"]})

    # правило принятия (только для AUC — первичная метрика)
    ci_excl0 = (st_auc["ci95_t"][0] > 0) or (st_auc["ci95_t"][1] < 0)
    practical = abs(st_auc["mean"]) > DELTA
    sig = holm_p["AUC"] < 0.05
    verdict = bool(ci_excl0 and practical and sig)

    summary = {
        "stage": stage, "seeds": seeds, "n_per_seed": n,
        "delta_threshold": DELTA,
        "baseline": "causal (без утечки)", "variant": "leaky (look-ahead)",
        "AUC": st_auc, "AP": st_ap,
        "holm_adjusted_p": holm_p,
        "acceptance": {
            "ci95_excludes_0": ci_excl0,
            "exceeds_practical_delta": practical,
            "holm_significant": sig,
            "VERDICT_leak_inflates": verdict,
        },
        "runs": runs,
    }

    # печать
    print(f"=== эксперимент: цена look-ahead-утечки ({stage}, seeds={seeds}) ===")
    print(f"{'seed':>4} {'AUC_causal':>11} {'AUC_leaky':>10} {'ΔAUC':>8}")
    for r in runs:
        print(f"{r['seed']:>4} {r['auc_causal']:>11.3f} {r['auc_leaky']:>10.3f} "
              f"{r['auc_leaky'] - r['auc_causal']:>8.3f}")
    print(f"\nAUC: causal {st_auc['a_mean']:.3f} → leaky {st_auc['b_mean']:.3f} | "
          f"ΔAUC = {st_auc['mean']:+.3f}  95%CI(t) [{st_auc['ci95_t'][0]:+.3f}, "
          f"{st_auc['ci95_t'][1]:+.3f}]  d_z={st_auc['cohen_dz']:.2f}  "
          f"p_holm={holm_p['AUC']:.4f}")
    print(f"AP : causal {st_ap['a_mean']:.3f} → leaky {st_ap['b_mean']:.3f} | "
          f"ΔAP  = {st_ap['mean']:+.3f}  p_holm={holm_p['AP']:.4f}")
    print(f"\nправило принятия: CI≠0 {ci_excl0} · |Δ|>δ({DELTA}) {practical} · "
          f"Холм<0.05 {sig}")

    if stage == "smoke":
        ok = np.isfinite(auc_c).all() and np.isfinite(auc_l).all() and \
            (st_auc["b_mean"] > st_auc["a_mean"])
        print("ПРОЙДЕН (smoke)" if ok else "ПРОВАЛ (smoke)")
        if not ok:
            sys.exit(1)
        return summary

    # full: пишем артефакты
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "experiment_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS / "experiment_runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "auc_causal", "auc_leaky", "d_auc",
                    "ap_causal", "ap_leaky", "d_ap"])
        for r in runs:
            w.writerow([r["seed"], f"{r['auc_causal']:.4f}", f"{r['auc_leaky']:.4f}",
                        f"{r['auc_leaky'] - r['auc_causal']:.4f}",
                        f"{r['ap_causal']:.4f}", f"{r['ap_leaky']:.4f}",
                        f"{r['ap_leaky'] - r['ap_causal']:.4f}"])
    print(f"\nВЕРДИКТ: утечка завышает AUC = {verdict}")
    print(f"артефакты → {RESULTS/'experiment_metrics.json'}, {RESULTS/'experiment_runs.csv'}")
    return summary


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "full"
    run(stage)
