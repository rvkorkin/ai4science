# -*- coding: utf-8 -*-
"""
Оптимальная каузальная предобработка DAS-данных для frac-hit предсказания.

Принципы:
  1. Только прошлые данные (causal) — применимо в реальном времени.
  2. Правильный LP-фильтр перед децимацией по времени (не просто "взять каждую 6-ю точку").
  3. Единая пространственная сглажка для всех источников данных.
  4. strain_rate вычисляется до децимации на мелкой сетке — результат чище.

Временная децимация:
    raw (dt_in) → [каузальный FIR LP cutoff=1/(2R)] → [↓R] → (dt_out=1 мин)

    Используем Hamming-windowed sinc длиной L = 3R (или задаётся вручную).
    Задержка фильтра = L-1 сэмплов исходной сетки = (L-1)*dt_in секунд.
    Для R=6 (10с→60с): L=18 сэмплов, задержка ≈ 170 с ≈ 3 мин — приемлемо.

Пространственная сглажка:
    Гауссов фильтр вдоль оси глубины, sigma_m — в метрах.
    Для FORGE (dz≈0.4м): sigma_m=1.5 → sigma≈4 канала.
    Для HFTS2 (dz≈2м): sigma_m=2 → sigma≈1 канал (почти нет эффекта, ок).

Strain rate:
    Вариант A (из СЫРЫХ данных, наилучший):
        dr_raw[t] = (s[t] - s[t-k]) / (k * dt_in)   (k=3..6 → 30-60 с window)
        → LP-фильтровать → децимировать → это и есть strain_rate в NPZ.
    Вариант B (из уже 1-минутных данных):
        dr[t] = s[t] - s[t-1]   (1-мин разность)
        → каузальное сглаживание EWMA(tau_min) с tau=1..2 мин.

Использование:
    from preprocess import decimate_causal, spatial_smooth, strain_rate_causal
    from preprocess import rebuild_npz_from_raw, improve_existing_npz
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────────────────────────────────────
#  1. Каузальный FIR LP фильтр
# ─────────────────────────────────────────────────────────────────────────────

def causal_lp_fir(R: int, n_taps: int | None = None) -> np.ndarray:
    """Коэффициенты каузального Hamming-sinc FIR LP фильтра.

    Parameters
    ----------
    R        : фактор децимации (cutoff = 0.5/R нормированная частота)
    n_taps   : длина фильтра (None → 3R; для более строгого подавления 4-5R)

    Returns
    -------
    h : (n_taps,) — h[0] = коэффициент для ТЕКУЩЕГО сэмпла (самого нового),
                    h[n_taps-1] = коэффициент для наиболее СТАРОГО сэмпла.
    """
    L = n_taps or (3 * R)
    # центр sinc помещаем в t=L-1 (правый конец) → все прошлые + текущий сэмпл
    k = np.arange(L)
    center = float(L - 1)
    fc = 0.5 / R
    h = np.sinc(2.0 * fc * (k - center)) * np.hamming(L)
    h = h[::-1]          # h[0] = вклад нового сэмпла, h[L-1] = старого
    h /= h.sum()
    return h.astype(np.float64)


def decimate_causal(x: np.ndarray, R: int, axis: int = 0,
                    n_taps: int | None = None) -> np.ndarray:
    """Каузальный LP-фильтр + децимация вдоль оси `axis`.

    Выходной сэмпл i использует только входные сэмплы 0..R*i.
    Форма выхода: вместо N → N//R (по оси `axis`).

    Parameters
    ----------
    x       : входной массив
    R       : фактор децимации (целый)
    axis    : ось времени (0 = строки)
    n_taps  : длина FIR (None → 3R)

    Notes
    -----
    Небольшой начальный транзиент: первые n_taps входных сэмплов фильтруются
    «в частичный» буфер (граничные условия 'edge' — предполагаем постоянство
    сигнала до начала записи), что адекватно для медленного геофизического сигнала.
    """
    h = causal_lp_fir(R, n_taps)
    L = len(h)
    N = x.shape[axis]

    # переставляем временну́ю ось в начало для удобства
    x = np.moveaxis(x, axis, 0)
    shape_rest = x.shape[1:]
    x_flat = x.reshape(N, -1).astype(np.float64)   # (N, D)
    D = x_flat.shape[1]

    # количество выходных сэмплов
    M = N // R
    out = np.empty((M, D), np.float64)

    for i in range(M):
        t = i * R                                   # позиция в исходном сигнале
        # собираем прошлые L сэмплов (с краевым условием: повторяем первый)
        t_start = t - L + 1
        if t_start >= 0:
            seg = x_flat[t_start: t + 1]           # (L, D)
        else:
            pad = -t_start
            seg = np.concatenate([
                np.tile(x_flat[:1], (pad, 1)),      # граничное условие edge
                x_flat[:t + 1]
            ], axis=0)                              # (L, D)
        # convolution h · seg (h[0] = новый, h[-1] = старый)
        # seg строки: [старейший, ..., новейший], h[0] = коэф. новейшего
        out[i] = seg[::-1].T @ h                   # (D,)

    out = out.reshape((M,) + shape_rest)
    return np.moveaxis(out, 0, axis).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Пространственная сглажка
# ─────────────────────────────────────────────────────────────────────────────

def spatial_smooth(x: np.ndarray, dz_m: float,
                   sigma_m: float = 2.0, axis: int = -1) -> np.ndarray:
    """Гауссова сглажка вдоль оси глубины.

    Parameters
    ----------
    x       : (T, D) или любой массив
    dz_m    : шаг по глубине в метрах
    sigma_m : радиус сглажки в метрах (по умолчанию 2 м)
    axis    : ось глубины (по умолчанию последняя)

    Returns
    -------
    сглаженный массив той же формы и типа
    """
    sigma_ch = max(0.0, sigma_m / max(dz_m, 1e-6))
    if sigma_ch < 0.5:
        return x                    # сглажка меньше полуканала — нет смысла
    return gaussian_filter1d(x.astype(np.float32), sigma=sigma_ch,
                             axis=axis, mode='nearest')


# ─────────────────────────────────────────────────────────────────────────────
#  3. Каузальный strain rate
# ─────────────────────────────────────────────────────────────────────────────

def strain_rate_causal(strain: np.ndarray, dt_sec: float,
                       smooth_sec: float = 0.0,
                       axis: int = 0) -> np.ndarray:
    """Каузальный strain rate из уже децимированного (1-мин) strain.

    Метод: backward difference + каузальное EWMA-сглаживание.

    Parameters
    ----------
    strain     : (T, D) strain в единицах [любых]
    dt_sec     : шаг по времени в секундах (обычно 60)
    smooth_sec : постоянная времени EWMA в секундах (0 = без сглаживания).
                 Рекомендуется 90-120 с (1.5-2 мин) для поля.
    axis       : ось времени

    Returns
    -------
    strain_rate : (T, D), первая строка = 0 (нет данных до начала)
    """
    strain = np.moveaxis(strain.astype(np.float64), axis, 0)
    T = strain.shape[0]
    dr = np.empty_like(strain)
    dr[0] = 0.0
    dr[1:] = (strain[1:] - strain[:-1]) / dt_sec

    if smooth_sec > 0.0:
        # EWMA: α = 1 - exp(-dt/τ), τ = smooth_sec
        alpha = 1.0 - np.exp(-dt_sec / smooth_sec)
        s = dr.copy()
        for t in range(1, T):
            s[t] = alpha * dr[t] + (1.0 - alpha) * s[t - 1]
        dr = s

    return np.moveaxis(dr, 0, axis).astype(np.float32)


def strain_rate_from_raw(raw: np.ndarray, dt_in_sec: float,
                         R: int, span_raw: int = 3,
                         n_taps: int | None = None,
                         dz_m: float | None = None,
                         sigma_m: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Оптимальный strain rate из СЫРЫХ данных высокой частоты.

    Шаги:
      1. Пространственная сглажка (если dz_m задан)
      2. Backward difference по времени с окном span_raw сэмплов → sr_raw
         dr[t] = (raw[t] - raw[t-span_raw]) / (span_raw * dt_in_sec)
      3. LP-децимация strain и sr_raw до 1 мин (R = dt_out / dt_in)

    Parameters
    ----------
    raw         : (T_raw, D) сырые данные в физических единицах
    dt_in_sec   : шаг сырых данных (10с для HFTS2 0.1 Hz, 2с для YPF)
    R           : фактор децимации (dt_out = dt_in * R → должен быть 60/dt_in)
    span_raw    : окно backward-diff в сырых сэмплах (span_raw * dt_in ≈ 30-60 с)
    n_taps      : длина FIR (None → 3R)
    dz_m        : шаг по глубине в метрах (None → без пространственной сглажки)
    sigma_m     : радиус пространственной сглажки в метрах

    Returns
    -------
    strain_1min    : (T_raw//R, D) — strain после LP-децимации
    sr_1min        : (T_raw//R, D) — strain rate после LP-децимации
    """
    raw = raw.astype(np.float32)

    # 1. Пространственная сглажка (вдоль последней оси — глубина)
    if dz_m is not None and sigma_m > 0:
        raw = spatial_smooth(raw, dz_m, sigma_m, axis=1)

    # 2. Backward-difference strain rate (span_raw сэмплов)
    T = raw.shape[0]
    dr_raw = np.empty_like(raw)
    dr_raw[:span_raw] = 0.0
    dr_raw[span_raw:] = (raw[span_raw:] - raw[:-span_raw]) / (span_raw * dt_in_sec)

    # 3. LP-децимация обоих по времени
    strain_1min = decimate_causal(raw, R, axis=0, n_taps=n_taps)
    sr_1min     = decimate_causal(dr_raw, R, axis=0, n_taps=n_taps)
    return strain_1min, sr_1min


# ─────────────────────────────────────────────────────────────────────────────
#  4. Патч существующих NPZ (без исходных данных)
# ─────────────────────────────────────────────────────────────────────────────

def improve_existing_npz(path_in: str, path_out: str | None = None,
                         sigma_m: float = 2.0,
                         sr_smooth_sec: float = 90.0,
                         overwrite: bool = False) -> dict:
    """Улучшает существующий NPZ без доступа к сырым данным.

    Что делается:
      - Пространственная сглажка strain (sigma_m метров вдоль глубины)
      - Пересчёт strain_rate как causal backward-diff + EWMA(sr_smooth_sec)
      - Перезапись (или новый файл)

    Parameters
    ----------
    path_in       : исходный .npz
    path_out      : выходной .npz (None → overwrite path_in)
    sigma_m       : пространственная сглажка (метры)
    sr_smooth_sec : постоянная времени EWMA для strain_rate (секунды)
    overwrite     : True → разрешить перезапись path_in

    Returns
    -------
    dict с ключами 'strain_changed', 'sr_added', 'sr_changed', 'path'
    """
    import os
    from pathlib import Path

    path_in  = str(path_in)
    path_out = path_out or path_in
    if path_out == path_in and not overwrite:
        raise ValueError("path_out == path_in — передайте overwrite=True явно")

    z = np.load(path_in, allow_pickle=False)
    data = {k: z[k] for k in z.files}

    strain = data['strain'].astype(np.float32)      # (T, D)
    depth  = data.get('depth')
    dt     = float(data['dt_sec']) if 'dt_sec' in data else 60.0

    if depth is not None:
        dz_arr = np.abs(np.diff(depth.astype(float)))
        dz_m = float(np.median(dz_arr)) if len(dz_arr) else 1.0
    else:
        dz_m = 1.0

    # пространственная сглажка
    strain_new = spatial_smooth(strain, dz_m, sigma_m=sigma_m, axis=1)
    strain_changed = not np.allclose(strain_new, strain, atol=1e-9)
    data['strain'] = strain_new

    # strain_rate
    sr_old = data.get('strain_rate')
    sr_new = strain_rate_causal(strain_new, dt, smooth_sec=sr_smooth_sec, axis=0)
    data['strain_rate'] = sr_new

    sr_added   = sr_old is None
    sr_changed = (not sr_added) and (not np.allclose(sr_new, sr_old.astype(np.float32),
                                                      atol=1e-9, equal_nan=True))
    # метаданные обработки
    data['dt_sec']          = np.float32(dt)
    data['preprocess_dz_m'] = np.float32(dz_m)
    data['preprocess_sigma_m'] = np.float32(sigma_m)
    data['preprocess_sr_smooth_sec'] = np.float32(sr_smooth_sec)

    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path_out, **data)

    return {
        'strain_changed': strain_changed,
        'sr_added': sr_added,
        'sr_changed': sr_changed,
        'path': path_out,
        'dz_m': dz_m,
        'sigma_ch': sigma_m / dz_m,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  5. Пакетная обработка всей папки field_data_enriched
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_folder(folder: str, out_folder: str | None = None,
                   sigma_m: float = 2.0, sr_smooth_sec: float = 90.0,
                   n_jobs: int = 1) -> list[dict]:
    """Применяет improve_existing_npz ко всем NPZ в папке.

    Parameters
    ----------
    folder        : папка с исходными NPZ
    out_folder    : куда писать (None → overwrite in place)
    sigma_m       : пространственная сглажка, м
    sr_smooth_sec : сглажка strain_rate, с
    n_jobs        : 1 = последовательно; >1 = multiprocessing.Pool

    Returns
    -------
    list[dict] — результаты по каждому файлу
    """
    from pathlib import Path

    folder_p    = Path(folder)
    out_folder_p = Path(out_folder) if out_folder else folder_p
    out_folder_p.mkdir(parents=True, exist_ok=True)

    files = sorted(folder_p.glob("*.npz"))
    if not files:
        print(f"Нет NPZ-файлов в {folder}")
        return []

    overwrite = (out_folder_p == folder_p)
    results = []

    def _proc(p):
        out = str(out_folder_p / p.name)
        try:
            r = improve_existing_npz(str(p), out,
                                     sigma_m=sigma_m,
                                     sr_smooth_sec=sr_smooth_sec,
                                     overwrite=overwrite)
            return r
        except Exception as e:
            return {'path': str(p), 'error': str(e)}

    if n_jobs == 1:
        for i, p in enumerate(files):
            r = _proc(p)
            tag = "NEW_SR" if r.get('sr_added') else ("UPD_SR" if r.get('sr_changed') else "ok")
            sp = "SMTH" if r.get('strain_changed') else "  -- "
            print(f"[{i+1:3d}/{len(files)}] {sp} {tag}  sigma_ch={r.get('sigma_ch', '?'):.1f}  {Path(r['path']).name}")
            results.append(r)
    else:
        from multiprocessing import Pool
        with Pool(n_jobs) as pool:
            results = pool.map(_proc, files)

    n_ok  = sum(1 for r in results if 'error' not in r)
    n_err = len(results) - n_ok
    print(f"\nГотово: {n_ok} файлов обработано, {n_err} ошибок")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  6. Конвертер из СЫРЫХ H5 (HFTS2 / YPF / Argentina стиль)
# ─────────────────────────────────────────────────────────────────────────────

def raw_h5_to_1min(lowfreq_dir: str,
                   t0_posix: float, t1_posix: float,
                   dt_in_sec: float = 10.0,
                   samples_per_file: int = 30,
                   span_raw: int = 3,
                   sigma_m: float = 2.0,
                   n_taps: int | None = None) -> dict | None:
    """Читает сырые HDF5-файлы OptaSense CWC 0.1 Hz и возвращает 1-мин strain + sr.

    Цепочка:
        H5 (30×10с) → конкатенация → пространственная сглажка
                    → strain_rate из backward-diff (span_raw×10с)
                    → каузальная LP-децимация до 1 мин

    Parameters
    ----------
    lowfreq_dir      : папка с sensor_*.h5 (0.1 Hz FBE)
    t0_posix, t1_posix : временное окно (Unix-секунды)
    dt_in_sec        : шаг сырых данных (10 с для OptaSense 0.1 Hz)
    samples_per_file : сэмплов в одном H5 файле (30 для 5-минутных файлов)
    span_raw         : окно backward-diff для sr (в сырых сэмплах, 3 → 30 с)
    sigma_m          : пространственная сглажка в метрах
    n_taps           : длина FIR (None → 3R)

    Returns
    -------
    dict с ключами:
        strain      : (T_min, D) float32
        strain_rate : (T_min, D) float32
        time_posix  : (T_min,) float64  — центры 1-мин bins
        depth_m     : (D,) float32
    или None при ошибке
    """
    import h5py, os, re
    from pathlib import Path

    _FN = re.compile(r"sensor_(\d{4})-(\d{2})-(\d{2})T(\d{2})(\d{2})(\d{2})(?:\.(\d+))?Z", re.I)

    def _ts(fname):
        from datetime import datetime, timezone
        m = _FN.search(fname)
        if not m:
            return None
        y, mo, d, h, mi, s = (int(m.group(i)) for i in range(1, 7))
        us = int((m.group(7) or "0").ljust(6, "0")[:6])
        return datetime(y, mo, d, h, mi, s, us, tzinfo=timezone.utc).timestamp()

    folder = Path(lowfreq_dir)
    fnames = sorted(f for f in os.listdir(folder) if f.lower().endswith(".h5"))
    ftimes = np.array([_ts(f) for f in fnames], dtype=object)
    mask   = np.array([t is not None and t0_posix <= t <= t1_posix for t in ftimes])
    sel_f  = [f for f, ok in zip(fnames, mask) if ok]
    sel_t  = [t for t, ok in zip(ftimes, mask) if ok]
    if not sel_f:
        return None

    blocks, rtimes = [], []
    dx, start = None, None
    for fname, ft in sorted(zip(sel_f, sel_t), key=lambda x: x[1]):
        try:
            with h5py.File(folder / fname, "r") as h:
                raw = h["Acquisition/Raw[0]/RawData"][:]           # (Nloci, Ntime)
                rt  = h["Acquisition/Raw[0]/RawDataTime"][:] / 1e6 # секунды
                if dx is None:
                    a   = h["Acquisition"].attrs
                    dx  = float(a.get("SpatialSamplingInterval", 1.0))
                    start = float(a.get("StartLocusIndex", 0))
            blocks.append(raw.T.astype(np.float32))                # (Ntime, Nloci)
            rtimes.append(rt)
        except Exception as e:
            print(f"  skip {fname}: {e}")

    if not blocks:
        return None

    strain_raw = np.concatenate(blocks, axis=0)   # (T_raw, D)
    t_raw      = np.concatenate(rtimes, axis=0)
    order      = np.argsort(t_raw)
    strain_raw, t_raw = strain_raw[order], t_raw[order]

    D = strain_raw.shape[1]
    depth_m = (start + np.arange(D) * dx).astype(np.float32)
    dz_m = float(dx)

    R = max(1, round(60.0 / dt_in_sec))          # фактор децимации (6 для 10с→60с)

    strain_1min, sr_1min = strain_rate_from_raw(
        strain_raw, dt_in_sec, R,
        span_raw=span_raw, n_taps=n_taps,
        dz_m=dz_m, sigma_m=sigma_m
    )
    # временна́я ось 1-минутных сэмплов (центры)
    M = strain_1min.shape[0]
    t0_out = t_raw[0]
    t_1min = t0_out + np.arange(M) * 60.0 + 30.0   # центр окна

    return {
        'strain':      strain_1min,
        'strain_rate': sr_1min,
        'time_posix':  t_1min.astype(np.float64),
        'depth_m':     depth_m,
        'dt_sec':      np.float32(60.0),
        'dz_m':        np.float32(dz_m),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print("\nИспользование:")
        print("  py -3 preprocess.py patch  <folder_in> [folder_out]  [sigma_m=2.0]  [sr_smooth=90]")
        print("  py -3 preprocess.py test                                            (быстрый smoke-test)")
        sys.exit(0)

    cmd = args[0]

    if cmd == "test":
        print("=== smoke test ===")
        rng = np.random.default_rng(42)
        # тестируем децимацию
        T_raw, D = 600, 100
        t_raw = np.cumsum(np.ones(T_raw) * 10.0)   # 10-с сетка
        raw = rng.standard_normal((T_raw, D)).astype(np.float32)
        # добавляем медленный тренд
        raw += np.sin(2 * np.pi * t_raw[:, None] / 600.0) * 10.0
        s1, sr1 = strain_rate_from_raw(raw, dt_in_sec=10.0, R=6, span_raw=3,
                                       dz_m=2.0, sigma_m=2.0)
        print(f"  decimate_causal:     raw{raw.shape} -> {s1.shape}  R=6  ✓")
        print(f"  strain_rate_from_raw: sr{sr1.shape}  "
              f"range=[{sr1.min():.4f}, {sr1.max():.4f}]  ✓")

        # тест improve_existing_npz
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            p_in  = os.path.join(tmp, "test.npz")
            p_out = os.path.join(tmp, "test_out.npz")
            depth = np.linspace(4000, 5000, D).astype(np.float32)
            np.savez(p_in, strain=raw[:100], depth=depth, dt_sec=np.float32(10.0))
            r = improve_existing_npz(p_in, p_out, sigma_m=2.0, sr_smooth_sec=90.0)
            with np.load(p_out) as z:
                sh_s  = z['strain'].shape
                sh_sr = z['strain_rate'].shape
            print(f"  improve_existing_npz: strain{sh_s}  "
                  f"sr{sh_sr}  dz_m={r['dz_m']:.2f}m  ✓")
        print("=== все тесты пройдены ===")

    elif cmd == "patch":
        folder_in  = args[1] if len(args) > 1 else "field_data_enriched"
        folder_out = args[2] if len(args) > 2 else None
        sigma_m    = float(args[3]) if len(args) > 3 else 2.0
        sr_smooth  = float(args[4]) if len(args) > 4 else 90.0
        print(f"Патч {folder_in} -> {folder_out or '(on-site)'}  "
              f"sigma={sigma_m}m  sr_smooth={sr_smooth}s")
        rebuild_folder(folder_in, folder_out,
                       sigma_m=sigma_m, sr_smooth_sec=sr_smooth)
    else:
        print(f"Неизвестная команда: {cmd}. Используйте patch | test")
        sys.exit(1)
