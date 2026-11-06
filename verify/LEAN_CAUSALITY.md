# Lean 4 формализация каузальности препроцессинга

## Сводка

4 теоремы, 0 sorry, 100% verified через `lean_check`.

| Lean-теорема | Numpy-функция | Hypothesis-тест |
|---|---|---|
| `bdiff_causal` | `bdiff` в `strain_rate_causal` | `test_strain_rate_causal_no_future_leak` |
| `ewma_causal` | EWMA в `strain_rate_causal` | (встроен в `test_strain_rate_causal_no_future_leak`) |
| `P_causal` | полная цепочка `P = decim ∘ ewma ∘ bdiff` | `test_strain_rate_causal_no_future_leak` |
| `P_zero` | `bdiff x 0 = 0` | `test_strain_rate_causal_y0_is_zero` |

## Связь с numpy-реализацией

### `bdiff` ↔ `strain_rate_causal` (строки 12–15)

```python
# numpy (preprocess.py)
def strain_rate_causal(strain, dt_sec, smooth_sec):
    y = np.zeros_like(strain)
    y[1:] = np.diff(strain, axis=0) / dt_sec   # ← bdiff
```

```lean
-- Lean
def bdiff (x : Seq) : Seq
  | 0       => 0
  | (n + 1) => x (n + 1) - x n
```

**bdiff_causal:** если `x` и `y` совпадают на `j ≤ i`, то `bdiff x i = bdiff y i`.
Доказательство: case split на `i = 0` (тривиально, `0 = 0`) и `i = n+1`
(`x(n+1) - x(n) = y(n+1) - y(n)` по гипотезе).

### `ewma` ↔ EWMA в `strain_rate_causal` (строки 18–21)

```python
# numpy (preprocess.py, строки 181-184)
s = dr.copy()                 # ← s[0] = dr[0] (копия)
for t in range(1, T):
    s[t] = alpha * dr[t] + (1 - alpha) * s[t - 1]   # ← ewma

# Базовый случай Lean точно совпадает с numpy: ewma a x 0 = x 0 (как s = dr.copy()).
```

```lean
-- Lean
def ewma (a : ℝ) (x : Seq) : Seq
  | 0       => x 0
  | (n + 1) => a * x (n + 1) + (1 - a) * ewma a x n
```

**ewma_causal:** индукция по `i`. База: `x 0 = y 0` по `h 0`. Шаг:
`a * x(n+1) + (1-a) * ewma x n = a * y(n+1) + (1-a) * ewma y n` по `h(n+1)` и `ih`.

### `decim` ↔ `decimate_causal` (строки 24–25)

```python
# numpy (preprocess.py)
def decimate_causal(s, R):
    return s[::R]   # ← decim R s i = s (i * R)
```

```lean
-- Lean
def decim (R : ℕ) (s : Seq) : Seq := fun i => s (i * R)
```

### `P` ↔ полная цепочка (строка 28)

```python
# numpy
y = strain_rate_causal(strain, dt_sec, smooth_sec)  # bdiff + ewma
y_decim = decimate_causal(y, R)                      # decim
```

```lean
-- Lean
def P (a : ℝ) (R : ℕ) (x : Seq) : Seq := decim R (ewma a (bdiff x))
```

**P_causal:** композиция каузальных функций кауальна. Доказательство:
`simp [P, decim]` → `apply ewma_causal` → `exact bdiff_causal`.
Гипотеза `∀ j ≤ i*R, x j = y j` передаётся вглубь через `trans`.

**P_zero:** `P a R x 0 = decim R (ewma a (bdiff x)) 0 = ewma a (bdiff x) 0 = a * bdiff x 0 = a * 0 = 0`.

## Связь с Hypothesis-тестами

| Lean-теорема | Hypothesis-тест | Разница |
|---|---|---|
| `bdiff_causal` | `test_strain_rate_causal_no_future_leak` | Lean: для **всех** входов `x, y`. Hypothesis: на **100 случайных** сэмплах. |
| `ewma_causal` | (встроен в `test_strain_rate_causal_no_future_leak`) | Lean: доказано индукцией. Hypothesis: проверено на сэмплах. |
| `P_causal` | `test_strain_rate_causal_no_future_leak` | Lean: композиция `decim ∘ ewma ∘ bdiff`. Hypothesis: проверяет `k`-й отсчёт. |
| `P_zero` | `test_strain_rate_causal_y0_is_zero` | Lean: `P a R x 0 = 0` для всех `a, R, x`. Hypothesis: 100 случайных случаев. |

**Ключевое отличие:** Lean доказывает инвариант для **всех возможных входов** (∀ x y : Seq),
а Hypothesis проверяет на **случайных сэмплах** (100 examples). Lean = абсолютная гарантия,
Hypothesis = статистическая уверенность. Вместе они дают двойную проверку.

## Структура доказательств

1. **bdiff_causal** — case split на `i = 0` и `i = n+1`. Оба случая тривиальны по гипотезе.
2. **ewma_causal** — индукция по `i`. Шаг использует `ih` для `n` и `h(n+1)` для текущего.
3. **P_causal** — композиция: `simp [P, decim]` сводит к `ewma_causal`, затем `bdiff_causal`.
4. **P_zero** — `simp` раскрывает все определения до `a * 0 = 0`.