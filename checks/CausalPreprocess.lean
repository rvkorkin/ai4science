import Mathlib.Data.Real.Basic

namespace CausalPreprocess

/-- Последовательность отсчётов во времени. -/
abbrev Seq := ℕ → ℝ

/-- Обратная разность с нулевой границей (↔ strain_rate_causal: diff, первый отсчёт 0). -/
def bdiff (x : Seq) : Seq
  | 0       => 0
  | (n + 1) => x (n + 1) - x n

/-- Каузальное EWMA: s 0 = x0 ; s(n+1) = a·x(n+1) + (1-a)·s n. -/
def ewma (a : ℝ) (x : Seq) : Seq
  | 0       => x 0
  | (n + 1) => a * x (n + 1) + (1 - a) * ewma a x n

/-- Прореживание фактором R (↔ decimate_causal: каждый R-й отсчёт). -/
def decim (R : ℕ) (s : Seq) : Seq := fun i => s (i * R)

/-- Полная цепочка препроцессинга. -/
def P (a : ℝ) (R : ℕ) (x : Seq) : Seq := decim R (ewma a (bdiff x))

/-- Каузальность diff: выход в i зависит только от входа при j ≤ i. -/
theorem bdiff_causal (x y : Seq) :
    ∀ i, (∀ j ≤ i, x j = y j) → bdiff x i = bdiff y i := by
  intro i h
  cases i with
  | zero   => simp [bdiff]
  | succ n =>
      simp only [bdiff]
      rw [h (n + 1) (le_refl _), h n (Nat.le_succ n)]

/-- Каузальность EWMA (индукция по i). -/
theorem ewma_causal (a : ℝ) (x y : Seq) :
    ∀ i, (∀ j ≤ i, x j = y j) → ewma a x i = ewma a y i := by
  intro i
  induction i with
  | zero =>
      intro h
      simp only [ewma]
      rw [h 0 (le_refl 0)]
  | succ n ih =>
      intro h
      simp only [ewma]
      rw [h (n + 1) (le_refl _), ih (fun j hj => h j (Nat.le_succ_of_le hj))]

/-- No-future-leak всей цепочки: выход i зависит только от входа при j ≤ i·R. -/
theorem P_causal (a : ℝ) (R : ℕ) (x y : Seq) (i : ℕ)
    (h : ∀ j ≤ i * R, x j = y j) : P a R x i = P a R y i := by
  simp only [P, decim]
  apply ewma_causal
  intro k hk
  exact bdiff_causal x y k (fun j hj => h j (hj.trans hk))

/-- Граничное условие y[0] = 0 (↔ тест test_strain_rate_causal_y0_is_zero). -/
theorem P_zero (a : ℝ) (R : ℕ) (x : Seq) : P a R x 0 = 0 := by
  simp [P, decim, ewma, bdiff]

end CausalPreprocess