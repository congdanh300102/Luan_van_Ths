"""
Combine verify_full_grid.py (5 models x none) and verify_full_grid2.py
(5 models x balanced + CatBoost x {none, balanced}) per-fold CSVs into one
fully paired 12-configuration table (all sharing the identical 25 folds),
then run Nadeau-Bengio corrected paired t-tests of the single best
configuration vs. every other configuration, with Holm-Bonferroni correction
across the resulting family of 11 comparisons.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).parent / "verify_out"

pf1 = pd.read_csv(OUT / "verify_full_grid_perfold.csv")   # 5 models x none
pf2 = pd.read_csv(OUT / "verify_full_grid2_perfold.csv")  # 5 models x balanced + catboost x2

combined = pd.concat([pf1, pf2], axis=1)
combined.to_csv(OUT / "verify_combined_perfold.csv", index=False)

N_SPLITS, N_REPEATS = 5, 5
N = N_SPLITS * N_REPEATS
ratio = (1 / N_SPLITS) / (1 - 1 / N_SPLITS)  # n_test/n_train for 5-fold = 0.25

summary = combined.mean().sort_values(ascending=False)
print("=== Mean Macro-F1, all 12 configurations (25 folds each) ===")
for name, m in summary.items():
    s = combined[name].std()
    print(f"{name:26s} {m:.4f} +/- {s:.4f}")

best_name = summary.index[0]
best = combined[best_name].values
print(f"\nBest configuration: {best_name} ({summary.iloc[0]:.4f})")

rows = []
others = [c for c in combined.columns if c != best_name]
for name in others:
    other = combined[name].values
    d = best - other
    mean_d = d.mean()
    var_d = d.var(ddof=1)
    corrected_var = var_d * (1 / N + ratio)
    corrected_var = max(corrected_var, 1e-12)
    t_stat = mean_d / np.sqrt(corrected_var)
    df = N - 1
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    rows.append({"comparator": name, "mean_diff": mean_d, "t": t_stat, "p_raw": p_val})

res = pd.DataFrame(rows).sort_values("p_raw")
# Holm-Bonferroni step-down correction
m = len(res)
res = res.reset_index(drop=True)
res["p_holm"] = 0.0
max_so_far = 0.0
for i in range(m):
    adj = (m - i) * res.loc[i, "p_raw"]
    max_so_far = max(max_so_far, adj)
    res.loc[i, "p_holm"] = min(max_so_far, 1.0)

res = res.sort_values("mean_diff")
print(f"\n=== {best_name} vs. all others: Nadeau-Bengio corrected paired t-test + Holm-Bonferroni ===")
for _, r in res.iterrows():
    print(f"{best_name} minus {r['comparator']:26s} diff={r['mean_diff']:+.4f}  t={r['t']:+.2f}  p_raw={r['p_raw']:.4f}  p_holm={r['p_holm']:.4f}")

res.to_csv(OUT / "verify_combined_stats.csv", index=False)
print("\nDone.")
