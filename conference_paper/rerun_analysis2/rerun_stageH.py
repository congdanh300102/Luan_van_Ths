"""
Stage H -- proper paired statistical test for model differences, using the
corrected resampled paired t-test (Nadeau & Bengio, 2003), which accounts for
the non-independence of folds in repeated k-fold cross-validation (a naive
paired t-test / Wilcoxon test on repeated-CV folds understates variance and
inflates Type I error, because folds within the same repeat share data).

Reads conference_paper_out2/repeatedcv_perfold_cw.csv (25 folds x 6 models,
from Stage G) and tests LightGBM against each other model.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).parent / "conference_paper_out2"
LOG = open(OUT / "stageH_log.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


df = pd.read_csv(OUT / "repeatedcv_perfold_cw.csv")
log(f"Loaded {len(df)} folds, models: {list(df.columns)}")

N_SPLITS, N_REPEATS = 5, 5
n = len(df)
ratio = (1 / N_SPLITS) / (1 - 1 / N_SPLITS)  # n_test / n_train = 0.25 for 5-fold


def corrected_resampled_ttest(a, b, ratio, n):
    d = np.asarray(a) - np.asarray(b)
    mean_d = d.mean()
    var_d = d.var(ddof=1)
    corrected_var = var_d * (1.0 / n + ratio)
    if corrected_var <= 0:
        return mean_d, np.nan, np.nan
    t_stat = mean_d / np.sqrt(corrected_var)
    df_ = n - 1
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_))
    return mean_d, t_stat, p_val


def naive_ttest(a, b):
    t_stat, p_val = stats.ttest_rel(a, b)
    return t_stat, p_val


baseline = "lightgbm"
rows = []
for other in ["xgboost", "random_forest", "catboost", "catboost_none", "decision_tree", "logistic"]:
    mean_d, t_c, p_c = corrected_resampled_ttest(df[baseline], df[other], ratio, n)
    t_naive, p_naive = naive_ttest(df[baseline], df[other])
    rows.append({
        "comparison": f"{baseline} vs {other}",
        "mean_diff": mean_d,
        "corrected_t": t_c, "corrected_p": p_c,
        "naive_t": t_naive, "naive_p": p_naive,
    })
    log(f"{baseline} vs {other:14s}  mean_diff={mean_d:+.4f}  "
        f"corrected: t={t_c:.3f} p={p_c:.4f}   naive: t={t_naive:.3f} p={p_naive:.4f}")

tests = pd.DataFrame(rows)
order = np.argsort(tests["corrected_p"].to_numpy())
m = len(tests)
adjusted_sorted = []
running_max = 0.0
for rank, idx in enumerate(order):
    candidate = min(1.0, (m - rank) * tests.loc[idx, "corrected_p"])
    running_max = max(running_max, candidate)
    adjusted_sorted.append(running_max)
adjusted = np.empty(m)
for idx, value in zip(order, adjusted_sorted):
    adjusted[idx] = value
tests["holm_p"] = adjusted
tests.to_csv(OUT / "stat_tests.csv", index=False)
log("\nSaved stat_tests.csv")
log("\nCorrected tests with Holm-adjusted p-values:")
log(tests.to_string(index=False))
log("\nStage H complete.")
LOG.close()
