"""
Descriptive / diagnostic numbers quoted in Sections 3.1-3.5 of the SEBL2026
paper, recomputed with the CURRENT preprocessing code so that every figure in
version 5 comes from one consistent pipeline:

  * Table 1   class distribution + stratified 80/20 split
  * Sec 3.2   Excel serial-date share, skew of BASE_BAL, IQR outlier rates
  * Sec 3.3   correlation of NHOMNO with the target
  * Table 2   Information Value of the 13-variable admissible pool
  * Sec 3.4   correlations among engineered / balance variables
  * Table 2b  DAYS_TO_MATURITY by debt group + share already past maturity
  * Sec 3.5   accuracy / F1 of the univariate "DAYS_TO_MATURITY < 0" rule
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from config.config import (
    DATA_RAW, TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERICAL_COLS,
    RANDOM_STATE, TEST_SIZE,
)
from src.preprocessing import prepare, parse_dates, engineer_features
from src.iv_analysis import compute_iv_table

OUT = Path(__file__).parent / "rerun_v5_out"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "descriptives_log.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


raw = pd.read_excel(DATA_RAW)
log(f"Raw shape: {raw.shape}")

# --- Table 1 -----------------------------------------------------------------
X_all, y_all = prepare(raw, DROP_COLS, TARGET_COL)
y0 = y_all - 1
_, _, y_tr, y_te = train_test_split(X_all, y0, test_size=TEST_SIZE,
                                    stratify=y0, random_state=RANDOM_STATE)
tab1 = pd.DataFrame({
    "total": pd.Series(y_all).value_counts().sort_index(),
    "train": pd.Series(y_tr + 1).value_counts().sort_index(),
    "test": pd.Series(y_te + 1).value_counts().sort_index(),
})
tab1["share_pct"] = (tab1["total"] / tab1["total"].sum() * 100).round(2)
tab1.to_csv(OUT / "table1_distribution.csv")
log("\n=== Table 1: debt-group distribution ===")
log(tab1.to_string())
log(f"Imbalance ratio group1/group3 = {tab1.loc[1,'total'] / tab1.loc[3,'total']:.1f}:1")
log(f"Groups 3-5 share = {tab1.loc[[3,4,5],'share_pct'].sum():.2f}%   "
    f"Groups 2-4 share = {tab1.loc[[2,3,4],'share_pct'].sum():.2f}%   "
    f"Group 5 share = {tab1.loc[5,'share_pct']:.2f}%")

# --- Section 3.2: serial dates, skew, IQR outliers ---------------------------
log("\n=== Section 3.2: date formats and numeric shape ===")
for col in ["OPEN_DATE", "NGAYDENHAN"]:
    n_serial = raw[col].map(lambda v: isinstance(v, (int, float, np.integer, np.floating))
                            and not pd.isnull(v)).sum()
    log(f"  {col}: {n_serial} ({n_serial/len(raw)*100:.2f}%) stored as Excel serial numbers")

dated = parse_dates(raw)
log(f"  unparsed after cleaning: OPEN_DATE={dated['OPEN_DATE'].isna().sum()}, "
    f"NGAYDENHAN={dated['NGAYDENHAN'].isna().sum()}")
eng = engineer_features(dated)

for col in ["BASE_BAL", "CURR_BAL", "LAISUAT", "UTIL_RATE",
            "LOAN_TENURE_DAYS", "DAYS_TO_MATURITY"]:
    s = pd.to_numeric(eng[col], errors="coerce").dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    out_rate = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100
    log(f"  {col:18s} mean={s.mean():.4g}  std={s.std():.4g}  "
        f"IQR-outliers={out_rate:.2f}%")

# --- Section 3.3: NHOMNO correlation with target -----------------------------
log("\n=== Section 3.3: leakage screen ===")
if "NHOMNO" in raw.columns:
    log(f"  corr(NHOMNO, {TARGET_COL}) = "
        f"{pd.to_numeric(raw['NHOMNO'], errors='coerce').corr(raw[TARGET_COL]):.4f}")

# --- Table 2: IV over the 13-variable admissible pool ------------------------
log("\n=== Table 2: Information Value, 13-variable pool ===")
pool_drop = [c for c in DROP_COLS if c not in ("SEX", "LOAIKH")]
X_pool, y_pool = prepare(raw, pool_drop, TARGET_COL)
iv_df = X_pool.copy()
iv_df[TARGET_COL] = y_pool
iv_tab = compute_iv_table(iv_df, TARGET_COL, good_class=1, bins=10)
iv_tab.to_csv(OUT / "table2_iv.csv", index=False)
log(iv_tab.to_string(index=False))

# --- Section 3.4: correlation structure --------------------------------------
log("\n=== Section 3.4: correlations among numeric variables ===")
num_present = [c for c in NUMERICAL_COLS if c in X_pool.columns]
corr = X_pool[num_present].corr()
corr.to_csv(OUT / "numeric_correlations.csv")
log(corr.round(3).to_string())
pairs = (corr.where(~np.eye(len(corr), dtype=bool)).abs().stack()
         .sort_values(ascending=False).drop_duplicates())
log("\nTop |r| pairs:")
for (a, b), v in pairs.head(6).items():
    log(f"  {a} - {b}: r = {corr.loc[a, b]:.3f}")

# --- Table 2b + Section 3.5 univariate rule ----------------------------------
log("\n=== Table 2b: DAYS_TO_MATURITY by debt group ===")
dtm = pd.DataFrame({"dtm": X_pool["DAYS_TO_MATURITY"].values, "grp": y_pool})
t2b = dtm.groupby("grp")["dtm"].agg(["mean", "median", "min", "max"])
t2b["share_negative_pct"] = dtm.groupby("grp")["dtm"].apply(lambda s: (s < 0).mean() * 100)
t2b.to_csv(OUT / "table2b_dtm_by_group.csv")
log(t2b.round(2).to_string())

y_bad = (y_pool != 1).astype(int)
pred_bad = (X_pool["DAYS_TO_MATURITY"].values < 0).astype(int)
log("\n=== Section 3.5: univariate rule 'DAYS_TO_MATURITY < 0' -> non-performing ===")
log(f"  accuracy on binary good/bad split = {(pred_bad == y_bad).mean():.4f}")
log(f"  F1 on the non-performing class    = {f1_score(y_bad, pred_bad):.4f}")

log("\nDone.")
LOG.close()
