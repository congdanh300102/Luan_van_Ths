"""
Stage E — re-run XGBoost SHAP (11-feature main configuration, same model as
the XGBoost row of the refreshed Table 3) in the current software
environment, completing the full-environment refresh alongside Stage D.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW, TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERICAL_COLS,
    RANDOM_STATE, TEST_SIZE,
)
from src.preprocessing import prepare, dataset1_feature_to_group
from src.models import build_pipeline
from src.explainability import compute_shap_values, _mean_abs_importance

OUT = Path(__file__).parent / "conference_paper_out"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "stageE_log.txt", "w", encoding="utf-8")


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


df_raw = pd.read_excel(DATA_RAW)
X_all, y_all = prepare(df_raw, DROP_COLS, TARGET_COL)
y_0 = y_all - 1

cat11 = [c for c in CATEGORICAL_COLS if c in X_all.columns]
num11 = [c for c in NUMERICAL_COLS if c in X_all.columns]

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_0, test_size=TEST_SIZE, stratify=y_0, random_state=RANDOM_STATE
)

pipe = build_pipeline("xgboost", cat11, num11, RANDOM_STATE, imbalance_strategy="smote_moderate")
pipe.fit(X_train, y_train)

shap_values, feature_names, X_display = compute_shap_values(pipe, X_test, "xgboost", max_samples=10000)
log(f"SHAP values shape: {shap_values.shape}")

mean_abs = _mean_abs_importance(shap_values, class_idx=None)
shap_series = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
log("\nXGBoost SHAP importance (mean |phi_j| across all classes):")
for feat, val in shap_series.items():
    log(f"  {feat:20s} {val:.4f}")
shap_series.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"}).to_csv(
    OUT / "xgboost_shap_top.csv", index=False)

df_imp = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
df_imp["group"] = df_imp["feature"].map(dataset1_feature_to_group)
grp = (df_imp.groupby("group")["importance"].sum().sort_values(ascending=False).reset_index())
grp["pct"] = (grp["importance"] / grp["importance"].sum() * 100).round(1)
grp.to_csv(OUT / "xgboost_shap_group.csv", index=False)
log("\nXGBoost SHAP group importance:")
log(grp.to_string(index=False))

log("\nStage E complete.")
LOG.close()
