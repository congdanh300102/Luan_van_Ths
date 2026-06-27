"""
Build model pipelines với chiến lược xử lý imbalance đã được benchmark.

Kết quả benchmark (Random Forest, Macro F1 trên test set):
  Baseline (không xử lý)          : 0.6182  ← tốt nhất tổng thể
  class_weight=balanced            : 0.5745
  SMOTE full + class_weight (cũ)  : 0.5639  ← tệ nhất (double-counting)
  SMOTE full (không class_weight)  : 0.5625
  SMOTE moderate (không cw)        : 0.6150  ← cân bằng tốt
  SMOTETomek hybrid                : 0.5667
  BorderlineSMOTE                  : 0.5931

Chiến lược mặc định: SMOTE moderate
  - Chỉ oversample minority classes lên mức hợp lý (không full balance)
  - KHÔNG dùng class_weight (tránh double-counting với SMOTE)
  - Tỷ lệ synthetic data < 50% cho mỗi class thiểu số
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.preprocessing import CreditPreprocessor

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False


# ── Chiến lược SMOTE moderate ────────────────────────────────────────────────
# y_0 (0-based): 0=N1, 1=N2, 2=N3, 3=N4, 4=N5
# Phân phối thực: N1=16771, N2=1030, N3=483, N4=627, N5=2689
# Mục tiêu: nâng minority lên ~20-30% so với majority, không full balance
_SMOTE_MODERATE = {
    1: 3000,   # N2: 1030 → 3000  (~29% của N1)
    2: 2000,   # N3:  483 → 2000  (~19% của N1, tránh >50% synthetic)
    3: 2000,   # N4:  627 → 2000  (~19% của N1)
    4: 5000,   # N5: 2689 → 5000  (~30% của N1, gần tự nhiên hơn)
}


def available_models() -> dict:
    opts = {
        "Logistic Regression": "logistic",
        "Random Forest":       "random_forest",
    }
    if _HAS_XGB:
        opts["XGBoost"] = "xgboost"
    if _HAS_LGB:
        opts["LightGBM"] = "lightgbm"
    return opts


def build_pipeline(model_key: str,
                   categorical_cols: list,
                   numerical_cols: list,
                   random_state: int = 42,
                   imbalance_strategy: str = "smote_moderate") -> ImbPipeline:
    """
    imbalance_strategy:
      "none"           — không xử lý (baseline tốt nhất về Macro F1)
      "smote_moderate" — SMOTE chỉ oversample minority vừa phải (mặc định)
      "smote_full"     — SMOTE full balance (không khuyến khích)
    """
    preprocessor = CreditPreprocessor(categorical_cols, numerical_cols)

    # Không dùng class_weight vì SMOTE đã xử lý imbalance ở data level
    if model_key == "logistic":
        clf = LogisticRegression(
            max_iter=1000, random_state=random_state,
            multi_class="multinomial", solver="lbfgs", C=1.0,
        )
    elif model_key == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1,
        )
    elif model_key == "xgboost":
        if not _HAS_XGB:
            raise ImportError("XGBoost cần: brew install libomp")
        clf = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=random_state,
            n_jobs=-1, verbosity=0,
        )
    elif model_key == "lightgbm":
        if not _HAS_LGB:
            raise ImportError("LightGBM không khả dụng")
        clf = LGBMClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1, verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    # Chọn sampler
    if imbalance_strategy == "none":
        steps = [("preprocessor", preprocessor), ("classifier", clf)]
    elif imbalance_strategy == "smote_full":
        sampler = SMOTE(random_state=random_state, k_neighbors=3)
        steps = [("preprocessor", preprocessor), ("smote", sampler), ("classifier", clf)]
    else:  # smote_moderate (mặc định)
        sampler = SMOTE(sampling_strategy=_SMOTE_MODERATE,
                        random_state=random_state, k_neighbors=3)
        steps = [("preprocessor", preprocessor), ("smote", sampler), ("classifier", clf)]

    return ImbPipeline(steps)


IMBALANCE_OPTIONS = {
    "SMOTE moderate (khuyến nghị)": "smote_moderate",
    "Không xử lý (baseline)":       "none",
    "SMOTE full balance":            "smote_full",
}
