from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.preprocessing import CreditPreprocessor

# Optional heavy dependencies
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


def available_models() -> dict:
    """Trả về dict {display_name: key} chỉ gồm model đang dùng được."""
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
                   random_state: int = 42) -> ImbPipeline:

    preprocessor = CreditPreprocessor(categorical_cols, numerical_cols)

    if model_key == "logistic":
        clf = LogisticRegression(
            max_iter=1000, random_state=random_state,
            class_weight="balanced", multi_class="multinomial",
            solver="lbfgs", C=1.0,
        )
    elif model_key == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=5,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
    elif model_key == "xgboost":
        if not _HAS_XGB:
            raise ImportError("XGBoost không khả dụng. Cài đặt libomp (brew install libomp).")
        clf = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=random_state, n_jobs=-1,
            verbosity=0,
        )
    elif model_key == "lightgbm":
        if not _HAS_LGB:
            raise ImportError("LightGBM không khả dụng.")
        clf = LGBMClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=random_state,
            n_jobs=-1, verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    return ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",        SMOTE(random_state=random_state, k_neighbors=3)),
        ("classifier",   clf),
    ])
