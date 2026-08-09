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
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

from src.preprocessing import CreditPreprocessor, CatBoostPreprocessor

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

try:
    from catboost import CatBoostClassifier as _CatBoostClassifier

    class CatBoostClassifier(_CatBoostClassifier):
        """CatBoostClassifier.predict() trả về shape (n,1) thay vì (n,) như mọi
        classifier khác của sklearn — gây lệch shape ngầm (numpy broadcasting)
        ở bất kỳ nơi nào trong app gọi pipe.predict(). Ravel tại nguồn để toàn
        bộ codebase dùng chung một hợp đồng (n,)."""
        def predict(self, X, **kwargs):
            return super().predict(X, **kwargs).ravel()

    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False


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
        "Decision Tree":       "decision_tree",
        "Random Forest":       "random_forest",
    }
    if _HAS_XGB:
        opts["XGBoost"] = "xgboost"
    if _HAS_LGB:
        opts["LightGBM"] = "lightgbm"
    if _HAS_CATBOOST:
        opts["CatBoost"] = "catboost"
    return opts


def build_pipeline(model_key: str,
                   categorical_cols: list,
                   numerical_cols: list,
                   random_state: int = 42,
                   imbalance_strategy: str = "smote_moderate",
                   custom_smote_strategy: dict | None = None,
                   model_params: dict | None = None) -> ImbPipeline:
    """
    model_params: override một phần siêu tham số mặc định của classifier
    (VD {"max_depth": 8, "learning_rate": 0.1}) — dùng cho grid search / so
    sánh nhiều cấu hình mà không phải sửa code cho từng lần thử. Tham số nào
    không truyền thì giữ giá trị mặc định đã benchmark. Không áp dụng cho
    random_state, class_weight, cat_features — các tham số cấu trúc của
    pipeline, không phải đối tượng của grid search.

    imbalance_strategy:
      "none"               — không xử lý (baseline tốt nhất về Macro F1)
      "smote_moderate"     — SMOTE chỉ oversample minority vừa phải (mặc định)
      "smote_full"         — SMOTE full balance (không khuyến khích)
      "custom"             — SMOTE theo custom_smote_strategy do người gọi truyền vào
      "class_weight"       — trọng số lớp nghịch đảo tần suất (không resample)
      "borderline_smote"   — BorderlineSMOTE (chỉ oversample điểm biên, giảm nhiễu so với SMOTE thường)
      "smote_tomek"        — SMOTE + loại cặp Tomek link (hybrid over/under-sampling)
      "adasyn"             — ADASYN (oversample thích ứng theo độ khó phân loại)
      "random_undersample" — RandomUnderSampler (giảm mẫu lớp đa số thay vì tăng lớp thiểu số)

    model_key == "catboost": dùng CatBoostPreprocessor (giữ categorical ở dạng
    chuỗi gốc, không label-encode) + CatBoostClassifier(cat_features=...).
    Vì SMOTE/BorderlineSMOTE/ADASYN nội suy khoảng cách nên yêu cầu dữ liệu số,
    không tương thích với categorical dạng chuỗi — với CatBoost, mọi
    imbalance_strategy khác "none" được ánh xạ sang auto_class_weights="Balanced"
    (cơ chế cân bằng lớp tích hợp sẵn của CatBoost) thay vì resampling.
    """
    use_class_weight = imbalance_strategy == "class_weight"
    overrides = model_params or {}

    if model_key == "catboost":
        if not _HAS_CATBOOST:
            raise ImportError("CatBoost không khả dụng: pip install catboost")
        preprocessor = CatBoostPreprocessor(categorical_cols, numerical_cols)
        cat_idx = list(range(len(numerical_cols), len(numerical_cols) + len(categorical_cols)))
        cb_params = {"iterations": 400, "depth": 6, "learning_rate": 0.05, **overrides}
        clf = CatBoostClassifier(
            loss_function="MultiClass", random_state=random_state,
            cat_features=cat_idx,
            auto_class_weights=None if imbalance_strategy == "none" else "Balanced",
            verbose=False,
            **cb_params,
        )
        return ImbPipeline([("preprocessor", preprocessor), ("classifier", clf)])

    preprocessor = CreditPreprocessor(categorical_cols, numerical_cols)
    class_weight = "balanced" if use_class_weight else None

    if model_key == "logistic":
        lr_params = {"C": 1.0, **overrides}
        clf = LogisticRegression(
            max_iter=1000, random_state=random_state,
            solver="lbfgs",
            class_weight=class_weight,
            **lr_params,
        )
    elif model_key == "decision_tree":
        dt_params = {"max_depth": 10, "min_samples_leaf": 10, **overrides}
        clf = DecisionTreeClassifier(
            random_state=random_state, class_weight=class_weight,
            **dt_params,
        )
    elif model_key == "random_forest":
        rf_params = {"n_estimators": 300, "max_depth": 15, "min_samples_leaf": 5, **overrides}
        clf = RandomForestClassifier(
            random_state=random_state, n_jobs=-1, class_weight=class_weight,
            **rf_params,
        )
    elif model_key == "xgboost":
        if not _HAS_XGB:
            raise ImportError("XGBoost cần: brew install libomp")
        xgb_params = {
            "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, **overrides,
        }
        clf = XGBClassifier(
            eval_metric="mlogloss", random_state=random_state,
            n_jobs=-1, verbosity=0,
            **xgb_params,
        )
    elif model_key == "lightgbm":
        if not _HAS_LGB:
            raise ImportError("LightGBM không khả dụng")
        lgb_params = {
            "n_estimators": 400, "max_depth": 8, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, **overrides,
        }
        clf = LGBMClassifier(
            random_state=random_state, n_jobs=-1, verbose=-1,
            class_weight=class_weight,
            **lgb_params,
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    # Chọn sampler (bỏ qua nếu dùng class_weight — tránh double-counting)
    sampler = None
    if imbalance_strategy in ("none", "class_weight"):
        sampler = None
    elif imbalance_strategy == "smote_full":
        sampler = SMOTE(random_state=random_state, k_neighbors=3)
    elif imbalance_strategy == "custom" and custom_smote_strategy is not None:
        sampler = SMOTE(sampling_strategy=custom_smote_strategy,
                        random_state=random_state, k_neighbors=3)
    elif imbalance_strategy == "borderline_smote":
        sampler = BorderlineSMOTE(random_state=random_state, k_neighbors=3)
    elif imbalance_strategy == "smote_tomek":
        sampler = SMOTETomek(random_state=random_state)
    elif imbalance_strategy == "adasyn":
        sampler = ADASYN(random_state=random_state, n_neighbors=3)
    elif imbalance_strategy == "random_undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    else:  # smote_moderate (mặc định)
        sampler = SMOTE(sampling_strategy=_SMOTE_MODERATE,
                        random_state=random_state, k_neighbors=3)

    steps = [("preprocessor", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("classifier", clf))

    return ImbPipeline(steps)


def compute_sample_weights(y) -> np.ndarray:
    """
    Trọng số mẫu tỉ lệ nghịch tần suất lớp — dùng cho XGBoost, vì
    XGBClassifier không có tham số class_weight built-in như các model
    sklearn khác (LogisticRegression/DecisionTree/RandomForest/LightGBM);
    phải truyền qua fit(sample_weight=...) mới có hiệu lực cost-sensitive.

    weight(class c) = n_samples / (n_classes * count(c))  — cùng công thức
    với sklearn class_weight="balanced", chỉ khác cách áp dụng.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {c: n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[label] for label in y])


IMBALANCE_OPTIONS = {
    "SMOTE moderate (khuyến nghị)": "smote_moderate",
    "Không xử lý (baseline)":       "none",
    "Class weight (balanced)":      "class_weight",
    "SMOTE full balance":           "smote_full",
    "Borderline-SMOTE":             "borderline_smote",
    "SMOTE + Tomek links":          "smote_tomek",
    "ADASYN":                       "adasyn",
    "Random undersampling":         "random_undersample",
}
