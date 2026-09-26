"""So sánh nhiều phương pháp xử lý mất cân bằng dữ liệu trên cùng 1 mô hình."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.models import build_pipeline
from src.evaluation import compute_metrics, NHOMNO_LABELS


def benchmark_strategies(model_key: str,
                         cat_cols: list, num_cols: list,
                         X_train, y_train, X_test, y_test,
                         strategies: dict,
                         random_state: int = 42,
                         custom_smote_strategy: dict | None = None) -> pd.DataFrame:
    """
    strategies: dict nhãn hiển thị -> imbalance_strategy key (xem models.IMBALANCE_OPTIONS)

    custom_smote_strategy: bắt buộc phải truyền nếu `strategies` có chứa
    "custom" hoặc "smote_class_weight" — nếu không, build_pipeline() sẽ âm
    thầm rơi về mốc _SMOTE_MODERATE mặc định (calibrate cho phân bố lớp của
    bộ A), gây sai lệch khi benchmark trên một bộ dữ liệu khác (VD bộ B).

    Trả về DataFrame: 1 hàng / chiến lược, gồm Macro F1, Weighted F1, ROC-AUC
    và Recall từng nhóm nợ (để thấy rõ đánh đổi giữa các lớp thiểu số).
    """
    rows = []
    for label, strategy_key in strategies.items():
        try:
            pipe = build_pipeline(model_key, cat_cols, num_cols,
                                  random_state=random_state,
                                  imbalance_strategy=strategy_key,
                                  custom_smote_strategy=custom_smote_strategy)
            pipe.fit(X_train, y_train)
        except ValueError:
            # VD: chiến lược SMOTE có target thấp hơn số mẫu gốc của lớp đó
            # (mốc target cố định không tương thích với phân phối lớp của
            # tập dữ liệu đang dùng) — bỏ qua chiến lược này thay vì crash
            # toàn bộ bảng so sánh.
            rows.append({"Chiến lược": label, "Macro F1": np.nan,
                        "Weighted F1": np.nan, "ROC-AUC": np.nan})
            continue

        y_pred  = pipe.predict(X_test) + 1
        y_proba = pipe.predict_proba(X_test)
        y_true  = y_test + 1

        metrics = compute_metrics(y_true, y_pred, y_proba)
        row = {
            "Chiến lược": label,
            "Macro F1": metrics["f1_macro"],
            "Weighted F1": metrics["f1_weighted"],
            "ROC-AUC": metrics["roc_auc"],
        }
        report = metrics.get("report", {})
        for cls in sorted(np.unique(y_true)):
            name = NHOMNO_LABELS.get(cls, str(cls))
            recall = report.get(name, {}).get("recall", np.nan)
            row[f"Recall {name}"] = recall
        rows.append(row)

    return pd.DataFrame(rows)


def plot_imbalance_comparison(df: pd.DataFrame) -> go.Figure:
    """Grouped bar so sánh Macro F1 / Weighted F1 / ROC-AUC giữa các chiến lược."""
    metrics = ["Macro F1", "Weighted F1", "ROC-AUC"]
    fig = go.Figure()
    for m in metrics:
        fig.add_trace(go.Bar(
            name=m, x=df["Chiến lược"], y=df[m],
            text=df[m].map(lambda v: f"{v:.4f}"), textposition="auto",
        ))
    fig.update_layout(
        barmode="group", title="So sánh các phương pháp xử lý mất cân bằng",
        yaxis=dict(range=[0, 1]), height=420,
    )
    return fig


def plot_recall_by_group(df: pd.DataFrame) -> go.Figure:
    """Grouped bar so sánh Recall từng nhóm nợ giữa các chiến lược — thấy rõ đánh đổi."""
    recall_cols = [c for c in df.columns if c.startswith("Recall ")]
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            name=row["Chiến lược"],
            x=[c.replace("Recall ", "") for c in recall_cols],
            y=[row[c] for c in recall_cols],
        ))
    fig.update_layout(
        barmode="group", title="Recall từng nhóm nợ theo chiến lược xử lý mất cân bằng",
        yaxis=dict(range=[0, 1], title="Recall"), height=420,
    )
    return fig
