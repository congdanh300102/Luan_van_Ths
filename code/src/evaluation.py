"""Evaluation utilities — trả về Plotly figures (không save file)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score,
)
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.base import clone
from itertools import product

NHOMNO_LABELS = {
    1: "Nhóm 1",
    2: "Nhóm 2",
    3: "Nhóm 3",
    4: "Nhóm 4",
    5: "Nhóm 5",
}
GROUP_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    f1_macro    = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    report = classification_report(
        y_true, y_pred,
        target_names=[NHOMNO_LABELS.get(c, str(c)) for c in sorted(np.unique(y_true))],
        output_dict=True,
    )
    return {"f1_macro": f1_macro, "f1_weighted": f1_weighted,
            "roc_auc": auc, "report": report}


def plot_confusion_matrix(y_true, y_pred) -> go.Figure:
    classes = sorted(np.unique(y_true))
    labels  = [NHOMNO_LABELS.get(c, str(c)) for c in classes]
    cm      = confusion_matrix(y_true, y_pred, labels=classes)
    cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    text = [[f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)"
             for j in range(len(classes))] for i in range(len(classes))]

    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels, text=text,
        texttemplate="%{text}", colorscale="Blues",
        showscale=True,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Dự báo", yaxis_title="Thực tế",
        height=420,
    )
    return fig


def plot_feature_importance(pipeline, top_n: int = 20) -> go.Figure | None:
    try:
        clf   = pipeline.named_steps["classifier"]
        prep  = pipeline.named_steps["preprocessor"]
        names = prep.get_feature_names()

        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_).mean(axis=0)
        else:
            return None

        idx = np.argsort(imp)[::-1][:top_n]
        fig = px.bar(
            x=imp[idx[::-1]], y=[names[i] for i in idx[::-1]],
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            title=f"Top {top_n} Feature Importance",
            color=imp[idx[::-1]],
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=max(400, top_n * 22), showlegend=False)
        return fig
    except Exception:
        return None


def plot_model_comparison(results: list) -> go.Figure:
    df  = pd.DataFrame(results)
    metrics = ["f1_macro", "f1_weighted", "roc_auc"]
    labels  = {"f1_macro": "Macro F1", "f1_weighted": "Weighted F1", "roc_auc": "ROC-AUC"}

    fig = go.Figure()
    for m in metrics:
        fig.add_trace(go.Bar(
            name=labels[m],
            x=df["model"],
            y=df[m],
            text=df[m].map(lambda v: f"{v:.4f}"),
            textposition="auto",
        ))
    fig.update_layout(
        barmode="group", title="So sánh hiệu năng các mô hình",
        yaxis=dict(range=[0, 1]), height=400,
    )
    return fig


def cross_val_scores(pipeline, X, y,
                     cv_folds: int = 5, random_state: int = 42) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["f1_macro", "f1_weighted"],
        n_jobs=-1,
    )
    return {
        "f1_macro_mean":    results["test_f1_macro"].mean(),
        "f1_macro_std":     results["test_f1_macro"].std(),
        "f1_weighted_mean": results["test_f1_weighted"].mean(),
        "f1_weighted_std":  results["test_f1_weighted"].std(),
    }


def repeated_stratified_recall(pipeline, X: pd.DataFrame, y: np.ndarray,
                               n_splits: int = 5, n_repeats: int = 10,
                               random_state: int = 42,
                               sample_weight_fn=None) -> tuple[pd.DataFrame, dict]:
    """
    Đánh giá Recall/Precision/F1 từng nhóm nợ qua nhiều lần lặp Stratified
    K-Fold thay vì một lần train/test split duy nhất.

    y : nhãn 0-based (quy ước y_0 = y_1based - 1 dùng xuyên suốt trang huấn
        luyện) — hàm tự +1 khi tra NHOMNO_LABELS để hiển thị.

    Lý do: với các nhóm có rất ít quan sát (VD nhóm 3, 4), một lần split có
    thể vô tình dồn gần hết hoặc gần như không có mẫu vào tập test — khiến
    Recall quan sát được (0% hay 100%) chỉ phản ánh may rủi của 1 lần chia,
    không phải năng lực thật của mô hình. Lặp lại nhiều lần và lấy trung bình
    cho ước lượng ổn định hơn, đồng thời total_support cho biết tổng số lần
    nhóm đó thực sự xuất hiện trong tập test qua tất cả các fold.

    Trả về
    -------
    per_class_df : mỗi hàng là 1 nhóm nợ — recall/precision/f1 mean ± std,
                   và total_support (tổng số mẫu của nhóm đó trên toàn bộ
                   test set của tất cả (n_splits × n_repeats) folds).
    summary      : macro_f1_mean, macro_f1_std, n_splits, n_repeats.

    sample_weight_fn : callable(y_train_fold) -> sample_weight, dùng để tái
                       tạo đúng cost-sensitive weight (VD compute_sample_weights
                       trong src/models.py) trên từng fold — cần thiết cho các
                       model không có class_weight built-in như XGBoost.
    """
    rcv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                  random_state=random_state)
    classes = sorted(np.unique(y))
    per_class = {c: {"recall": [], "precision": [], "f1": [], "support": 0} for c in classes}
    macro_f1_folds = []

    X_arr = X.reset_index(drop=True) if hasattr(X, "reset_index") else X

    for train_idx, test_idx in rcv.split(X_arr, y):
        X_tr = X_arr.iloc[train_idx] if hasattr(X_arr, "iloc") else X_arr[train_idx]
        X_te = X_arr.iloc[test_idx] if hasattr(X_arr, "iloc") else X_arr[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = clone(pipeline)
        if sample_weight_fn is not None:
            model.fit(X_tr, y_tr, classifier__sample_weight=sample_weight_fn(y_tr))
        else:
            model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        report = classification_report(
            y_te, y_pred, labels=classes, output_dict=True, zero_division=0,
        )
        macro_f1_folds.append(report["macro avg"]["f1-score"])
        for c in classes:
            r = report[str(c)]
            per_class[c]["recall"].append(r["recall"])
            per_class[c]["precision"].append(r["precision"])
            per_class[c]["f1"].append(r["f1-score"])
            per_class[c]["support"] += int(r["support"])

    rows = []
    for c in classes:
        r = per_class[c]
        rows.append({
            # y truyền vào hàm này là 0-based (quy ước y_0 = y_1based - 1 dùng
            # xuyên suốt trang huấn luyện), còn NHOMNO_LABELS khoá 1-based —
            # +1 để hiển thị đúng "Nhóm 1"…"Nhóm 5" thay vì "0".."4".
            "Nhóm": NHOMNO_LABELS.get(c + 1, str(c + 1)),
            "Recall (mean±std)":    f"{np.mean(r['recall']):.3f} ± {np.std(r['recall']):.3f}",
            "Precision (mean±std)": f"{np.mean(r['precision']):.3f} ± {np.std(r['precision']):.3f}",
            "F1 (mean±std)":        f"{np.mean(r['f1']):.3f} ± {np.std(r['f1']):.3f}",
            "Tổng support (tất cả fold)": r["support"],
        })

    summary = {
        "macro_f1_mean": float(np.mean(macro_f1_folds)),
        "macro_f1_std":  float(np.std(macro_f1_folds)),
        "n_splits": n_splits, "n_repeats": n_repeats,
    }
    return pd.DataFrame(rows), summary


def predict_with_class_weights(proba: np.ndarray, boost: dict[int, float]) -> np.ndarray:
    """
    Dự báo bằng argmax có trọng số thay vì argmax thô: nhân xác suất mỗi lớp
    với hệ số boost trước khi chọn nhãn. Đây là cách tăng Recall của nhóm hiếm
    (đổi lấy Precision) mà KHÔNG cần huấn luyện lại mô hình — chỉ dịch ranh
    giới quyết định ở bước dự báo.

    proba : mảng (n_samples, n_classes) từ pipeline.predict_proba(), cột theo
            đúng thứ tự lớp 0..n_classes-1.
    boost : dict {class_index (0-based): hệ số nhân}. Lớp không có trong dict
            giữ hệ số 1.0.
    """
    weights = np.array([boost.get(c, 1.0) for c in range(proba.shape[1])])
    return np.argmax(proba * weights, axis=1)


def tune_class_boost(y_true: np.ndarray, proba: np.ndarray,
                     target_classes: list, boost_range=(1, 2, 3, 5, 8, 12, 20),
                     metric: str = "f1_macro") -> tuple[dict, float]:
    """
    Grid-search hệ số boost cho các lớp hiếm (target_classes, 0-based) nhằm
    tối đa hoá Macro-F1 trên tập validation/test hiện có — không cần huấn
    luyện lại mô hình, chỉ tìm lại ranh giới quyết định tối ưu hơn cho việc
    phát hiện nhóm hiếm.

    Cảnh báo: đây là hiệu chỉnh post-hoc trên chính tập dùng để đánh giá, nên
    kết quả boost tìm được mang tính minh hoạ hướng cải thiện; trong thực tế
    cần tìm boost trên một tập validation riêng, tách khỏi tập test cuối cùng
    dùng để báo cáo.

    Trả về (best_boost, best_score).
    """
    best_score, best_boost = -1.0, {c: 1.0 for c in target_classes}
    for combo in product(boost_range, repeat=len(target_classes)):
        boost = dict(zip(target_classes, combo))
        y_pred = predict_with_class_weights(proba, boost)
        score = f1_score(y_true, y_pred, average="macro")
        if score > best_score:
            best_score, best_boost = score, boost
    return best_boost, best_score
