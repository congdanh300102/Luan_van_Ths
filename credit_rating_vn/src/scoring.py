"""Credit scoring utilities — trả về Plotly figures."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

RISK_WEIGHTS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
SCORE_MIN = 300
SCORE_MAX = 850

SCORE_BANDS = [
    (750, 850, "A+", "Xuất sắc",         "#1a9850"),
    (700, 749, "A",  "Tốt",               "#66bd63"),
    (650, 699, "B+", "Khá",               "#a6d96a"),
    (600, 649, "B",  "Trung bình khá",     "#fee08b"),
    (550, 599, "C+", "Trung bình",         "#fdae61"),
    (500, 549, "C",  "Trung bình yếu",     "#f46d43"),
    (450, 499, "D",  "Yếu",               "#d73027"),
    (300, 449, "E",  "Rất yếu / Từ chối", "#a50026"),
]


def proba_to_score(proba: np.ndarray) -> np.ndarray:
    risk  = proba @ RISK_WEIGHTS
    score = SCORE_MAX - risk * (SCORE_MAX - SCORE_MIN)
    return np.round(score).astype(int)


def classify_score(score: int) -> tuple[str, str, str]:
    for lo, hi, grade, desc, color in SCORE_BANDS:
        if lo <= score <= hi:
            return grade, desc, color
    return "E", "Không xác định", "#a50026"


def build_score_df(proba: np.ndarray, y_pred: np.ndarray,
                   y_true: np.ndarray | None = None) -> pd.DataFrame:
    scores  = proba_to_score(proba)
    grades  = [classify_score(s) for s in scores]

    df = pd.DataFrame({
        **({} if y_true is None else {"nhom_thuc_te": y_true}),
        "nhom_du_bao":   y_pred,
        **{f"xac_suat_nhom_{i+1}": proba[:, i].round(4) for i in range(5)},
        "diem_tin_dung": scores,
        "hang":   [g[0] for g in grades],
        "mo_ta":  [g[1] for g in grades],
    })
    return df


def plot_score_distribution(df_scores: pd.DataFrame) -> go.Figure:
    scores = df_scores["diem_tin_dung"]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Phân phối điểm tín dụng", "Phân bổ hạng tín dụng"),
    )

    # Histogram
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=40,
        marker_color="steelblue", name="Điểm",
        showlegend=False,
    ), row=1, col=1)
    for lo, _, grade, _, color in SCORE_BANDS:
        fig.add_vline(x=lo, line_dash="dash", line_color="red",
                      line_width=1, row=1, col=1)

    # Bar chart by grade
    grade_order = [b[2] for b in SCORE_BANDS]
    vc = df_scores["hang"].value_counts().reindex(grade_order, fill_value=0)
    colors = [b[4] for b in SCORE_BANDS]
    fig.add_trace(go.Bar(
        x=vc.index, y=vc.values,
        marker_color=colors, name="Hạng",
        text=[f"{v:,}" for v in vc.values],
        textposition="auto",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(height=400, title_text="Kết quả chấm điểm tín dụng")
    return fig


def plot_score_by_group(df_scores: pd.DataFrame) -> go.Figure:
    if "nhom_thuc_te" not in df_scores.columns:
        return None
    group_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
    groups = sorted(df_scores["nhom_thuc_te"].unique())
    fig = go.Figure()
    for g, color in zip(groups, group_colors):
        data = df_scores.loc[df_scores["nhom_thuc_te"] == g, "diem_tin_dung"]
        fig.add_trace(go.Box(
            y=data, name=f"Nhóm {g}",
            marker_color=color, boxpoints="outliers",
        ))
    fig.update_layout(
        title="Điểm tín dụng theo nhóm nợ thực tế",
        yaxis_title="Điểm tín dụng",
        height=420,
    )
    return fig


def score_gauge(score: int) -> go.Figure:
    grade, desc, color = classify_score(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 600},
        gauge={
            "axis": {"range": [SCORE_MIN, SCORE_MAX]},
            "bar":  {"color": color},
            "steps": [
                {"range": [lo, hi], "color": c}
                for lo, hi, _, _, c in reversed(SCORE_BANDS)
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75, "value": score,
            },
        },
        title={"text": f"Hạng {grade} — {desc}"},
        number={"suffix": " điểm"},
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=20, l=20, r=20))
    return fig
