"""
Information Value (IV) & Weight of Evidence (WoE) — chuẩn Basel II/III
Dùng cho: lựa chọn feature, phát hiện leakage, WoE encoding.

IV thresholds (Siddiqi 2006):
  < 0.02  : Useless
  0.02–0.1: Weak
  0.1–0.3 : Medium
  0.3–0.5 : Strong
  > 0.5   : Very strong (> 1.0 → nghi ngờ leakage)
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _woe_iv_series(x: pd.Series, bad: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Tính WoE và IV cho một feature so với binary bad flag."""
    df = pd.DataFrame({"x": x, "bad": bad.astype(int)})

    if pd.api.types.is_numeric_dtype(x):
        try:
            df["bucket"] = pd.qcut(x, q=bins, duplicates="drop").astype(str)
        except Exception:
            df["bucket"] = x.astype(str)
    else:
        df["bucket"] = x.fillna("MISSING").astype(str)

    total_bad  = bad.sum()
    total_good = (bad == 0).sum()
    if total_bad == 0 or total_good == 0:
        return pd.DataFrame()

    g = df.groupby("bucket")["bad"].agg(["sum", "count"]).rename(
        columns={"sum": "n_bad", "count": "n_total"}
    )
    g["n_good"]   = g["n_total"] - g["n_bad"]
    g["pct_bad"]  = g["n_bad"]  / total_bad
    g["pct_good"] = g["n_good"] / total_good
    g = g[(g["pct_bad"] > 0) & (g["pct_good"] > 0)]
    g["woe"] = np.log(g["pct_bad"] / g["pct_good"])
    g["iv"]  = (g["pct_bad"] - g["pct_good"]) * g["woe"]
    return g.reset_index()


def compute_iv_table(df: pd.DataFrame, target_col: str,
                     good_class: int = 1, bins: int = 10) -> pd.DataFrame:
    """
    Tính IV cho tất cả features.
    good_class: nhóm nợ tốt (1 = đủ tiêu chuẩn); bad = tất cả nhóm còn lại.
    """
    bad = (df[target_col] != good_class).astype(int)
    feature_cols = [c for c in df.columns if c != target_col]

    rows = []
    for col in feature_cols:
        g = _woe_iv_series(df[col], bad, bins)
        iv = g["iv"].sum() if len(g) else 0.0
        # Xác định mức độ
        if iv < 0.02:
            level = "Useless"
        elif iv < 0.1:
            level = "Weak"
        elif iv < 0.3:
            level = "Medium"
        elif iv < 0.5:
            level = "Strong"
        elif iv < 1.0:
            level = "Very strong"
        else:
            level = "⚠️ Suspicious (leakage?)"
        rows.append({"feature": col, "iv": round(iv, 4), "level": level})

    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def plot_iv_bar(iv_table: pd.DataFrame, threshold: float = 0.02) -> go.Figure:
    """Biểu đồ IV các features, tô màu theo mức độ."""
    color_map = {
        "Useless":                 "#bdc3c7",
        "Weak":                    "#f1c40f",
        "Medium":                  "#e67e22",
        "Strong":                  "#27ae60",
        "Very strong":             "#2980b9",
        "⚠️ Suspicious (leakage?)": "#e74c3c",
    }
    df = iv_table.copy()
    df["color"] = df["level"].map(color_map)

    fig = go.Figure(go.Bar(
        x=df["iv"],
        y=df["feature"],
        orientation="h",
        marker_color=df["color"],
        text=df["iv"].map(lambda v: f"{v:.4f}"),
        textposition="auto",
        customdata=df["level"],
        hovertemplate="<b>%{y}</b><br>IV = %{x:.4f}<br>Mức: %{customdata}<extra></extra>",
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Ngưỡng loại ({threshold})")
    fig.update_layout(
        title="Information Value (IV) từng feature",
        xaxis_title="IV",
        height=max(400, len(df) * 28),
        showlegend=False,
    )
    return fig


def plot_woe_bins(df: pd.DataFrame, feature: str, target_col: str,
                  good_class: int = 1, bins: int = 10) -> go.Figure:
    """Biểu đồ WoE theo từng bin của một feature."""
    bad = (df[target_col] != good_class).astype(int)
    g   = _woe_iv_series(df[feature], bad, bins)
    if g.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=g["bucket"], y=g["n_total"],
        name="Số lượng", marker_color="#bdc3c7", opacity=0.6,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=g["bucket"], y=g["woe"],
        name="WoE", mode="lines+markers",
        marker=dict(size=8), line=dict(color="#e74c3c", width=2),
    ), secondary_y=True)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=True)
    fig.update_layout(
        title=f"WoE — {feature}  (IV = {g['iv'].sum():.4f})",
        xaxis_title=feature,
        height=380,
    )
    fig.update_yaxes(title_text="Số lượng", secondary_y=False)
    fig.update_yaxes(title_text="WoE", secondary_y=True)
    return fig


def recommend_features(iv_table: pd.DataFrame,
                       min_iv: float = 0.02,
                       max_iv: float = 1.0) -> list:
    """Trả về list feature có IV trong khoảng [min_iv, max_iv]."""
    mask = (iv_table["iv"] >= min_iv) & (iv_table["iv"] <= max_iv)
    return iv_table.loc[mask, "feature"].tolist()
