"""
Regenerate English-labelled figures for the SEBL 2026 conference paper.

Part A (Fig 1-4): re-derived directly from the raw source file
code/data/raw/Data_credit_rating_VN.xlsx, replicating the cleaning /
feature-engineering steps described in the thesis (Chuong3.tex):
  - Excel serial-date fix for OPEN_DATE / NGAYDENHAN
  - LOAN_TENURE_DAYS, DAYS_TO_MATURITY (reference date 2021-12-31), UTIL_RATE (clip 0-10)

Part B (Fig 5-7): plotted from the already-verified summary numbers reported
in the paper's Tables 3-6 (model comparison, top-k curve, Gain vs SHAP group
importance) -- these are results tables, not raw data, so the numbers are
hardcoded from the paper text rather than recomputed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT = "conference_paper/figures"
PALETTE = ["#2A6F97", "#61A0AF", "#EE8434", "#C73E1D", "#8E9AAF", "#4C956C"]

# ---------------------------------------------------------------------------
# Load and clean raw data (Dataset A)
# ---------------------------------------------------------------------------
df = pd.read_excel("code/data/raw/Data_credit_rating_VN.xlsx")


def parse_date(series):
    s = series.copy()
    # numeric-looking values -> Excel serial date; else parse dd/mm/yyyy text
    is_num = pd.to_numeric(s, errors="coerce").notna()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out.loc[is_num] = pd.to_datetime(
        pd.to_numeric(s.loc[is_num]), unit="D", origin="1899-12-30"
    )
    out.loc[~is_num] = pd.to_datetime(s.loc[~is_num], format="%d/%m/%Y", errors="coerce")
    return out


df["OPEN_DATE_DT"] = parse_date(df["OPEN_DATE"])
df["NGAYDENHAN_DT"] = parse_date(df["NGAYDENHAN"])

d0 = pd.Timestamp("2021-12-31")
df["LOAN_TENURE_DAYS"] = (df["NGAYDENHAN_DT"] - df["OPEN_DATE_DT"]).dt.days.clip(lower=0)
df["DAYS_TO_MATURITY"] = (df["NGAYDENHAN_DT"] - d0).dt.days
util = np.where(df["BASE_BAL"] > 0, df["CURR_BAL"] / df["BASE_BAL"], 0.0)
df["UTIL_RATE"] = np.clip(util, 0, 10)

# ---------------------------------------------------------------------------
# Figure 1 -- Debt-group class distribution (Table 1)
# ---------------------------------------------------------------------------
counts = df["NHOMNOMOI"].value_counts().sort_index()
labels = [f"Group {g}" for g in counts.index]
shares = counts / counts.sum() * 100

fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar(labels, counts.values, color=PALETTE[0], width=0.6)
for b, s in zip(bars, shares):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{s:.1f}%",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Number of loan contracts")
ax.set_title("Figure 1. Debt-group distribution in Dataset A (n = 27,001)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.margins(y=0.12)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_class_distribution.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 -- Distribution of key numeric variables
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

ax = axes[0, 0]
vals = np.log10(df.loc[df["BASE_BAL"] > 0, "BASE_BAL"])
ax.hist(vals, bins=40, color=PALETTE[0])
ax.set_title("Original balance (BASE_BAL, log$_{10}$ scale)")
ax.set_xlabel("log$_{10}$(value)")
ax.set_ylabel("Number of contracts")

ax = axes[0, 1]
vals = np.log10(df.loc[df["CURR_BAL"] > 0, "CURR_BAL"])
ax.hist(vals, bins=40, color=PALETTE[1])
ax.set_title("Current balance (CURR_BAL, log$_{10}$ scale)")
ax.set_xlabel("log$_{10}$(value)")
ax.set_ylabel("Number of contracts")

ax = axes[1, 0]
ax.hist(df["UTIL_RATE"], bins=40, color=PALETTE[2])
ax.set_title("Utilization rate (UTIL_RATE, clipped at 10)")
ax.set_xlabel("UTIL_RATE")
ax.set_ylabel("Number of contracts")

ax = axes[1, 1]
ax.hist(df["DAYS_TO_MATURITY"], bins=40, color=PALETTE[3])
ax.set_title("Days to maturity (DAYS_TO_MATURITY)")
ax.set_xlabel("DAYS_TO_MATURITY")
ax.set_ylabel("Number of contracts")

fig.suptitle("Figure 2. Distribution of key engineered and balance variables -- Dataset A", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/fig2_distributions.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3 -- Correlation heatmap
# ---------------------------------------------------------------------------
corr_vars = ["BASE_BAL", "CURR_BAL", "LAISUAT", "ORGNBR", "PARENTORGNBR",
             "LOAN_TENURE_DAYS", "DAYS_TO_MATURITY", "UTIL_RATE"]
corr = df[corr_vars].corr(method="pearson")

fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_vars)))
ax.set_yticks(range(len(corr_vars)))
ax.set_xticklabels(corr_vars, rotation=45, ha="right")
ax.set_yticklabels(corr_vars)
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if abs(v) > 0.55 else "black", fontsize=9)
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Pearson correlation coefficient")
ax.set_title("Figure 3. Pearson correlation matrix -- Dataset A")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_correlation.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4 -- DAYS_TO_MATURITY by debt group (outliers hidden for readability)
# ---------------------------------------------------------------------------
groups = sorted(df["NHOMNOMOI"].unique())
data = [df.loc[df["NHOMNOMOI"] == g, "DAYS_TO_MATURITY"].values for g in groups]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
bp = ax.boxplot(data, tick_labels=[f"Group {g}" for g in groups], showfliers=False,
                 patch_artist=True, medianprops=dict(color="#C73E1D", linewidth=1.6))
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_ylabel("DAYS_TO_MATURITY (days)")
ax.set_title("Figure 4. DAYS_TO_MATURITY by debt group -- Dataset A\n(outliers hidden to show box shape)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_boxplot_maturity.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4b -- Share of each debt group already past original maturity
# (reviewer diagnostic for DAYS_TO_MATURITY leakage concern, Section 3.5)
# ---------------------------------------------------------------------------
past_share = (df.assign(past=lambda d: d["DAYS_TO_MATURITY"] < 0)
                .groupby("NHOMNOMOI")["past"].mean() * 100)
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar([f"Group {g}" for g in past_share.index], past_share.values,
              color=PALETTE, width=0.6)
for b, v in zip(bars, past_share.values):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Share already past original maturity (%)")
ax.set_ylim(0, 100)
ax.set_title("Figure 4b. Share of each debt group with DAYS_TO_MATURITY < 0")
fig.tight_layout()
fig.savefig(f"{OUT}/fig4b_past_maturity_share.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5 -- Model comparison (Table 3, class-weighting methodology)
# ---------------------------------------------------------------------------
models = ["Logistic\nRegression", "Decision\nTree", "Random\nForest", "XGBoost", "LightGBM", "CatBoost"]
accuracy = [0.6091, 0.6441, 0.7784, 0.8298, 0.8548, 0.7730]
macro_f1 = [0.3946, 0.4962, 0.5699, 0.5966, 0.6044, 0.5460]
weighted_f1 = [0.6886, 0.7243, 0.8229, 0.8556, 0.8713, 0.8192]
roc_auc = [0.7744, 0.8647, 0.9071, 0.9147, 0.9128, 0.9062]

x = np.arange(len(models))
w = 0.2
fig, ax = plt.subplots(figsize=(10.5, 5.4))
ax.bar(x - 1.5 * w, accuracy, w, label="Accuracy", color=PALETTE[0])
ax.bar(x - 0.5 * w, macro_f1, w, label="Macro-F1 (primary)", color=PALETTE[2])
ax.bar(x + 0.5 * w, weighted_f1, w, label="Weighted-F1", color=PALETTE[1])
ax.bar(x + 1.5 * w, roc_auc, w, label="ROC-AUC (macro)", color=PALETTE[4])
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("Figure 5. Out-of-sample model comparison -- 11-feature config., class weighting")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_model_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 6 -- Top-k feature performance curve: XGBoost vs LightGBM (Table 7)
# ---------------------------------------------------------------------------
k_vals = [3, 5, 7, 9, 11, 13]
macro_f1_xgb = [0.4819, 0.5516, 0.5903, 0.5944, 0.5899, 0.6036]
macro_f1_lgb = [0.5324, 0.5724, 0.5911, 0.5966, 0.6065, 0.6077]
embedded_k9_mean, embedded_k9_std = 0.5878, 0.0101

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_vals, macro_f1_xgb, marker="o", color=PALETTE[0], linewidth=2, label="XGBoost ranking")
ax.plot(k_vals, macro_f1_lgb, marker="s", color=PALETTE[2], linewidth=2, label="LightGBM ranking")
ax.errorbar([9], [embedded_k9_mean], yerr=[embedded_k9_std], fmt="D", color=PALETTE[3],
            zorder=5, markersize=8, capsize=4,
            label="Embedded in-fold top-9\n(mean $\\pm$ std, 25 folds)")
ax.set_xlabel("Number of top-$k$ features used (out of 13)")
ax.set_ylabel("Macro-F1")
ax.set_title("Figure 6. Macro-F1 vs. top-$k$ features\nXGBoost vs. LightGBM ranking, class weighting")
ax.legend(loc="lower right", fontsize=8.5, frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_topk_curve.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 7 -- Gain vs SHAP group importance, XGBoost vs LightGBM (Tables 8-9)
# ---------------------------------------------------------------------------
groups7 = ["Loan tenure /\nmaturity", "Balance &\nutilization", "Branch / org.\nunit",
           "Product &\nloan purpose", "Interest\nrate"]
gain_xgb = [31.7, 16.8, 11.2, 21.7, 9.0]
gain_lgb = [49.3, 24.6, 12.5, 5.7, 6.2]
shap_xgb = [42.0, 28.0, 13.9, 7.8, 8.3]
shap_lgb = [42.8, 28.7, 13.7, 8.1, 6.7]

y = np.arange(len(groups7))
h = 0.35
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

ax = axes[0]
ax.barh(y + h / 2, gain_xgb, h, label="XGBoost", color=PALETTE[0])
ax.barh(y - h / 2, gain_lgb, h, label="LightGBM", color=PALETTE[2])
ax.set_yticks(y)
ax.set_yticklabels(groups7)
ax.invert_yaxis()
ax.set_xlabel("Share of total importance (%)")
ax.set_title("Gain-based (same definition; shares vary)")
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
ax.barh(y + h / 2, shap_xgb, h, label="XGBoost", color=PALETTE[0])
ax.barh(y - h / 2, shap_lgb, h, label="LightGBM", color=PALETTE[2])
ax.set_yticks(y)
ax.set_yticklabels(groups7)
ax.invert_yaxis()
ax.set_xlabel("Share of total importance (%)")
ax.set_title("SHAP-based (consistent across models)")
ax.legend(frameon=False, fontsize=9)

fig.suptitle("Figure 7. Business feature-group importance: Gain vs. SHAP, XGBoost vs. LightGBM", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/fig7_gain_vs_shap.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 8 -- LightGBM confusion matrix (Table 4)
# ---------------------------------------------------------------------------
cm = np.array([
    [3755, 323, 54, 47, 15],
    [99, 121, 21, 15, 1],
    [24, 27, 50, 18, 2],
    [31, 13, 14, 86, 13],
    [11, 9, 4, 43, 605],
])
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
labels = [f"Group {i}" for i in range(1, 6)]

fig, ax = plt.subplots(figsize=(6.5, 5.8))
im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
for i in range(5):
    for j in range(5):
        color = "white" if cm_pct[i, j] > 55 else "black"
        ax.text(j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]:.0f}%)", ha="center", va="center",
                color=color, fontsize=9)
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Row-normalised share (%)")
ax.set_title("Figure 8. LightGBM confusion matrix\n(11-feature config., class weighting)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig8_confusion_matrix.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 9 -- Repeated-CV Macro-F1 distribution (Table 10)
# Full symmetric 6-model x 2-strategy grid (25 folds each, matched splits);
# see code/verify_full_grid.py, verify_full_grid2.py, verify_combine_grid.py.
# ---------------------------------------------------------------------------
try:
    perfold = pd.read_csv("code/verify_out/verify_combined_perfold.csv")
except FileNotFoundError:
    perfold = None

if perfold is not None:
    order = ["lightgbm_none", "xgboost_none", "random_forest_none", "decision_tree_none",
              "catboost_none", "lightgbm_balanced", "xgboost_balanced", "random_forest_balanced",
              "catboost_balanced", "decision_tree_balanced", "logistic_balanced", "logistic_none"]
    disp = {"lightgbm_none": "LightGBM\nnone", "xgboost_none": "XGBoost\nnone",
            "random_forest_none": "Random Forest\nnone", "decision_tree_none": "Decision Tree\nnone",
            "catboost_none": "CatBoost\nnone", "lightgbm_balanced": "LightGBM\nbalanced",
            "xgboost_balanced": "XGBoost\nbalanced", "random_forest_balanced": "Random Forest\nbalanced",
            "catboost_balanced": "CatBoost\nbalanced", "decision_tree_balanced": "Decision Tree\nbalanced",
            "logistic_balanced": "Logistic Reg.\nbalanced", "logistic_none": "Logistic Reg.\nnone"}
    model_color = {"lightgbm": PALETTE[0], "xgboost": PALETTE[1], "random_forest": PALETTE[2],
                   "decision_tree": PALETTE[3], "catboost": PALETTE[4], "logistic": PALETTE[5]}
    data = [perfold[c].values for c in order]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bp = ax.boxplot(data, tick_labels=[disp[c] for c in order], patch_artist=True,
                     medianprops=dict(color="#C73E1D", linewidth=1.6), widths=0.55)
    for patch, name in zip(bp["boxes"], order):
        model = name.rsplit("_", 1)[0]
        patch.set_facecolor(model_color[model])
        patch.set_alpha(1.0 if name.endswith("_none") else 0.45)
    for i, c in enumerate(order):
        ax.scatter(np.random.normal(i + 1, 0.06, size=len(data[i])), data[i],
                   color="black", alpha=0.35, s=8, zorder=3)
    ax.tick_params(axis="x", labelsize=8)
    ax.axvline(5.5, color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("Macro-F1 (25 folds, 5x5 repeated CV)")
    ax.set_title("Figure 9. Repeated cross-validation Macro-F1, full model x strategy grid\n"
                  "(solid = no weighting, faded = class weighting; LightGBM-none is best, Table 10)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig9_repeatedcv_boxplot.png")
    plt.close(fig)
else:
    print("WARNING: code/verify_out/verify_combined_perfold.csv not found, skipping Figure 9")

print("Done. Figures written to", OUT)
