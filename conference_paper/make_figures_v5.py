"""
Regenerate Figures 5-9 of the SEBL2026 paper for version 5.

Unlike make_figures.py, which hardcoded results numbers copied out of the
manuscript, every value here is read from the CSV outputs of the version-5
re-run (code/rerun_v5_out/). Figures and tables therefore cannot drift apart.

Figure sizes are deliberately unchanged from make_figures.py so that the
images can be swapped into the .docx without disturbing the stored extents.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "code" / "rerun_v5_out"
OUT = ROOT / "conference_paper" / "figures"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})
PALETTE = ["#2A6F97", "#61A0AF", "#EE8434", "#C73E1D", "#8E9AAF", "#4C956C"]
NONE_C, BAL_C = PALETTE[0], PALETTE[2]

MODELS = ["logistic", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
SHORT = {"logistic": "Logistic\nRegression", "decision_tree": "Decision\nTree",
         "random_forest": "Random\nForest", "xgboost": "XGBoost",
         "lightgbm": "LightGBM", "catboost": "CatBoost"}

hold = pd.read_csv(GRID / "holdout_grid.csv").set_index("config")
cv = pd.read_csv(GRID / "repeatedcv_summary.csv").set_index("config")
perfold = pd.read_csv(GRID / "repeatedcv_perfold.csv")
tests = pd.read_csv(GRID / "paired_tests.csv").set_index("comparator")

best_cv = cv["macro_f1_mean"].idxmax()
tied = {best_cv} | {c for c in tests.index if tests.loc[c, "p_holm"] >= 0.05}

# ---------------------------------------------------------------------------
# Figure 5 -- hold-out Macro-F1 under both imbalance treatments (Table 3)
# ---------------------------------------------------------------------------
x = np.arange(len(MODELS))
w = 0.36
f1_none = [hold.loc[f"{m}_none", "f1_macro"] for m in MODELS]
f1_bal = [hold.loc[f"{m}_balanced", "f1_macro"] for m in MODELS]
acc_none = [hold.loc[f"{m}_none", "accuracy"] for m in MODELS]

fig, ax = plt.subplots(figsize=(10.5, 5.4))
b1 = ax.bar(x - w / 2, f1_none, w, label="Macro-F1, no weighting", color=NONE_C)
b2 = ax.bar(x + w / 2, f1_bal, w, label="Macro-F1, class weighting", color=BAL_C)
ax.scatter(x - w / 2, acc_none, marker="_", s=420, linewidths=2.2, color="#3B3B3B",
           zorder=5, label="Accuracy, no weighting (for contrast)")
for bars in (b1, b2):
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
ax.set_xticks(x)
ax.set_xticklabels([SHORT[m] for m in MODELS])
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score (hold-out, 5,401 loans)")
ax.set_title("Figure 5. Hold-out performance by model and imbalance treatment\n"
             "(11-feature configuration)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig5_model_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 6 (file fig8) -- confusion matrix of the best hold-out configuration
# ---------------------------------------------------------------------------
best_hold = hold["f1_macro"].idxmax()
cm = pd.read_csv(GRID / f"confusion_{best_hold}.csv", index_col=0).values.astype(float)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
labels = [f"Group {i}" for i in range(1, 6)]
nice = best_hold.replace("_none", " — no weighting").replace(
    "_balanced", " — class weighting").replace("lightgbm", "LightGBM").replace(
    "decision_tree", "Decision Tree").replace("xgboost", "XGBoost").replace(
    "random_forest", "Random Forest").replace("catboost", "CatBoost")

fig, ax = plt.subplots(figsize=(6.5, 5.8))
im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{int(cm[i, j]):,}\n({cm_pct[i, j]:.0f}%)", ha="center", va="center",
                color="white" if cm_pct[i, j] > 55 else "black", fontsize=9)
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Row-normalised share (%)")
ax.set_title(f"Figure 6. Confusion matrix\n{nice}")
fig.tight_layout()
fig.savefig(OUT / "fig8_confusion_matrix.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 7 (file fig6) -- top-k curve, no weighting (Table 7)
# ---------------------------------------------------------------------------
tk_x = pd.read_csv(GRID / "topk_xgboost_none.csv").set_index("k")
tk_l = pd.read_csv(GRID / "topk_lightgbm_none.csv").set_index("k")
emb = pd.read_csv(GRID / "embedded_topk.csv").set_index("treatment").loc["none"]
ks = list(tk_x.index)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ks, tk_x["f1_macro"], marker="o", color=PALETTE[0], linewidth=2, label="XGBoost ranking")
ax.plot(ks, tk_l["f1_macro"], marker="s", color=PALETTE[2], linewidth=2, label="LightGBM ranking")
ax.errorbar([9], [emb.embedded_k9_mean], yerr=[emb.embedded_k9_std], fmt="D",
            color=PALETTE[3], zorder=5, markersize=8, capsize=4,
            label="Embedded in-fold top-9\n(mean $\\pm$ sd, 25 folds)")
ax.set_xlabel("Number of top-$k$ features used (out of 13)")
ax.set_ylabel("Macro-F1")
ax.set_title("Figure 7. Macro-F1 vs. top-$k$ features\nXGBoost vs. LightGBM ranking, no class weighting")
ax.legend(loc="lower right", fontsize=8.5, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig6_topk_curve.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 8 (file fig7) -- Gain vs SHAP, both models and both treatments
# ---------------------------------------------------------------------------
GROUPS_EN = {
    "Kỳ hạn khoản vay": "Loan tenure /\nmaturity",
    "Dư nợ & Hạn mức sử dụng": "Balance &\nutilization",
    "Đơn vị / Chi nhánh": "Branch / org.\nunit",
    "Sản phẩm & Mục đích vay": "Product &\nloan purpose",
    "Lãi suất": "Interest\nrate",
    "Nhân khẩu học (IV thấp)": "Demographics\n(low IV)",
}


def gpct(kind, model, tag):
    df = pd.read_csv(GRID / f"{kind}_group_{model}_{tag}.csv")
    return {GROUPS_EN.get(g, g): p for g, p in zip(df["group"], df["pct"])}


series = [("xgboost", "none", "XGBoost, none", PALETTE[0], 1.0),
          ("lightgbm", "none", "LightGBM, none", PALETTE[2], 1.0),
          ("xgboost", "balanced", "XGBoost, weighted", PALETTE[0], 0.45),
          ("lightgbm", "balanced", "LightGBM, weighted", PALETTE[2], 0.45)]

shap_ref = gpct("shap", "lightgbm", "none")
names = sorted(shap_ref, key=lambda n: -shap_ref[n])
names += [n for n in gpct("gain", "lightgbm", "none") if n not in names]

y = np.arange(len(names))
h = 0.2
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, kind, title in [(axes[0], "gain", "Gain-based (shares vary by library and treatment)"),
                        (axes[1], "shap", "SHAP-based (stable across both)")]:
    for i, (m, t, lab, col, alpha) in enumerate(series):
        d = gpct(kind, m, t)
        ax.barh(y + (1.5 - i) * h, [d.get(n, 0.0) for n in names], h,
                label=lab, color=col, alpha=alpha)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Share of total importance (%)")
    ax.set_title(title, fontsize=11)
axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
fig.suptitle("Figure 8. Business feature-group importance: Gain vs. SHAP, "
             "two algorithms × two imbalance treatments", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT / "fig7_gain_vs_shap.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 9 -- repeated-CV Macro-F1 across the full 12-configuration grid
# ---------------------------------------------------------------------------
order = list(cv["macro_f1_mean"].sort_values(ascending=False).index)
disp = {c: c.rsplit("_", 1)[0].replace("_", " ").title().replace("Logistic", "Logistic Reg.")
           + "\n" + c.rsplit("_", 1)[1] for c in order}
model_color = {"lightgbm": PALETTE[0], "xgboost": PALETTE[1], "random_forest": PALETTE[2],
               "decision_tree": PALETTE[3], "catboost": PALETTE[4], "logistic": PALETTE[5]}

data = [perfold[c].values for c in order]
fig, ax = plt.subplots(figsize=(11, 5.5))
bp = ax.boxplot(data, tick_labels=[disp[c] for c in order], patch_artist=True,
                medianprops=dict(color="#C73E1D", linewidth=1.6), widths=0.55)
for patch, name in zip(bp["boxes"], order):
    patch.set_facecolor(model_color[name.rsplit("_", 1)[0]])
    patch.set_alpha(1.0 if name.endswith("_none") else 0.45)
for i, c in enumerate(order):
    ax.scatter(np.random.default_rng(i).normal(i + 1, 0.06, size=len(data[i])), data[i],
               color="black", alpha=0.35, s=8, zorder=3)
if len(tied) >= 1:
    ax.axvspan(0.5, len(tied) + 0.5, color="#EE8434", alpha=0.10, zorder=0)
    ax.text(len(tied) / 2 + 0.5, ax.get_ylim()[1],
            f"not separable after Holm adjustment (n = {len(tied)})",
            ha="center", va="top", fontsize=8.5, color="#8A4B12")
ax.tick_params(axis="x", labelsize=8)
ax.set_ylabel("Macro-F1 (25 folds, 5$\\times$5 repeated CV)")
ax.set_title("Figure 9. Repeated cross-validation Macro-F1, full model $\\times$ treatment grid\n"
             "(solid = no weighting, faded = class weighting; identical folds throughout)")
fig.tight_layout()
fig.savefig(OUT / "fig9_repeatedcv_boxplot.png")
plt.close(fig)

print("Regenerated Figures 5-9 in", OUT)
