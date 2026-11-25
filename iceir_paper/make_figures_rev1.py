"""Figure 2 of the revised ICEIR 2026 paper: Recall@k (mean ± sd over 50 customer-grouped
folds). Numbers are read from code/results/iceir/rev1/main_folds.csv, produced by
code/src/review_iceir_rev1.py (MAIN) from the two real snapshots (20260430, 20260531).

Run:  python3 iceir_paper/make_figures_rev1.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT.parent / "code" / "results" / "iceir" / "rev1"
OUT = ROOT / "figures"

INK, INK2, GRAY = "#0b0b0b", "#52514e", "#8a8984"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
})

df = pd.read_csv(RES / "main_folds.csv", encoding="utf-8-sig")
KS = [1, 2, 5, 10]
SERIES = [  # (method in CSV, label, colour, marker, label offset)
    ("XGB + overdue fields + G_T", "XGBoost + overdue fields + $G_T$ (main)", "#2a78d6", "o", 5),
    ("Rule: distance to next threshold", "Rule: distance to next DPD threshold", "#eb6834", "s", -9),
    ("XGB + overdue fields", "XGBoost + overdue fields, no $G_T$", "#4a3aa7", "D", -3),
    ("XGB, 20 features", "XGBoost, 20 fields (no overdue fields, no $G_T$)", "#1baf7a", "^", -3),
]

fig, ax = plt.subplots(figsize=(6.2, 3.2))
for key, label, colour, marker, dy in SERIES:
    g = df[df.method == key]
    m = [100 * g[f"recall@{k}"].mean() for k in KS]
    s = [100 * g[f"recall@{k}"].std() for k in KS]
    ax.errorbar(KS, m, yerr=s, color=colour, marker=marker, ms=4.5, lw=1.8, capsize=2, label=label)
ax.plot([0] + KS, [0] + KS, color=GRAY, lw=1.2, ls="--", label="Random review")
ax.set_xlabel("Review budget $k$ (% of portfolio, ranked by score)")
ax.set_ylabel("Recall@$k$ (% of migrations found)")
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 100)
ax.set_xticks([0] + KS)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig2_recall_at_k_rev1.png", bbox_inches="tight")
fig.savefig(OUT / "fig2_recall_at_k_rev1.pdf", bbox_inches="tight")
print("written fig2_recall_at_k_rev1")
