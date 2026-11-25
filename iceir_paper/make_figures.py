"""Figure for the ICEIR 2026 paper: Recall@k (mean ± sd over 50 folds) of the main
early-warning methods. Numbers are read from code/results/iceir/m2_baselines.csv,
produced by code/src/review_iceir.py from the two real snapshots (20260430, 20260531).

Run:  python3 iceir_paper/make_figures.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT.parent / "code" / "results" / "iceir"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

INK, INK2, GRAY = "#0b0b0b", "#52514e", "#8a8984"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
})

with open(RES / "m2_baselines.csv", encoding="utf-8-sig") as f:
    rows = {r["method"]: r for r in csv.DictReader(f)}

KS = [1, 2, 5, 10]
SERIES = [  # (method in CSV, label, colour, marker)
    ("XGBoost + overdue fields", "XGBoost + overdue fields at $T$", "#2a78d6", "o"),
    ("Distance to next threshold", "Rule: distance to next DPD threshold", "#eb6834", "s"),
    ("XGBoost (20 features)", "XGBoost, 20 features (no overdue fields)", "#1baf7a", "^"),
    ("WOE-logistic scorecard", "WOE-logistic scorecard", "#4a3aa7", "D"),
]

fig, ax = plt.subplots(figsize=(6.2, 3.2))
for key, label, colour, marker in SERIES:
    r = rows[key]
    m = [100 * float(r[f"recall@{k}_mean"]) for k in KS]
    s = [100 * float(r[f"recall@{k}_std"]) for k in KS]
    ax.errorbar(KS, m, yerr=s, color=colour, marker=marker, ms=4.5, lw=1.8, capsize=2, label=label)
    dy = {"#2a78d6": 4, "#eb6834": -9}.get(colour, -3)
    ax.annotate(f"{m[-1]:.0f}%", (KS[-1], m[-1]), textcoords="offset points", xytext=(6, dy), fontsize=7, color=INK)
ax.plot([0] + KS, [0] + KS, color=GRAY, lw=1.2, ls="--", label="Random review")
ax.set_xlabel("Review budget $k$ (% of portfolio, ranked by score)")
ax.set_ylabel("Recall@$k$ (% of migrations found)")
ax.set_xlim(0, 11.3)
ax.set_ylim(0, 100)
ax.set_xticks([0] + KS)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig2_recall_at_k.png", bbox_inches="tight")
fig.savefig(OUT / "fig2_recall_at_k.pdf", bbox_inches="tight")
print("written fig2_recall_at_k")
