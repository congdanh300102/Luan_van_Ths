"""
Build SEBL2026_paper_version_5 from version 4, applying the reviewer's minor
revisions. Every number written into the manuscript is read from the CSV
outputs of code/rerun_v5_grid.py, code/rerun_v5_features.py and
code/rerun_v5_descriptives.py, so the text cannot drift from the experiments.

Reviewer items addressed:
  R1  Section 4.6 claimed the LightGBM-over-XGBoost gap was confirmed, but the
      Holm-adjusted p did not reject at 5%.
  R2  Section 4.4 claimed CatBoost-none beats class-weighted LightGBM, a
      contrast Table 10 never reported (everything was tested against
      LightGBM-none).
  R3  Data Availability declared the data proprietary while embedding a public
      Google Drive link to it.
  R4  The abstract framed the result under class weighting, contradicting
      Section 4.4 and the Conclusion.
  R5  Table 3 reported only class-weighted configurations, which Table 10
      shows to be inferior for every algorithm.
  R6  Governance claims needed to be conditional on out-of-time validation;
      the Group-5 share needed an explanation.
  R7  Doubled plus-minus signs, the 12-14% / 11-14% inconsistency, the
      Section 2.3/2.4 cross-references that should point to 2.5, and "v.v".
"""
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
from docx import Document

sys.path.insert(0, str(Path(__file__).parent))
from docx_tools import set_table, set_text, image_parts_in_order  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "code" / "rerun_v5_out"
FIGS = ROOT / "conference_paper" / "figures"
SRC = ROOT / "conference_paper" / "SEBL2026_paper_version_4_Dam_Cong_Danh.docx"
DST = ROOT / "conference_paper" / "SEBL2026_paper_version_5_Dam_Cong_Danh.docx"

PRETTY = {"logistic": "Logistic Regression", "decision_tree": "Decision Tree",
          "random_forest": "Random Forest", "xgboost": "XGBoost",
          "lightgbm": "LightGBM", "catboost": "CatBoost"}
# The manuscript writes configurations as "LightGBM-none" with an en dash.
LABEL = {f"{k}_{t}": f"{v}–{t}"
         for k, v in PRETTY.items() for t in ("balanced", "none")}


def f4(x):
    return f"{x:.4f}"


def pfmt(p):
    """Format a p-value the way the manuscript's tables do."""
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
          8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def word(n):
    """Small counts are spelled out in the manuscript's prose."""
    return _WORDS.get(n, str(n))


# ---------------------------------------------------------------------------
# Load every experimental result
# ---------------------------------------------------------------------------
hold = pd.read_csv(GRID / "holdout_grid.csv").set_index("config")
cv = pd.read_csv(GRID / "repeatedcv_summary.csv").set_index("config")
tests = pd.read_csv(GRID / "paired_tests.csv").set_index("comparator")
extra = pd.read_csv(GRID / "extra_contrasts.csv")
abl = pd.read_csv(GRID / "dtm_ablation.csv")
d0 = pd.read_csv(GRID / "d0_sensitivity.csv")
emb = pd.read_csv(GRID / "embedded_topk.csv").set_index("treatment")
tab1 = pd.read_csv(GRID / "table1_distribution.csv", index_col=0)

best_cv = cv["macro_f1_mean"].idxmax()
best_hold = hold["f1_macro"].idxmax()


def contrast(a, b):
    row = extra[(extra.config_a == a) & (extra.config_b == b)]
    return row.iloc[0]


def per_class(cfg):
    df = pd.read_csv(GRID / f"perclass_{cfg}.csv", index_col=0)
    return df


def confusion(cfg):
    return pd.read_csv(GRID / f"confusion_{cfg}.csv", index_col=0)


shutil.copy(SRC, DST)
doc = Document(DST)
P = doc.paragraphs
T = doc.tables


def edit(i, new, expect=None):
    """Replace paragraph i, asserting it still starts with `expect`."""
    if expect is not None:
        actual = P[i].text.strip()
        assert actual.startswith(expect), (
            f"paragraph {i} moved: expected {expect!r}, found {actual[:90]!r}")
    set_text(P[i], new)


# ===========================================================================
# ABSTRACT  (R4)
# ===========================================================================
weight_helps = [c for c in cv.index if c.endswith("_none")
                and cv.loc[c, "macro_f1_mean"] < cv.loc[c.replace("_none", "_balanced"), "macro_f1_mean"]]
n_hurt = 6 - len(weight_helps)

# Configurations the corrected, Holm-adjusted test cannot separate from the
# leading one. These are the paper's joint-best set: naming a single winner
# among them would be exactly the over-claim Section 4.6 warns against.
tied = [best_cv] + [c for c in tests.index if tests.loc[c, "p_holm"] >= 0.05]
tied = sorted(tied, key=lambda c: -cv.loc[c, "macro_f1_mean"])
tied_lo = cv.loc[tied, "macro_f1_mean"].min()
tied_hi = cv.loc[tied, "macro_f1_mean"].max()
tied_txt = ", ".join(LABEL[c] for c in tied)
all_unweighted = all(c.endswith("_none") for c in tied)

edit(9, (
    "Robust, transparent credit-risk classification is a governance precondition for "
    "responsible private-sector lending in Vietnam. Using 27,001 loan contracts from a "
    "Vietnamese commercial bank, this paper builds a machine-learning pipeline for five-group "
    "debt classification. After data preparation with Information Value screening, an explicit "
    "leakage diagnostic and feature engineering, six classifiers — Logistic Regression, "
    "Decision Tree, Random Forest, XGBoost, LightGBM and CatBoost — are each evaluated "
    "under two imbalance treatments, inverse-frequency class weighting and no weighting, giving "
    "a symmetric twelve-configuration comparison on a common hold-out set and under repeated "
    "cross-validation. The central finding is that the imbalance treatment, not the choice of "
    f"algorithm, is what the evidence identifies: class weighting lowers Macro-F1 for "
    f"{word(n_hurt)} of the six classifiers, and the {word(len(tied))} leading configurations are "
    f"{'all unweighted tree ensembles' if all_unweighted else 'led by unweighted tree ensembles'} "
    f"({tied_txt}), spanning Macro-F1 {f4(tied_lo)} to {f4(tied_hi)}. A corrected resampled "
    "paired t-test with Holm adjustment cannot separate them, so this paper deliberately names no "
    "single winning algorithm, while the advantage of dropping class weighting survives the same "
    "correction for every tree-based model. Class weighting is not merely worse, however: it trades precision on the "
    "non-performing debt groups for recall, a distinction a single Macro-F1 figure conceals and "
    "that a bank must therefore decide deliberately rather than inherit from a default. "
    "We contribute a regulatory-compliant, explainable pipeline that evaluates under extreme "
    "class imbalance using Macro-F1 and a corrected resampled paired t-test with Holm "
    "adjustment on a real-world bank dataset; and we advance model governance by showing "
    "empirically that SHAP explanations are more stable than Gain-based importance across both "
    "algorithms and imbalance treatments, and by implementing validation protocols that close "
    "identified data-leakage paths. All governance claims are conditional on the out-of-time "
    "validation this single-snapshot dataset cannot supply."
), expect="Robust, transparent credit-risk")

# ===========================================================================
# INTRODUCTION  (v4 announced "four questions" but listed three)
# ===========================================================================
edit(15, (
    "This paper addresses these three requirements using a real dataset of 27,001 loan "
    "contracts from a Vietnamese commercial bank (Data_credit_rating_VN.xlsx), covering the "
    "debt-group target NHOMNOMOI (groups 1–5) and 21 raw variables describing the loan, the "
    "borrower, and the branch. Building on a broader ongoing thesis-level research programme on "
    "AI applications in credit rating, this paper isolates and reports the full pipeline and "
    "results obtained on this single dataset, reframed around the following three questions, "
    "chosen for their direct relevance to credit governance and private-sector financing:"
), expect="This paper addresses these three requirements")

edit(16, (
    "RQ1. How do standard machine-learning classifiers compare on a multi-class, severely "
    "imbalanced debt-group classification task, and how far does the ranking depend on the "
    "imbalance treatment applied, under imbalance-aware evaluation metrics?"
), expect="RQ1.")

# ===========================================================================
# 3.1  Group-5 share  (R6b)
# ===========================================================================
g5 = tab1.loc[5, "share_pct"]
g234 = tab1.loc[[2, 3, 4], "share_pct"].sum()
ratio = tab1.loc[1, "total"] / tab1.loc[3, "total"]
edit(39, (
    f"The imbalance ratio between the largest and smallest groups is approximately {ratio:.0f}:1 "
    "(Group 1 vs. Group 3), which motivates the imbalance-aware modelling and evaluation choices "
    f"described below. The shape of the distribution also deserves comment, because it is not the "
    f"shape of a normally performing active loan book. In such a book the groups decline "
    f"monotonically with severity, since loans migrate downward one step at a time and most are "
    f"resolved or written off before reaching the final group. Here the opposite holds: Group 5 "
    f"alone ({g5:.2f}%) is larger than Groups 2, 3 and 4 combined ({g234:.2f}%). The most "
    "plausible reading is that the file contains a legacy stock of loss-classified loans that "
    "have accumulated over several years and remain on the balance sheet pending recovery, "
    "write-off or off-balance-sheet transfer, rather than a normal flow of newly deteriorating "
    "credits. The maturity evidence in Table 2b supports this: the median Group-5 loan is "
    "roughly six years past its original contractual maturity date, and 92.1% of Group-5 loans "
    "are past maturity, against 44.2% in Group 2. This matters for how the modelling results "
    "below should be read. Group 5 is not a hard early-warning target but a largely settled, "
    "easily separable end state, so strong per-class performance on Group 5 (Section 4.1) should "
    "not be read as evidence that the model anticipates deterioration; the operationally "
    "demanding groups are the transitional ones, 2 to 4."
), expect="The imbalance ratio between the largest")

# ===========================================================================
# 3.2  "v.v"  (R7)
# ===========================================================================
edit(45, (
    "Some raw numeric variables (BASE_BAL and CURR_BAL in particular) show strong right skew "
    "typical of monetary variables in banking data — for instance, the standard deviation of "
    "BASE_BAL (6.40 × 10⁹) is more than ten times its mean (4.28 × 10⁸) — "
    "and elevated Interquartile-Range (IQR) outlier rates for the two balance variables "
    "(CURR_BAL: 9.12%; BASE_BAL: 7.13%), consistent with a small number of very large "
    "corporate-type exposures within an otherwise retail-dominated book."
), expect="Some raw numeric variables")

# ===========================================================================
# 3.3 and 3.5  cross-references  (R7)
# ===========================================================================
set_text(P[49], P[49].text.replace(
    "the third research gap identified in Section 2.4",
    "the third research gap identified in Section 2.5"))
set_text(P[63], P[63].text.replace(
    "checked against the leakage criterion stated in Section 2.3",
    "checked against the two-part admissibility criterion stated in Section 3.3"))

# ===========================================================================
# Copy-editing fixes carried over from version 4, plus scoping statements that
# the symmetric twelve-configuration design has outgrown.
# ===========================================================================
COPY_EDITS = [
    # Two words lost their space somewhere in version 4's editing history.
    (50, "an input must beavailable no later than",
         "an input must be available no later than"),
    (50, "a direct or derivedrestatement of NHOMNOMOI",
         "a direct or derived restatement of NHOMNOMOI"),
    (37, "(Data_credit_rating_VN, shorten by Dataset A)",
         "(Data_credit_rating_VN, shortened to Dataset A)"),
    # 9,949 / 27,001 = 36.847%, which rounds up.
    (44, "36.84% of OPEN_DATE", "36.85% of OPEN_DATE"),
    (32, "2.5.  The research gaps", "2.5 The research gaps"),
    # The Gain instability is now demonstrated across treatments as well.
    (34, "can disagree sharply with each other, a methodological caution",
         "can disagree sharply with each other, and disagree again when the imbalance "
         "treatment changes, a methodological caution"),
    # SHAP is now cross-checked on two axes, not one.
    (66, "it is also the single dominant SHAP predictor under both models tested",
         "it is also the single dominant SHAP predictor under both models and both "
         "imbalance treatments"),
    # Figure 3 plots the branch codes numerically; Section 3.6 models them as nominal.
    (59, "stand out as the darkest off-diagonal cells.",
         "stand out as the darkest off-diagonal cells. ORGNBR and PARENTORGNBR are shown "
         "here descriptively, as codes; the modelling in Section 3.6 treats them as "
         "nominal categorical variables rather than as quantities."),
    # Paragraph 78 is rewritten wholesale further down, so it is not listed here.
    (79, "are reported for both XGBoost and LightGBM throughout Section 4",
         "are reported for both XGBoost and LightGBM, and under both imbalance treatments, "
         "throughout Section 4"),
]
for _idx, _old, _new in COPY_EDITS:
    _txt = P[_idx].text
    assert _old in _txt, f"copy-edit target missing in paragraph {_idx}: {_old!r}"
    set_text(P[_idx], _txt.replace(_old, _new))

# ===========================================================================
# 3.6  hyperparameters must be stated: a depth-constrained Decision Tree now
# carries the highest repeated-CV point estimate, and a reader cannot judge
# that result without knowing the tree was pruned.
# ===========================================================================
edit(74, (
    "Six classifiers were trained and compared: Logistic Regression, Decision Tree, Random "
    "Forest, XGBoost, LightGBM and CatBoost, all trained and evaluated in a single software "
    "environment (scikit-learn 1.9.0, xgboost 3.3.0, lightgbm 4.7.0, catboost 1.2.10; full "
    "versions in the Reproducibility Statement) so that every number reported below is directly "
    "comparable. Hyperparameters were fixed a priori at the values below and were not tuned per "
    "configuration, so that the hold-out set and the cross-validation folds are never used for "
    "model selection: Logistic Regression, L2 penalty with C = 1.0 and lbfgs, max_iter = 1,000; "
    "Decision Tree, max_depth = 10 and min_samples_leaf = 10; Random Forest, 300 trees, "
    "max_depth = 15, min_samples_leaf = 5; XGBoost, 400 rounds, max_depth = 6, learning rate "
    "0.05, subsample and colsample_bytree 0.8; LightGBM, the same settings with max_depth = 8; "
    "CatBoost, 400 iterations, depth 6, learning rate 0.05. The Decision Tree is therefore a "
    "pruned tree, not an unconstrained one — a point that matters for reading Section 4.4, "
    "where it carries the highest point estimate. Because no configuration received a tuning "
    "budget the others did not, the comparison is between algorithm families at sensible "
    "defaults rather than between fully optimised models, and Section 5 notes this as a "
    "limitation. All numerical preprocessing (imputation, standardisation) was fitted on the "
    "training partition only, to prevent information from the test partition from influencing "
    "model training."
), expect="Six classifiers were trained")

# ===========================================================================
# 3.6  imbalance handling is now symmetric  (R5)
# ===========================================================================
edit(75, (
    "Class imbalance was handled in two ways, and every model was run under both. The first "
    "treatment applies inverse-frequency class weights; the second applies no weighting at all. "
    "Reporting both for all six classifiers, rather than fixing one treatment and testing the "
    "other only where a model appeared to struggle, is what makes the comparison in Table 3 and "
    "Table 10 symmetric: the imbalance treatment is part of the configuration being compared, "
    "not a fixed background assumption. Five of eleven features are categorical, including "
    "nominal branch identifiers ORGNBR and PARENTORGNBR; these are not treated as continuous "
    "quantities. For non-CatBoost models they remain label-encoded, a limitation stated in "
    "Section 5. SMOTE was rejected because interpolation between encoded category values has no "
    "contractual meaning. Models were evaluated on the stratified 20% hold-out set using four "
    "metrics:"
), expect="Class imbalance was handled")

edit(78, (
    "with K = 5 classes. Because Group 1 alone accounts for 77.65% of observations, Accuracy is "
    "reported for completeness but Macro-F1 is treated as the primary criterion, consistent with "
    "the imbalance literature reviewed in Section 2.2. Per-class Precision/Recall/F1 and the full "
    "5 × 5 confusion matrix are also reported for the best hold-out configuration "
    "(Section 4.1), "
    "because Macro-F1 alone does not show which debt groups drive a model's score — nor, as "
    "Section 4.1 shows, whether a given score was obtained through precision or through recall. "
    "Because a single stratified split can make a small, split-specific gap between two close "
    "models look more decisive than it is, the single-split comparison in Table 3 is supplemented "
    "with a repeated stratified cross-validation check (5-fold, 5 repeats, 25 train/test "
    "partitions) in which all twelve configurations are evaluated on the identical folds, so that "
    "every pairwise difference is properly paired. Table 10 reports Macro-F1 as a mean ± "
    "standard deviation for each configuration, together with a corrected resampled paired t-test "
    "(Nadeau & Bengio, 2003) comparing the top configuration against every other one — an "
    "explicit correction for the fact that folds within repeated k-fold CV share data and are not "
    "independent, so a naive paired t-test or Wilcoxon test on the same folds understates variance "
    "and overstates significance (Section 4.4)."
), expect="with K = 5 classes.")

# ===========================================================================
# TABLE 3  (R5)
# ===========================================================================
edit(83, ("Table 3. Hold-out results on the 11-feature main configuration: "
          "six classifiers × two imbalance treatments"),
     expect="Table 3.")

rows = [["Model", "Imbalance treatment", "Accuracy", "Macro-F1", "Weighted-F1", "ROC-AUC (macro)"]]
for key in ["logistic", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]:
    for tag, label in (("balanced", "Class weighting"), ("none", "No weighting")):
        cfg = f"{key}_{tag}"
        r = hold.loc[cfg]
        star = "**" if cfg == best_hold else ""
        rows.append([PRETTY[key], label,
                     f"{star}{f4(r.accuracy)}{star}", f"{star}{f4(r.f1_macro)}{star}",
                     f"{star}{f4(r.f1_weighted)}{star}", f"{star}{f4(r.roc_auc)}{star}"])
set_table(T[3], rows)

lg_n, lg_b = hold.loc["lightgbm_none"], hold.loc["lightgbm_balanced"]
dt_n = hold.loc["decision_tree_none"]
lr_n, lr_b = hold.loc["logistic_none"], hold.loc["logistic_balanced"]
cb_n, cb_b = hold.loc["catboost_none"], hold.loc["catboost_balanced"]

edit(85, (
    "Figure 5. Macro-F1 on the hold-out set for the six classifiers under both imbalance "
    "treatments (Table 3). Removing class weighting raises Macro-F1 for every tree-based model "
    "and lowers it only for Logistic Regression."
), expect="Figure 5.")

edit(86, (
    "Three results stand out. First, the imbalance treatment matters more than the choice of "
    "algorithm across most of this grid: turning class weighting off raises hold-out Macro-F1 for "
    f"all five tree-based models, by between {min(abs(hold.loc[f'{k}_none'].f1_macro - hold.loc[f'{k}_balanced'].f1_macro) for k in ['decision_tree','random_forest','xgboost','lightgbm','catboost']):.4f} "
    f"and {max(abs(hold.loc[f'{k}_none'].f1_macro - hold.loc[f'{k}_balanced'].f1_macro) for k in ['decision_tree','random_forest','xgboost','lightgbm','catboost']):.4f} "
    "Macro-F1 points, so the class-weighted comparison alone would have ranked these models on "
    "a configuration none of them performs best in. Second, Logistic Regression is the sole "
    f"exception: weighting improves it ({f4(lr_b.f1_macro)} against {f4(lr_n.f1_macro)}), and it "
    "remains far behind every tree-based configuration either way, so the general statement is "
    "that weighting is dominated for the tree ensembles rather than for all six models without "
    f"qualification. Third, the leading configuration is unweighted LightGBM "
    f"(Macro-F1 = {f4(lg_n.f1_macro)}), but unweighted Decision Tree reaches {f4(dt_n.f1_macro)} "
    "on the same split — a gap far too small for one stratified split to resolve, which is "
    "why Section 4.4 re-estimates the entire grid under repeated cross-validation rather than "
    "declaring a winner here. Accuracy is a particularly poor guide in this table: unweighted "
    f"CatBoost reaches {f4(cb_n.accuracy)} Accuracy against {f4(cb_b.accuracy)} when weighted, a "
    "gap that reflects the 77.65% Group-1 majority far more than any genuine improvement in "
    "distinguishing the risky groups."
), expect="Under the common class-weighting treatment")

# ===========================================================================
# TABLE 4  -- best configuration's confusion matrix  (R5)
# ===========================================================================
edit(87, ("Table 4. Confusion matrix and per-class metrics for the best hold-out configuration "
          "in Table 3, LightGBM without class weighting (11-feature configuration)"),
     expect="Table 4.")

cm_n, pc_n = confusion("lightgbm_none"), per_class("lightgbm_none")
rows = [["True \\ Pred", "1", "2", "3", "4", "5", "Prec.", "Recall", "F1", "Support"]]
for g in range(1, 6):
    rows.append([str(g)] + [f"{int(v):,}" for v in cm_n.loc[f"true_{g}"]] +
                [f"{pc_n.loc[str(g), 'precision']:.3f}",
                 f"{pc_n.loc[str(g), 'recall']:.3f}",
                 f"{pc_n.loc[str(g), 'f1-score']:.3f}",
                 f"{int(pc_n.loc[str(g), 'support']):,}"])
rows.append(["Macro avg", "", "", "", "", "",
             f"{pc_n.loc['macro avg', 'precision']:.3f}",
             f"{pc_n.loc['macro avg', 'recall']:.3f}",
             f"{pc_n.loc['macro avg', 'f1-score']:.3f}",
             f"{int(pc_n.loc['macro avg', 'support']):,}"])
set_table(T[4], rows)

edit(88, (
    "Rows are true debt groups, columns are predicted debt groups; e.g., row 3 shows that of the "
    f"{int(pc_n.loc['3', 'support'])} loans truly in Group 3, the model predicted "
    f"{int(cm_n.loc['true_3', 'pred_3'])} correctly and assigned "
    f"{int(cm_n.loc['true_3', 'pred_1'])} to Group 1. The class-weighted counterpart of this "
    "table, which the discussion below compares against, is reported in Appendix B (Table B1)."
), expect="Rows are true debt groups")

edit(90, (
    "Figure 6. Row-normalised confusion matrix for the best hold-out configuration (Table 4). The "
    "darkest cells on the diagonal (Groups 1 and 5) show where the model is most reliable; the "
    "lighter diagonal cells for Groups 2–4, together with the heavy leakage of those rows "
    "into the Group-1 column, show where its flags require most caution."
), expect="Figure 6.")

pc_b = per_class("lightgbm_balanced")
edit(91, (
    "The macro-averaged figure hides a sharp asymmetry. The unweighted model attains macro "
    f"Precision {pc_n.loc['macro avg', 'precision']:.3f} against macro Recall "
    f"{pc_n.loc['macro avg', 'recall']:.3f}: when it flags a loan as Group 2, 3 or 4 it is "
    "usually right, but it misses most of the loans that genuinely belong there — Group-2 "
    f"Recall is only {pc_n.loc['2', 'recall']:.3f}, meaning roughly four in five special-mention "
    "loans are classified as standard. Errors are concentrated rather than spread across adjacent "
    f"groups: {int(cm_n.loc['true_3', 'pred_1'])} true Group-3 loans and "
    f"{int(cm_n.loc['true_4', 'pred_1'])} true Group-4 loans are assigned to Group 1, which is "
    "under-classification of exactly the kind a provisioning process cannot absorb quietly. "
    "Table 5 shows that this is a property of the imbalance treatment rather than of LightGBM."
), expect="Groups 2 and 3 have Precision")

# ===========================================================================
# TABLE 5  -- repurposed: per-class metrics under both treatments  (R5, R6a)
# ===========================================================================
edit(92, ("Table 5. Per-class Precision/Recall/F1 for LightGBM under both imbalance treatments "
          "(11-feature configuration, hold-out set)"),
     expect="Table 5.")

rows = [["Debt group", "Prec. (none)", "Recall (none)", "F1 (none)",
         "Prec. (weighted)", "Recall (weighted)", "F1 (weighted)"]]
names = {1: "1 (standard)", 2: "2 (special mention)", 3: "3 (sub-standard)",
         4: "4 (doubtful)", 5: "5 (loss)"}
for g in range(1, 6):
    rows.append([names[g],
                 f"{pc_n.loc[str(g), 'precision']:.3f}", f"{pc_n.loc[str(g), 'recall']:.3f}",
                 f"{pc_n.loc[str(g), 'f1-score']:.3f}",
                 f"{pc_b.loc[str(g), 'precision']:.3f}", f"{pc_b.loc[str(g), 'recall']:.3f}",
                 f"{pc_b.loc[str(g), 'f1-score']:.3f}"])
rows.append(["Macro avg",
             f"{pc_n.loc['macro avg', 'precision']:.3f}", f"{pc_n.loc['macro avg', 'recall']:.3f}",
             f"{pc_n.loc['macro avg', 'f1-score']:.3f}",
             f"{pc_b.loc['macro avg', 'precision']:.3f}", f"{pc_b.loc['macro avg', 'recall']:.3f}",
             f"{pc_b.loc['macro avg', 'f1-score']:.3f}"])
set_table(T[5], rows)

edit(93, (
    "The two configurations reach similar Macro-F1 by opposite routes. Class weighting raises "
    f"Group-2 Recall from {pc_n.loc['2', 'recall']:.3f} to {pc_b.loc['2', 'recall']:.3f} and "
    f"Group-4 Recall from {pc_n.loc['4', 'recall']:.3f} to {pc_b.loc['4', 'recall']:.3f}, while "
    f"cutting Group-2 Precision from {pc_n.loc['2', 'precision']:.3f} to "
    f"{pc_b.loc['2', 'precision']:.3f}. Neither configuration is better in the abstract: the "
    "unweighted model suits a process in which a flag triggers costly manual review and false "
    "positives waste scarce analyst time, whereas the weighted model suits a process in which "
    "missing a deteriorating loan is the more expensive error. This is a policy choice for the "
    "bank's credit-risk committee, and it should be recorded as one. Reporting Macro-F1 alone "
    "would have concealed it entirely, which is the governance point developed in Section 4.6."
), expect="Turning auto_class_weights off")

# ===========================================================================
# 4.2  ablation, reference date, embedded selection  (R7)
# ===========================================================================
abl_i = abl.set_index(["model", "treatment"])
edit(95, (
    "DAYS_TO_MATURITY ablation. Table 6 refits the 11-feature main configuration with "
    "DAYS_TO_MATURITY removed (10 features), for both XGBoost and LightGBM and under both "
    "imbalance treatments, so that the ablation is not itself conditional on a treatment the "
    "model comparison rejects."
), expect="DAYS_TO_MATURITY ablation.")

edit(96, "Table 6. Effect of removing DAYS_TO_MATURITY from the 11-feature configuration",
     expect="Table 6.")

rows = [["Model", "Treatment", "Macro-F1 (11 feat.)", "Macro-F1 (10 feat., no DTM)",
         "Change", "ROC-AUC (10 feat.)"]]
for key in ["xgboost", "lightgbm"]:
    for tag, label in (("none", "No weighting"), ("balanced", "Class weighting")):
        r = abl_i.loc[(key, tag)]
        rows.append([PRETTY[key], label, f4(r.f1_11feat), f4(r.f1_10feat),
                     f"−{r.delta:.4f} (−{abs(r['pct_change']):.1f}%)", f4(r.auc_10feat)])
set_table(T[6], rows)

cost_lo, cost_hi = abl["pct_change"].abs().min(), abl["pct_change"].abs().max()
COST_RANGE = f"{cost_lo:.0f}–{cost_hi:.0f}%"
edit(97, (
    f"Removing the leading SHAP variable costs {COST_RANGE} of Macro-F1 across the four "
    "model/treatment combinations, establishing a non-redundant predictive contribution that no "
    "remaining feature reproduces. It does not establish temporal admissibility or rule out proxy "
    "leakage; those questions require the bank's actual snapshot date and classification "
    "methodology."
), expect="Removing the leading SHAP variable")

d0n = d0[d0.treatment == "none"]
# The invariance claim below is only worth making if it actually holds; fail
# the build rather than print a sentence the data does not support.
for _t, _g in d0.groupby("treatment"):
    assert _g.f1_macro.round(4).nunique() == 1, (
        f"d0 sensitivity is not invariant under {_t}: {_g.f1_macro.tolist()}")
edit(98, (
    "Reference-date invariance. Only DAYS_TO_MATURITY, not LOAN_TENURE_DAYS, depends on d0. "
    "Changing d0 adds one constant to every record and therefore leaves the tree partitions "
    "unchanged: across the five candidate reference dates from 31 December 2020 to 31 December "
    f"2023, Macro-F1 is identical to four decimal places under both treatments "
    f"({f4(d0n.f1_macro.iloc[0])} unweighted; ROC-AUC {f4(d0n.roc_auc.iloc[0])}). This rules out "
    "tuning the numerical value of d0 to inflate tree performance, but does not establish the "
    "true snapshot date, temporal availability, or the absence of proxy leakage."
), expect="Reference-date invariance.")

en, eb = emb.loc["none"], emb.loc["balanced"]
edit(99, (
    "Embedded (in-fold) feature selection. The LightGBM ranking was recomputed inside every "
    "training fold, with k = 9 fixed before this verification, so the selected subset can never "
    "have been informed by the fold's test partition. Under no weighting the embedded result is "
    f"{f4(en.embedded_k9_mean)} ± {f4(en.embedded_k9_std)} against "
    f"{f4(en.fixed_11_mean)} ± {f4(en.fixed_11_std)} for the fixed 11-feature "
    f"configuration; under class weighting, {f4(eb.embedded_k9_mean)} ± "
    f"{f4(eb.embedded_k9_std)} against {f4(eb.fixed_11_mean)} ± {f4(eb.fixed_11_std)}. "
    "Selecting features honestly in-fold therefore costs little in either treatment."
), expect="Embedded (in-fold) feature selection.")

# ===========================================================================
# TABLE 7  -- top-k under the leading treatment
# ===========================================================================
edit(100, ("Table 7. Macro-F1 and ROC-AUC by number of top-k features (13-variable pool, "
           "no class weighting), XGBoost vs. LightGBM ranking"),
     expect="Table 7.")

tk_x = pd.read_csv(GRID / "topk_xgboost_none.csv").set_index("k")
tk_l = pd.read_csv(GRID / "topk_lightgbm_none.csv").set_index("k")
tk_xb = pd.read_csv(GRID / "topk_xgboost_balanced.csv").set_index("k")
tk_lb = pd.read_csv(GRID / "topk_lightgbm_balanced.csv").set_index("k")
rows = [["k", "XGB Macro-F1", "XGB ROC-AUC", "LGBM Macro-F1", "LGBM ROC-AUC"]]
for k in [3, 5, 7, 9, 11, 13]:
    label = "13 (full pool)" if k == 13 else str(k)
    rows.append([label, f4(tk_x.loc[k, "f1_macro"]), f4(tk_x.loc[k, "roc_auc"]),
                 f4(tk_l.loc[k, "f1_macro"]), f4(tk_l.loc[k, "roc_auc"])])
set_table(T[7], rows)

edit(102, ("Figure 7. Macro-F1 as a function of the number of top-ranked features used, "
           "XGBoost vs. LightGBM ranking (Table 7), without class weighting."),
     expect="Figure 7.")

def recovered(tbl, k):
    """Share of a curve's own best Macro-F1 that k features recover."""
    return tbl.loc[k, "f1_macro"] / tbl["f1_macro"].max() * 100


def rng(values, unit="%"):
    """Format a min-max range, collapsing to one value when they coincide."""
    lo, hi = min(values), max(values)
    return f"{lo:.0f}{unit}" if round(lo) == round(hi) else f"{lo:.0f}–{hi:.0f}{unit}"


r9x, r9l = recovered(tk_x, 9), recovered(tk_l, 9)
r9xb, r9lb = recovered(tk_xb, 9), recovered(tk_lb, 9)
r3x, r3l = recovered(tk_x, 3), recovered(tk_l, 3)
edit(103, (
    f"Nine of the thirteen features recover {rng([r9x, r9l])} of each model's best hold-out "
    f"Macro-F1 without class weighting, and {rng([r9xb, r9lb])} under class weighting, so the "
    "feature-efficiency conclusion does not depend on the imbalance treatment. The unweighted "
    "curve is flatter still: even three features — DAYS_TO_MATURITY, LOAN_TENURE_DAYS and "
    f"one product or utilization variable — already recover {rng([r3x, r3l])}. Two "
    "qualifications belong with that number. The curve is not monotone, because each k refits a "
    "different model rather than pruning one, so small differences between adjacent k are noise "
    "rather than evidence of an optimum; and embedded in-fold verification was performed only "
    "for LightGBM. The curve therefore identifies a broad efficient region, not a unique best k, "
    "and the governance argument is for dropping features that are free to drop rather than for "
    "any specific count."
), expect="Nine features recover")

# ===========================================================================
# TABLES 8 and 9  -- Gain vs SHAP, both models and both treatments
# ===========================================================================
GROUPS_EN = {
    "Kỳ hạn khoản vay": "Loan tenure / maturity",
    "Dư nợ & Hạn mức sử dụng": "Balance & utilization",
    "Đơn vị / Chi nhánh": "Branch / organisational unit",
    "Sản phẩm & Mục đích vay": "Product & loan purpose",
    "Lãi suất": "Interest rate",
    "Nhân khẩu học (IV thấp)": "Demographics (low IV)",
}


def group_pcts(kind, model, tag):
    df = pd.read_csv(GRID / f"{kind}_group_{model}_{tag}.csv")
    return {GROUPS_EN.get(g, g): p for g, p in zip(df["group"], df["pct"])}


def group_table(kind, order_key):
    cols = {(m, t): group_pcts(kind, m, t)
            for m in ["xgboost", "lightgbm"] for t in ["none", "balanced"]}
    names = sorted(cols[order_key], key=lambda n: -cols[order_key][n])
    rows = [["Business group", "XGBoost (none)", "LightGBM (none)",
             "XGBoost (weighted)", "LightGBM (weighted)"]]
    for n in names:
        rows.append([n] + [f"{cols[(m, t)].get(n, 0.0):.1f}"
                           for m, t in [("xgboost", "none"), ("lightgbm", "none"),
                                        ("xgboost", "balanced"), ("lightgbm", "balanced")]])
    return rows, cols


edit(105, ("Table 8. Gain-based group importance share (%) — XGBoost vs. LightGBM, "
           "both imbalance treatments (13-variable pool)"),
     expect="Table 8.")
gain_rows, gain_cols = group_table("gain", ("lightgbm", "none"))
set_table(T[8], gain_rows)

edit(106, ("Table 9. SHAP-based group importance share (%) — XGBoost vs. LightGBM, "
           "both imbalance treatments (11-feature main configuration)"),
     expect="Table 9.")
shap_rows, shap_cols = group_table("shap", ("lightgbm", "none"))
set_table(T[9], shap_rows)


def spread(cols, name):
    vals = [c.get(name, 0.0) for c in cols.values()]
    return min(vals), max(vals)


gp_lo, gp_hi = spread(gain_cols, "Product & loan purpose")
sp_lo, sp_hi = spread(shap_cols, "Loan tenure / maturity")
sb_lo, sb_hi = spread(shap_cols, "Balance & utilization")

edit(108, ("Figure 8. Gain-based (left) vs. SHAP-based (right) group importance share, "
           "XGBoost vs. LightGBM (Tables 8–9)."),
     expect="Figure 8.")

edit(109, (
    "With LightGBM's importance_type set explicitly to \"gain\", so that both libraries report "
    "the same quantity, Gain agrees on the leading group — loan tenure/maturity — but "
    "not on anything below it. The share attributed to product and loan purpose ranges from "
    f"{gp_lo:.1f}% to {gp_hi:.1f}% across the four model/treatment combinations in Table 8, a "
    "spread wide enough to reverse its rank. SHAP is markedly more consistent: loan "
    f"tenure/maturity takes {sp_lo:.1f}–{sp_hi:.1f}% and balance/utilization "
    f"{sb_lo:.1f}–{sb_hi:.1f}% across all four combinations, preserving the same ordering "
    "in every case."
), expect="After explicitly setting LightGBM")

edit(110, (
    "This is read as evidence that Gain-based importance is not a stable property of the data "
    "— it depends materially on which tree-boosting implementation computed it and on the "
    "imbalance treatment in force — while SHAP, cross-checked across two algorithms and two "
    "treatments, is comparatively stable evidence for which information genuinely drives "
    "predictions. The practical implication for a bank is unchanged: a feature-importance-based "
    "data-investment decision should not be made from a single model's Gain ranking alone."
), expect="This is read as evidence")

# ===========================================================================
# 4.4  repeated CV and the corrected paired test  (R1, R2)
# ===========================================================================
edit(112, (
    "Table 10 reports 5-fold × 5-repeat cross-validation for all twelve configurations on "
    "the identical 25 partitions, so every comparison is properly paired. The Nadeau–Bengio "
    "correction replaces the naive variance factor 1/25 = 0.04 with 1/25 + 1/4 = 0.29; Holm "
    "adjustment controls the family of eleven comparisons against the leading configuration."
), expect="Table 10 reports")

edit(113, (f"Table 10. Repeated cross-validation (5-fold × 5-repeat, identical folds) and "
           f"corrected paired t-test against {LABEL[best_cv]}"),
     expect="Table 10.")

rows = [["Configuration", "Macro-F1 (mean ± sd)",
         f"{LABEL[best_cv]} minus comparator", "t", "Holm p"]]
order = cv["macro_f1_mean"].sort_values(ascending=False).index
for cfg in order:
    r = cv.loc[cfg]
    stat = f"{f4(r.macro_f1_mean)} ± {f4(r.macro_f1_std)}"
    if cfg == best_cv:
        rows.append([f"**{LABEL[cfg]}**", f"**{stat}**", "—", "—", "—"])
    else:
        tr = tests.loc[cfg]
        rows.append([LABEL[cfg], stat, f"+{tr.mean_diff:.4f}", f"{tr.t:.2f}", pfmt(tr.p_holm)])
set_table(T[10], rows)

sep = [c for c in tests.index if tests.loc[c, "p_holm"] >= 0.05]
sig = [c for c in tests.index if tests.loc[c, "p_holm"] < 0.05]
cb_vs_lgbmb = contrast("catboost_none", "lightgbm_balanced")
lg_vs_xg = contrast("lightgbm_none", "xgboost_none")

sep_txt = ", ".join(LABEL[c] for c in sep) if sep else "no other configuration"
own_pairs = [contrast(f"{k}_none", f"{k}_balanced")
             for k in ["decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]]
worst_own_p = max(r.p_raw for r in own_pairs)

edit(115, (
    f"The highest point estimate belongs to {LABEL[best_cv]} "
    f"({f4(cv.loc[best_cv, 'macro_f1_mean'])} ± {f4(cv.loc[best_cv, 'macro_f1_std'])}), but "
    "the reading Table 10 actually supports is narrower than a ranking. After Holm adjustment, "
    f"{word(len(sep))} of the eleven comparisons fail to reject at the 5% level ({sep_txt}), so "
    f"the {word(len(tied))} configurations spanning Macro-F1 {f4(tied_lo)} to {f4(tied_hi)} are "
    "statistically indistinguishable on this evidence. This paper therefore names no single "
    "best algorithm; a preference among these configurations has to rest on grounds other than "
    "Macro-F1 — training cost, interpretability, or the precision/recall profile in Table 5 "
    "— and stating which ground was used is itself part of the governance record. "
    f"The remaining {word(len(sig))} comparisons do reject. What survives correction is a "
    "treatment effect rather than an algorithm effect: every tree-based model beats its own "
    f"class-weighted counterpart (all five within-model contrasts reject, the weakest at "
    f"p = {pfmt(worst_own_p)}), "
    "and the class-weighted configurations occupy the bottom of the table. Because every row of "
    "Table 10 is tested against the same reference configuration, a contrast between two "
    "non-reference rows cannot be read off the table, and an earlier version of this paper "
    "inferred one that way. The specific comparison between unweighted CatBoost and "
    "class-weighted LightGBM is therefore tested directly: it gives a difference of "
    f"{abs(cb_vs_lgbmb.mean_diff):.4f} Macro-F1 in favour of "
    f"{LABEL[cb_vs_lgbmb.config_a] if cb_vs_lgbmb.mean_diff > 0 else LABEL[cb_vs_lgbmb.config_b]} "
    f"(t = {cb_vs_lgbmb.t:.2f}, p = {cb_vs_lgbmb.p_raw:.4f} unadjusted, "
    f"{cb_vs_lgbmb.p_holm_within_family:.4f} after Holm adjustment within the family of eight "
    "contrasts reported in this subsection). The direction asserted previously is therefore "
    "reproduced, but the evidence for it is weaker than a bare unadjusted p-value suggests and "
    "the claim is restated accordingly. Model ranking is inseparable from imbalance treatment, "
    "and any claim about one must name the other."
), expect="LightGBM is significantly better")

edit(117, ("Figure 9. Distribution of Macro-F1 across the 25 repeated-CV folds for all twelve "
           "configurations, grouped by imbalance treatment."),
     expect="Figure 9.")

# ===========================================================================
# 4.5  literature comparison  (R7 cross-reference)
# ===========================================================================
tree_keys = ["decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
treat_effects = [cv.loc[f"{k}_none", "macro_f1_mean"] - cv.loc[f"{k}_balanced", "macro_f1_mean"]
                 for k in tree_keys]
algo_spread = (max(cv.loc[f"{k}_none", "macro_f1_mean"] for k in tree_keys)
               - min(cv.loc[f"{k}_none", "macro_f1_mean"] for k in tree_keys))

edit(119, (
    "The results are consistent with, and extend, both the international benchmarking literature "
    "and the Vietnamese studies reviewed in Section 2.4. Consistent with Lessmann et al. (2015) "
    "and Baesens et al. (2003), every tree-based learner clearly outperforms Logistic Regression "
    "here, but their more specific caution — that no family dominates uniformly — holds "
    "in an unusually sharp form on this dataset: the highest repeated-CV Macro-F1 belongs to a "
    "depth-constrained single Decision Tree, and the three boosting and bagging ensembles are not "
    "statistically separable from it (Table 10). This is closer to Brown and Mues (2012), who "
    "found the best-performing algorithm changes with the degree of imbalance, than to any "
    "expectation that gradient boosting should lead by default. Consistent with Baesens et al. "
    "(2003), the gap between models narrows substantially on ROC-AUC relative to Macro-F1, "
    "illustrating that ranking quality and hard-classification performance diverge under class "
    "imbalance. This study adds a dimension those benchmarks largely hold fixed. The imbalance "
    f"treatment moves Macro-F1 by {min(treat_effects):.4f} to {max(treat_effects):.4f} across the "
    f"five tree-based models, while the entire spread among those five algorithms at their better "
    f"treatment is {algo_spread:.4f} — so on this dataset the treatment decision matters "
    "more than the algorithm decision, and a benchmark that fixes the treatment and ranks "
    "algorithms is measuring the smaller of the two effects. Relative to the "
    "Vietnamese studies reviewed in Section 2.4, this paper's central methodological contrast is "
    "the one developed across Sections 4.1–4.4: reporting Accuracy alongside Macro-F1, a "
    "confusion matrix (Table 4) and per-class metrics under both treatments (Table 5) shows a "
    "materially less optimistic picture of minority-class detection than Accuracy alone would "
    "suggest — precisely the gap this paper's contribution (Section 2.5) argues the existing "
    "literature has not yet closed."
), expect="The results are consistent with")

# ===========================================================================
# 4.6  governance implications  (R1, R6a, R7)
# ===========================================================================
edit(122, (
    "The results in Section 4 carry four concrete implications for how a bank should govern an "
    "AI-based credit-classification system in a way that supports sustainable finance "
    "development. They are stated here as conditional recommendations. Every empirical result "
    "behind them comes from a single institution and a single classification snapshot, evaluated "
    "by random stratified resampling of that snapshot; none of them has been validated "
    "out-of-time, on a later classification date, which is the test that would establish whether "
    "the relationships hold as the portfolio and the macroeconomic environment move. Each "
    "implication below should therefore be read as applying once out-of-time validation has been "
    "carried out and has confirmed the in-sample finding, and the ordering of the checks, rather "
    "than the specific coefficients, is what transfers."
), expect="The results in Section 4 carry four")

edit(123, (
    "First, imbalance-aware, multi-metric evaluation — including per-class figures and a "
    "confusion matrix, not only a single macro-averaged number — is a governance safeguard. "
    "Tables 4 and 5 show that two configurations of the same algorithm reach comparable Macro-F1 "
    f"({f4(hold.loc['lightgbm_none', 'f1_macro'])} against "
    f"{f4(hold.loc['lightgbm_balanced', 'f1_macro'])}) "
    "while behaving in opposite ways: unweighted LightGBM flags Group-2 loans with "
    f"Precision {pc_n.loc['2', 'precision']:.3f} but Recall only {pc_n.loc['2', 'recall']:.3f}, "
    f"whereas the class-weighted configuration reaches Recall {pc_b.loc['2', 'recall']:.3f} at "
    f"Precision {pc_b.loc['2', 'precision']:.3f}. A bank choosing between them on Macro-F1 alone "
    "would be choosing blind between a model that rarely raises a false alarm and one that rarely "
    "misses a deteriorating loan. Reporting per-class Precision/Recall/F1 and the confusion "
    "matrix alongside Macro-F1 is a low-cost practice with a direct operational reading: it tells "
    "a credit-risk unit which specific groups a model's flags can and cannot be trusted for, "
    "rather than only whether the model is “good” in aggregate."
), expect="First, imbalance-aware")

edit(124, (
    "Second, feature efficiency may lower governance cost. Nine of thirteen features recover "
    f"{rng([r9x, r9l])} of the best hold-out Macro-F1, and the "
    "honest in-fold version of that selection costs little: "
    f"{f4(en.embedded_k9_mean)} ± {f4(en.embedded_k9_std)} against "
    f"{f4(en.fixed_11_mean)} ± {f4(en.fixed_11_std)} for the fixed 11-feature "
    "configuration. Each feature dropped is one fewer data feed to source, document, monitor for "
    "drift and explain to a supervisor. The evidence supports a flat efficiency region rather "
    "than a universal optimum, so the governance argument is for parsimony where it is free, not "
    "for a specific k."
), expect="Second, feature efficiency")

edit(125, (
    "Third, no single measure of model quality or feature importance should be trusted until "
    "checked against an independent one, and Section 4 shows this is inexpensive to do in "
    "practice. Gain-based importance alone would support different conclusions about which "
    "information group ranks second depending only on which tree-boosting library computed it "
    "and which imbalance treatment was in force (Table 8); only SHAP, checked across both "
    "algorithms and both treatments, supports “loan tenure/maturity dominates, "
    "balance/utilization second” as a property of the data (Table 9). The same discipline "
    "applies to model ranking, and here it overturns a claim this paper would otherwise have "
    f"made. On the single hold-out split, {LABEL['lightgbm_none']} leads "
    f"{LABEL['xgboost_none']} ({f4(hold.loc['lightgbm_none', 'f1_macro'])} against "
    f"{f4(hold.loc['xgboost_none', 'f1_macro'])}), which invites the conclusion that LightGBM is "
    "the better algorithm here. Under repeated cross-validation the ordering reverses "
    f"({f4(cv.loc['lightgbm_none', 'macro_f1_mean'])} against "
    f"{f4(cv.loc['xgboost_none', 'macro_f1_mean'])}), and the corrected paired test on the "
    f"difference gives p = {lg_vs_xg.p_raw:.4f} — not close to rejecting in either direction. "
    "The apparent ranking was an artefact of one split. A test that fails to reject is as much a "
    "governance result as one that rejects, and a ranking that flips between two honest "
    "evaluation designs is a signal to report both rather than to pick the flattering one."
), expect="Third, no single measure")

edit(126, (
    "Fourth, a variable flagged by an automated screen should trigger a documented, falsifiable "
    "check, not only a plausible written justification. Section 3.5's leakage diagnostic and "
    "Section 4.2's ablation and reference-date checks together test, rather than assert, that "
    "DAYS_TO_MATURITY is a genuine risk signal: a univariate rule on the variable alone "
    "reconstructs the label at only 67.3% accuracy; removing it costs "
    f"{COST_RANGE} of Macro-F1 rather than collapsing the model; and its predictive contribution "
    "is provably unaffected by the arbitrary reference-date convention used to compute it. None "
    "of these three results proves the variable is free of leakage — Section 5 restates this "
    "limitation plainly — but each is a test that could have gone the other way and did not, "
    "which is a materially stronger basis for a governance decision than a plausible narrative on "
    "its own."
), expect="Fourth, a variable flagged")

edit(127, (
    "Taken together, these points connect the technical results to the conference's "
    "private-sector-financing theme as scoped in Section 1: a credit-classification pipeline "
    "whose every non-trivial claim about model quality, feature importance and leakage risk has "
    "been checked against an independent test, rather than asserted, is a pipeline a bank can "
    "begin to govern responsibly. The qualification matters: on the evidence assembled here that "
    "verification is necessary but not sufficient, because a single-snapshot study cannot show "
    "that the relationships survive a change of classification date. Responsible governance of "
    "credit-risk classification is a precondition, not a substitute, for a bank's capacity to "
    "extend credit to private enterprises and households with confidence."
), expect="Taken together, these points")

# ===========================================================================
# CONCLUSION and LIMITATIONS
# ===========================================================================
edit(129, (
    "This study makes three principal contributions corresponding to RQ1–RQ3. First, the "
    "model comparison in RQ1 shows that algorithm performance cannot be separated from imbalance "
    "treatment, and that on this dataset the treatment is the better-identified of the two: for "
    "every tree-based classifier tested, turning class weighting off improves repeated-CV "
    f"Macro-F1 and the improvement survives Holm adjustment, whereas the {word(len(tied))} leading "
    f"configurations ({tied_txt}, Macro-F1 {f4(tied_lo)}–{f4(tied_hi)}) cannot be separated "
    f"from one another by the same corrected test. The highest point estimate is {LABEL[best_cv]} "
    f"({f4(cv.loc[best_cv, 'macro_f1_mean'])} ± {f4(cv.loc[best_cv, 'macro_f1_std'])}) and "
    f"the best hold-out configuration is {LABEL[best_hold]} "
    f"({f4(hold.loc[best_hold, 'f1_macro'])}); the paper reports both and claims neither as a "
    "winner. Logistic Regression is the one model class weighting helps, and it remains far "
    "behind either way. Second, the feature-efficiency analysis in RQ2 shows that nine-feature "
    f"configurations retain {rng([r9x, r9l])} of the best hold-out "
    "Macro-F1, with the in-fold embedded analysis recording only a small further reduction, "
    "supporting a more parsimonious and auditable specification. Third, the explainability "
    "analysis in RQ3 shows that SHAP identifies loan maturity as the leading information group "
    "consistently across two algorithms and two imbalance treatments, while consistently defined "
    "Gain agrees on the leader but not on the ranks below it — demonstrating the value of "
    "triangulating importance measures before a data-investment decision. Together, these "
    "findings provide a verification-oriented framework in which predictive performance, feature "
    "economy, and interpretability are evaluated jointly rather than treated as separate "
    "governance objectives."
), expect="This study makes three principal")

edit(130, (
    "**Limitations. **Four limitations should be mentioned for this study. First, the small "
    "number of observations in some debt groups — only 121 Group-3 loans in the test set "
    "— introduces uncertainty into the per-class estimates in Tables 4 and 5; model "
    "comparisons should therefore rely primarily on the repeated cross-validation results in "
    "Table 10, and even there several configurations are statistically indistinguishable. Second, "
    "evidence from a single institution and a single classification snapshot limits external and "
    "temporal validity, and this bounds every governance claim in Section 4.6. Because the file "
    "records no classification date, all resampling in this paper is random rather than temporal; "
    "an out-of-time evaluation, training on an earlier classification date and testing on a later "
    "one, is the necessary next step before any operational use, and the Group-5 concentration "
    "documented in Section 3.1 suggests that a legacy loss stock may make in-sample performance "
    "flattering relative to what such a test would show. Third, categorical variables are "
    "label-encoded for the non-CatBoost models, which imposes an arbitrary ordering on nominal "
    "codes such as branch identifiers; native categorical handling was verified for CatBoost only. "
    "Fourth, hyperparameters were fixed a priori rather than tuned (Section 3.6). This keeps the "
    "comparison clean, since no configuration received a search budget the others did not and no "
    "evaluation partition was consumed by model selection, but it means the study compares "
    "algorithm families at sensible defaults rather than at their respective optima. A tuning "
    "budget could plausibly reorder the four configurations that the corrected test already "
    "cannot separate; it is less likely to overturn the treatment effect, which is an order of "
    "magnitude larger and consistent in sign across all five tree-based models. Forecasting "
    "future debt-group transitions remains a separate research task requiring longitudinal data."
), expect="Limitations.")

# ===========================================================================
# DATA AVAILABILITY  (R3)  -- rewriting the paragraph also removes the
# field-code HYPERLINK runs that embedded the Google Drive URL.
# ===========================================================================
edit(132, (
    "The raw dataset (Data_credit_rating_VN.xlsx) is proprietary loan-level data obtained from a "
    "Vietnamese commercial bank under an academic research arrangement. It is not publicly "
    "released, no copy is distributed with this paper, and no public link to it is provided. "
    "Researchers with a legitimate academic interest may approach the corresponding author, who "
    "will forward the request to the source institution; any access requires that institution's "
    "written approval and an appropriate data-sharing agreement. All preprocessing, "
    "feature-engineering, modelling and evaluation code is organised as a standalone Python "
    "package (config/, src/, pages/) built on pandas, scikit-learn, imbalanced-learn, xgboost, "
    "lightgbm, catboost and shap; a fixed random_state = 42 is used for every split, resample and "
    "model fit reported in this paper. Every number in Tables 3–10 and Appendix B was "
    "regenerated for this version in a single pass of one software environment (scikit-learn "
    "1.9.0, xgboost 3.3.0, lightgbm 4.7.0, catboost 1.2.10, shap 0.52.0), with all twelve "
    "configurations evaluated on the identical 25 cross-validation partitions, so that every "
    "reported difference is properly paired and directly comparable. The scripts that produce "
    "each table, together with their console logs and CSV outputs, are retained alongside this "
    "paper for independent verification."
), expect="The raw dataset")

# ===========================================================================
# APPENDIX B
# ===========================================================================
edit(168, ("Table B1. Confusion matrix and per-class metrics for LightGBM with class weighting, "
           "11-feature main configuration"),
     expect="Table B1.")

cm_b = confusion("lightgbm_balanced")
rows = [["True \\ Pred", "1", "2", "3", "4", "5", "Prec.", "Recall", "F1", "Support"]]
for g in range(1, 6):
    rows.append([str(g)] + [f"{int(v):,}" for v in cm_b.loc[f"true_{g}"]] +
                [f"{pc_b.loc[str(g), 'precision']:.3f}",
                 f"{pc_b.loc[str(g), 'recall']:.3f}",
                 f"{pc_b.loc[str(g), 'f1-score']:.3f}",
                 f"{int(pc_b.loc[str(g), 'support']):,}"])
rows.append(["Macro avg", "", "", "", "", "",
             f"{pc_b.loc['macro avg', 'precision']:.3f}",
             f"{pc_b.loc['macro avg', 'recall']:.3f}",
             f"{pc_b.loc['macro avg', 'f1-score']:.3f}",
             f"{int(pc_b.loc['macro avg', 'support']):,}"])
set_table(T[12], rows)

edit(169, ("Table B2. SHAP individual-feature importance (mean |φⱼ| across all classes), "
           "11-feature main configuration, no class weighting"),
     expect="Table B2.")

sx = pd.read_csv(GRID / "shap_feature_xgboost_none.csv").set_index("feature")["mean_abs_shap"]
sl = pd.read_csv(GRID / "shap_feature_lightgbm_none.csv").set_index("feature")["mean_abs_shap"]
rows = [["Feature", "XGBoost", "LightGBM"]]
for feat in sl.sort_values(ascending=False).index:
    rows.append([feat, f"{sx.get(feat, float('nan')):.4f}", f"{sl[feat]:.4f}"])
set_table(T[13], rows)

# ===========================================================================
# Figures 5-9 (document images 6-10, 0-based indices 5-9)
# ===========================================================================
for idx, name in [(5, "fig5_model_comparison.png"),
                  (6, "fig8_confusion_matrix.png"),
                  (7, "fig6_topk_curve.png"),
                  (8, "fig7_gain_vs_shap.png"),
                  (9, "fig9_repeatedcv_boxplot.png")]:
    path = FIGS / name
    if path.exists():
        rid, part = image_parts_in_order(doc)[idx]
        part._blob = path.read_bytes()
        print(f"  replaced image {idx + 1} ({rid}) <- {name}")

doc.save(DST)
print(f"\nSaved {DST}")

# ---------------------------------------------------------------------------
# Post-conditions
# ---------------------------------------------------------------------------
zf = zipfile.ZipFile(DST)
raw = zf.read("word/document.xml").decode("utf-8")
for name in zf.namelist():
    if name.endswith(".xml") or name.endswith(".rels"):
        assert "drive.google" not in zf.read(name).decode("utf-8", "replace"), \
            f"Google Drive link still reachable via {name}"
assert "±±" not in raw, "doubled plus-minus still present"
assert "v.v" not in raw, "Vietnamese abbreviation still present"
print("Checks passed: no Google Drive link, no doubled +/-, no 'v.v'.")
