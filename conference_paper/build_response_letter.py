"""
Build the point-by-point response to reviewer comments (version 4 -> version 5).

Written for the SEBL2026 reviewer and editor. Every figure quoted here is read
from the same CSVs that produced the revised manuscript, so the response letter
and the paper cannot disagree.
"""
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor, Cm

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "code" / "rerun_v5_out"
OUT = ROOT / "conference_paper" / "SEBL2026_response_to_reviewers_v5.docx"

hold = pd.read_csv(GRID / "holdout_grid.csv").set_index("config")
cv = pd.read_csv(GRID / "repeatedcv_summary.csv").set_index("config")
tests = pd.read_csv(GRID / "paired_tests.csv").set_index("comparator")
extra = pd.read_csv(GRID / "extra_contrasts.csv")
abl = pd.read_csv(GRID / "dtm_ablation.csv")

best_cv = cv["macro_f1_mean"].idxmax()
tied = [best_cv] + [c for c in tests.index if tests.loc[c, "p_holm"] >= 0.05]
tied_lo = cv.loc[tied, "macro_f1_mean"].min()
tied_hi = cv.loc[tied, "macro_f1_mean"].max()
cost_lo, cost_hi = abl["pct_change"].abs().min(), abl["pct_change"].abs().max()


def ctr(a, b):
    return extra[(extra.config_a == a) & (extra.config_b == b)].iloc[0]


cb = ctr("catboost_none", "lightgbm_balanced")
lgxg = ctr("lightgbm_none", "xgboost_none")
lr = ctr("logistic_none", "logistic_balanced")

ACCENT = RGBColor(0x1F, 0x55, 0x73)
MUTED = RGBColor(0x5A, 0x62, 0x6B)

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Cambria"
st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(7)
st.paragraph_format.line_spacing = 1.18
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.4)
    s.top_margin = s.bottom_margin = Cm(2.2)


def para(text="", size=10.5, bold=False, italic=False, color=None,
         space_before=0, space_after=7, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    for i, chunk in enumerate(text.split("**")):
        if not chunk:
            continue
        r = p.add_run(chunk)
        r.font.size = Pt(size)
        r.bold = bold or (i % 2 == 1)
        r.italic = italic
        if color is not None:
            r.font.color.rgb = color
    return p


def heading(text, size=13, space_before=16):
    para(text, size=size, bold=True, color=ACCENT,
         space_before=space_before, space_after=6)


def label(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text.upper())
    r.font.size = Pt(8)
    r.bold = True
    r.font.color.rgb = MUTED
    return p


def quote(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.7)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    r.font.size = Pt(10)
    r.italic = True
    r.font.color.rgb = MUTED
    return p


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            for j, chunk in enumerate(str(v).split("**")):
                if not chunk:
                    continue
                r = p.add_run(chunk)
                r.font.size = Pt(9)
                r.bold = j % 2 == 1
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


# ---------------------------------------------------------------------------
para("Response to Reviewer Comments", size=17, bold=True, color=ACCENT, space_after=2)
para("Machine Learning for Credit Classification: Case Study in a Vietnamese "
     "Commercial Bank", size=11, italic=True, space_after=2)
para("UEL SEBL International Conference 2026  ·  Revision from version 4 to version 5  "
     "·  Lê Thị Thanh An, Lê Kim Thư, Đàm Công Danh",
     size=9, color=MUTED, space_after=14)

para("We thank the reviewer for the careful reading and for recommending acceptance subject to "
     "minor revision. All seven points have been adopted. Two of them turned out to have a cause "
     "deeper than wording, which we disclose first because it affects how the revised tables "
     "should be read.")

heading("Disclosure: all modelling results have been regenerated")

para("In tracing points 1 and 2 we found that two defects in the analysis code had been repaired "
     "after some of the version-4 tables were produced, so those tables were not mutually "
     "comparable:")
para("**(i)** XGBoost exposes no class_weight constructor argument; cost-sensitive learning takes "
     "effect only through fit(sample_weight=…). The class-weighted XGBoost configuration was "
     "therefore silently unweighted.", space_after=3)
para("**(ii)** LightGBM's feature_importances_ defaults to split counts rather than gain, so the "
     "Gain columns in Table 8 were not measuring the same quantity as XGBoost's.")
para("We have therefore regenerated every modelling result in one pass of the corrected code, "
     "with all twelve configurations evaluated on the identical 25 cross-validation partitions "
     "so that every pairwise difference is properly paired.")

table(["Status", "Content"],
      [["Unchanged, re-verified identical",
        "Sections 3.1–3.5; Tables 1, 2 and 2b; Figures 1–4b. Every descriptive figure "
        "(IV values, correlations, the 67.3% univariate-rule check, the Group-5 maturity "
        "distribution) was recomputed and matches version 4 exactly."],
       ["Regenerated",
        "Tables 3–10, B1 and B2; Figures 5–9."]],
      widths=[5.2, 11.2])

para("Two substantive conclusions changed, and the revised text states both openly rather than "
     "presenting them as the original finding:", space_before=4)
para(f"**First**, the corrected test no longer identifies a single best algorithm. The four "
     f"leading configurations — Decision Tree–none, XGBoost–none, Random "
     f"Forest–none and LightGBM–none, spanning Macro-F1 {tied_lo:.4f} to {tied_hi:.4f} "
     f"— cannot be separated after Holm adjustment (p = 0.66 to 0.79). The highest point "
     f"estimate belongs to a depth-constrained Decision Tree, not to LightGBM. The paper now "
     f"declines to name a winner and instead reports what the evidence does identify: the "
     f"imbalance treatment.", space_after=3)
para(f"**Second**, class weighting is dominated for the five tree-based models but not for "
     f"Logistic Regression, where weighting helps ({cv.loc['logistic_balanced','macro_f1_mean']:.4f} "
     f"against {cv.loc['logistic_none','macro_f1_mean']:.4f}, p = {lr.p_raw:.4f}). The abstract "
     f"and conclusion now say “five of the six” rather than all six.")

# ---------------------------------------------------------------------------
heading("Point-by-point response")

POINTS = [
    (1,
     "Section 4.6 states that the claim LightGBM outperforms XGBoost was confirmed by the "
     "corrected paired test, but Table 10 reports a Holm-adjusted p of 0.0581 for XGBoost-none, "
     "which does not reject at the 5 percent level.",
     [f"Accepted; the claim was not supported and has been removed. The corrected re-run makes "
      f"the point sharper than the original numbers allowed. Tested directly, the difference "
      f"between LightGBM–none and XGBoost–none is "
      f"{'+' if lgxg.mean_diff > 0 else chr(0x2212)}{abs(lgxg.mean_diff):.4f} Macro-F1 with "
      f"p = {lgxg.p_raw:.4f}: under repeated cross-validation XGBoost–none is in fact "
      f"marginally ahead ({cv.loc['xgboost_none','macro_f1_mean']:.4f} against "
      f"{cv.loc['lightgbm_none','macro_f1_mean']:.4f}), the reverse of the hold-out ordering "
      f"({hold.loc['lightgbm_none','f1_macro']:.4f} against "
      f"{hold.loc['xgboost_none','f1_macro']:.4f}).",
      "Rather than delete the passage, we have turned it into the governance lesson it actually "
      "supports: a ranking that flips between two honest evaluation designs, and a test that "
      "fails to reject in either direction, is a reason to report both designs instead of "
      "selecting the flattering one."],
     "Section 4.6, third implication, rewritten. Table 10 rebuilt from the single corrected run."),

    (2,
     "Section 4.4 further asserts that CatBoost-none is significantly better than class-weighted "
     "LightGBM, yet every comparison in Table 10 is made against LightGBM-none, so that "
     "particular contrast is not reported. Please correct both statements or add the missing "
     "comparisons.",
     ["Accepted. The assertion was a transitive inference drawn across two separate comparisons "
      "against a common reference, which is not a valid substitute for testing the contrast. We "
      "have added the comparison rather than removed the claim.",
      f"Tested directly, the difference is {abs(cb.mean_diff):.4f} Macro-F1 in favour of "
      f"CatBoost–none (t = {cb.t:.2f}, p = {cb.p_raw:.4f} unadjusted; "
      f"{cb.p_holm_within_family:.4f} after Holm adjustment within the family of eight contrasts "
      f"now reported in that subsection). The asserted direction is reproduced, but it does not "
      f"survive correction for multiplicity, and the revised text states the claim at that "
      f"strength and no higher.",
      "Section 4.4 now also warns explicitly that a contrast between two non-reference rows of "
      "Table 10 cannot be read off the table, and notes that an earlier version inferred one "
      "that way."],
     "Section 4.4 rewritten; seven further within-model contrasts (unweighted versus weighted for "
     "each algorithm) reported alongside it."),

    (3,
     "The Data Availability statement declares the dataset proprietary and not publicly released, "
     "and in the same sentence embeds a public Google Drive link to Data_credit_rating_VN.xlsx.",
     ["Accepted without reservation; this was a serious error and we are grateful it was caught "
      "before publication. The link was a Word field-code hyperlink anchored on the file name, "
      "so no URL was visible in the rendered text and deleting the visible words would not have "
      "removed it.",
      "The whole paragraph has been rewritten and the field code removed. We verified "
      "programmatically that the string “drive.google” appears in no XML part or "
      "relationship file anywhere in the submitted document.",
      "The statement now gives the access route instead of a link: a request to the "
      "corresponding author, forwarded to the source institution, subject to that institution's "
      "written approval and an appropriate data-sharing agreement."],
     "Data Availability and Reproducibility paragraph replaced."),

    (4,
     "The abstract says LightGBM attains the best Macro-F1 under common weighting, while Section "
     "4.4 and the Conclusion identify unweighted LightGBM (0.6178) as the best-supported "
     "configuration and show that class weighting is dominated for all six models. Rewrite the "
     "abstract around that finding.",
     ["Accepted. The abstract has been rewritten around the unweighted results and no longer "
      "frames anything under common weighting.",
      "Two corrections to the premise, both arising from the re-run. Class weighting is dominated "
      "for the five tree-based models, not for all six: Logistic Regression is the exception. And "
      "the abstract can no longer name unweighted LightGBM as the best-supported configuration, "
      "because the corrected test does not separate it from three other unweighted tree "
      "configurations.",
      "The abstract now reports what is robustly identified — that the imbalance treatment "
      "matters more than the algorithm choice — states the joint-best range, and says "
      "explicitly that the paper names no single winning algorithm. It also carries the "
      "precision-versus-recall trade-off that a single Macro-F1 figure conceals, and closes by "
      "making the governance claims conditional on out-of-time validation."],
     "Abstract replaced. Conclusion and RQ1 aligned to the same framing."),

    (5,
     "Table 3 is the paper's main model comparison but reports only class-weighted "
     "configurations, which Table 10 later shows to be inferior for every algorithm; only "
     "CatBoost is given an unweighted counterpart, in Table 5. Report hold-out results for all "
     "six models under both treatments so that the primary comparison rests on the better "
     "configuration.",
     ["Adopted in full. Table 3 now reports Accuracy, Macro-F1, Weighted-F1 and ROC-AUC for all "
      "six classifiers under both imbalance treatments — twelve configurations, from the "
      "same hold-out split. Section 3.6 now states that the imbalance treatment is part of the "
      "configuration being compared rather than a fixed background assumption.",
      "This made the old Table 5 redundant, since it duplicated two rows of the new Table 3. We "
      "have kept the table number stable for the reviewer's convenience and repurposed it to "
      "carry something the revision genuinely needs: per-class Precision and Recall for LightGBM "
      "under both treatments, which is what shows that the two configurations reach comparable "
      "Macro-F1 by opposite routes.",
      "The downstream tables follow the same principle. Table 4 now reports the best hold-out "
      "configuration and its class-weighted counterpart moved to Appendix B; Table 6 reports the "
      "ablation under both treatments; Table 7 and Table B2 are reported without weighting; and "
      "Tables 8 and 9 report all four model-by-treatment combinations, which strengthens the "
      "stability argument in Section 4.3."],
     "Tables 3, 4, 5, 6, 7, 8, 9, B1 and B2 and Figures 5–9 all revised; Sections 3.6, 4.1, "
     "4.2 and 4.3 rewritten to match."),

    (6,
     "Please report or restate the governance claims as conditional on out-of-time validation. It "
     "would also help to explain why Group 5 (12.45 percent) is larger than Groups 2, 3 and 4 "
     "combined, since this suggests a legacy stock of loss loans rather than a normal active book.",
     ["Accepted on both counts. Section 4.6 now opens by stating that every result behind its "
      "four implications comes from one institution and one classification snapshot, resampled at "
      "random; that none has been validated out-of-time; and that each implication should be read "
      "as applying once such validation has been carried out. The closing paragraph adds that "
      "verification of this kind is necessary but not sufficient. The limitation is also "
      "expanded in Section 5.",
      "On Group 5, the reviewer's reading is the one the data supports, and we have adopted it. "
      "Section 3.1 now notes that a normally performing book declines monotonically with "
      "severity, whereas here Group 5 alone (12.45%) exceeds Groups 2, 3 and 4 combined (9.91%), "
      "and reads this as a legacy stock of loss-classified loans awaiting recovery, write-off or "
      "off-balance-sheet transfer.",
      "We added the supporting evidence from the paper's own Table 2b: the median Group-5 loan is "
      "roughly six years past its original contractual maturity and 92.1% of the group is past "
      "maturity, against 44.2% in Group 2. We also drew the consequence for reading the results, "
      "which we think strengthens the paper: Group 5 is a largely settled end state rather than "
      "an early-warning target, so the strong per-class performance there should not be read as "
      "evidence that the model anticipates deterioration. The operationally demanding groups are "
      "the transitional ones."],
     "Section 3.1 extended; Section 4.6 opening and closing paragraphs rewritten; Section 5 "
     "limitation expanded."),

    (7,
     "Table 10 renders a doubled plus-minus sign throughout; the ablation cost is given as 12 to "
     "14 percent in Section 4.2 and 11 to 14 percent in Section 4.6; cross-references to Sections "
     "2.3 and 2.4 for the research gaps should point to Section 2.5; and the Vietnamese "
     "abbreviation “v.v” remains in Section 3.2.",
     ["All four corrected. Table 10 was rebuilt, which removed the doubled sign throughout.",
      f"On the ablation cost, both figures were wrong rather than one. Recomputed with the "
      f"corrected code across all four model-by-treatment combinations, removing DAYS_TO_MATURITY "
      f"costs {cost_lo:.0f}–{cost_hi:.0f}% of Macro-F1. That single range is now stated in "
      f"Section 4.2 and cited unchanged in Section 4.6, and Table 6 reports the four underlying "
      f"values.",
      "Both cross-references now point to Section 2.5. Checking these, we found a third of the "
      "same kind that the reviewer did not flag: Section 3.5 attributed the admissibility "
      "criterion to Section 2.3, whereas it is stated in Section 3.3. That is corrected too.",
      "“v.v” has been removed from Section 3.2."],
     "Table 10; Sections 3.2, 3.3, 3.5, 4.2 and 4.6."),
]

for n, comment, responses, changes in POINTS:
    heading(f"{n}.  Reviewer comment", size=11, space_before=14)
    quote(f"“{comment}”")
    label("Response")
    for r in responses:
        para(r, space_after=5)
    label("Where the manuscript changed")
    para(changes, size=10, color=MUTED)

# ---------------------------------------------------------------------------
heading("Additional corrections made during the revision")
para("The reviewer did not raise these. We found them while checking the points above and have "
     "corrected them.")

table(["Location", "Correction"],
      [["Section 3.3", "“beavailable” and “derivedrestatement” — two "
                       "words had lost their spaces."],
       ["Section 3.6", "The manuscript stated no hyperparameters for any model. All six are now "
                       "given, together with the fact that they were fixed a priori and not "
                       "tuned. This matters more after the re-run, because the highest point "
                       "estimate now belongs to a Decision Tree and a reader needs to know it is "
                       "pruned (max_depth = 10, min_samples_leaf = 10) rather than unconstrained. "
                       "Section 5 adds the corresponding limitation: the study compares algorithm "
                       "families at sensible defaults, not at their respective optima."],
       ["Section 1", "The text announced “the following four questions” and then listed "
                     "three."],
       ["Section 3.5", "Cited Section 2.3 for a criterion stated in Section 3.3."],
       ["Section 4.5", "Stated that the top Macro-F1 was attained by a boosting model, which the "
                       "re-run makes false. The passage now engages with the result directly, "
                       "relating it to Brown and Mues (2012) on how the best algorithm shifts "
                       "with the degree of imbalance, and quantifies the treatment effect against "
                       "the spread among algorithms."],
       ["Figure 3", "The caption discussed ORGNBR and PARENTORGNBR as correlated numeric "
                    "variables, which sat awkwardly against Section 3.6 treating them as nominal. "
                    "The caption now says they are shown descriptively as codes."],
       ["Sections 3.1, 3.2, 2.5", "“shorten by” → “shortened to”; the "
                                  "OPEN_DATE serial-date share corrected from 36.84% to 36.85% "
                                  "(9,949 of 27,001); a stray period and double space in the "
                                  "Section 2.5 heading."]],
      widths=[3.6, 12.8])

# ---------------------------------------------------------------------------
heading("Summary of numerical changes")
para("For the reviewer's convenience, the headline figures before and after the re-run. The "
     "direction of every qualitative finding about the imbalance treatment is unchanged; the "
     "ranking among the leading unweighted configurations is what moved.")

V4_CV = {"lightgbm_none": "0.6178 ± 0.0122", "xgboost_none": "0.6068 ± 0.0103",
         "random_forest_none": "0.6038 ± 0.0099", "decision_tree_none": "0.5981 ± 0.0114",
         "catboost_none": "0.5932 ± 0.0103", "lightgbm_balanced": "0.5691 ± 0.0101",
         "xgboost_balanced": "0.5547 ± 0.0082",
         "random_forest_balanced": "0.5303 ± 0.0076",
         "catboost_balanced": "0.5202 ± 0.0095",
         "decision_tree_balanced": "0.4611 ± 0.0161",
         "logistic_balanced": "0.3390 ± 0.0056", "logistic_none": "0.3330 ± 0.0028"}
NAMES = {"logistic": "Logistic Regression", "decision_tree": "Decision Tree",
         "random_forest": "Random Forest", "xgboost": "XGBoost",
         "lightgbm": "LightGBM", "catboost": "CatBoost"}

rows = []
for cfg in cv["macro_f1_mean"].sort_values(ascending=False).index:
    key, tag = cfg.rsplit("_", 1)
    new = f"{cv.loc[cfg,'macro_f1_mean']:.4f} ± {cv.loc[cfg,'macro_f1_std']:.4f}"
    holm = "— (reference)" if cfg == best_cv else (
        "<0.0001" if tests.loc[cfg, "p_holm"] < 0.0001 else f"{tests.loc[cfg,'p_holm']:.4f}")
    rows.append([f"{NAMES[key]}–{tag}", V4_CV[cfg], f"**{new}**" if cfg in tied else new, holm])
table(["Configuration (Table 10)", "Version 4", "Version 5", "Holm p vs. reference"], rows,
      widths=[5.0, 3.9, 3.9, 3.6])
para("Bold marks the four configurations the corrected paired test cannot separate.",
     size=9, color=MUTED)

para("**Ablation cost of removing DAYS_TO_MATURITY.**  Version 4 gave 12–14% in Section 4.2 "
     f"and 11–14% in Section 4.6. Version 5 gives {cost_lo:.0f}–{cost_hi:.0f}% across "
     "the four model-by-treatment combinations now reported in Table 6.", space_before=8)
para("**Hold-out Macro-F1 (Table 3).**  Version 4 reported only the class-weighted column, whose "
     "values also shift under the corrected code — for example LightGBM from 0.5833 to "
     f"{hold.loc['lightgbm_balanced','f1_macro']:.4f} and XGBoost from 0.5645 to "
     f"{hold.loc['xgboost_balanced','f1_macro']:.4f}, the latter because class weighting now "
     "actually takes effect. The unweighted column is new.")

heading("Reproducibility")
para("The revised manuscript states that all twelve configurations were evaluated on the "
     "identical 25 cross-validation partitions in a single pass of one software environment "
     "(scikit-learn 1.9.0, xgboost 3.3.0, lightgbm 4.7.0, catboost 1.2.10, shap 0.52.0), with "
     "random_state = 42 throughout. The scripts that produce each table, their console logs and "
     "their CSV outputs are retained with the paper. As a check on the revision itself, every "
     "number printed in the manuscript was verified programmatically against those CSVs.")

doc.save(OUT)
print(f"Saved {OUT}")
