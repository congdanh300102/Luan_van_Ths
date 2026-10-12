#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the SEBL 2026 conference presentation deck.

Every number in this file is transcribed from the tables of
`SEBL2026_paper_Dam_Cong_Danh.docx` (the version submitted to the conference).
The paper is the single source of truth; if a table there changes, change it
here and regenerate rather than editing the .pptx by hand.

Usage
-----
    py generate_slides.py [output.pptx]

Default output: SEBL2026_presentation_Dam_Cong_Danh.pptx (next to this file).
"""

import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.oxml.ns import qn

# --------------------------------------------------------------------------
# Design system (matches the original deck)
# --------------------------------------------------------------------------
NAVY = RGBColor(0x00, 0x2B, 0x5C)
BLUE = RGBColor(0x00, 0x70, 0xC0)
PALEBLUE = RGBColor(0xBB, 0xDB, 0xF5)
WASHBLUE = RGBColor(0xD6, 0xE8, 0xF7)
GREY = RGBColor(0x33, 0x33, 0x33)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BAND = RGBColor(0xF2, 0xF2, 0xF2)
RED = RGBColor(0xC0, 0x00, 0x00)
DARKRED = RGBColor(0x8B, 0x00, 0x00)
GREEN = RGBColor(0x00, 0x80, 0x00)
DARKGREEN = RGBColor(0x00, 0x60, 0x00)
WASHRED = RGBColor(0xFB, 0xE4, 0xE4)
HILITE = RGBColor(0xDD, 0xEE, 0xCC)

FONT = "Calibri"
TABLE_STYLE = "{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"

SLIDE_W, SLIDE_H = 13.333, 7.5
HEADER_H = 1.10


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def _set_bg(slide, rgb):
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = rgb


def textbox(slide, left, top, width, height, paras):
    """paras: list of dicts {t, size, color, bold, align, space_after, line}."""
    box = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = Inches(0.07)
    tf.margin_right = Inches(0.07)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)

    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = spec.get("align", PP_ALIGN.LEFT)
        p.space_after = Pt(spec.get("space_after", 4))
        if spec.get("line"):
            p.line_spacing = spec["line"]
        f = p.font
        f.name = FONT
        f.size = Pt(spec.get("size", 17))
        f.bold = spec.get("bold", False)
        f.italic = spec.get("italic", False)
        f.color.rgb = spec.get("color", GREY)
        if spec.get("t"):
            p.add_run().text = spec["t"]
    return box


def rect(slide, left, top, width, height, rgb):
    from pptx.enum.shapes import MSO_SHAPE

    sh = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb
    sh.line.fill.background()
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def callout(slide, left, top, width, height, paras, fill=WASHBLUE):
    """A tinted box with text inside it (box drawn first, text on top)."""
    rect(slide, left, top, width, height, fill)
    textbox(slide, left + 0.20, top + 0.12, width - 0.40, height - 0.24, paras)


def table(
    slide,
    data,
    left,
    top,
    width,
    col_widths,
    row_height=0.36,
    font_size=13,
    header_fill=NAVY,
    header_size=None,
    band=True,
    align=PP_ALIGN.CENTER,
    first_col_align=None,
    highlight_rows=(),
    highlight_fill=HILITE,
):
    """data: list of rows; each cell a str ('\\n' -> extra paragraph)."""
    rows, cols = len(data), len(data[0])
    height = row_height * rows
    gf = slide.shapes.add_table(
        rows, cols, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tbl = gf.table

    # match the original deck's table style
    styleid = tbl._tbl.find(qn("a:tblPr")).find(qn("a:tableStyleId"))
    if styleid is not None:
        styleid.text = TABLE_STYLE
    tbl.first_row = True
    tbl.horz_banding = True

    for ci, w in enumerate(col_widths):
        tbl.columns[ci].width = Inches(w)
    for r in tbl.rows:
        r.height = Inches(row_height)

    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.01)
            cell.margin_bottom = Inches(0.01)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

            if ri == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
            elif ri in highlight_rows:
                cell.fill.solid()
                cell.fill.fore_color.rgb = highlight_fill
            elif band and ri % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = BAND

            tf = cell.text_frame
            tf.word_wrap = True
            lines = str(val).split("\n")
            for li, line in enumerate(lines):
                p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
                if first_col_align is not None and ci == 0 and ri > 0:
                    p.alignment = first_col_align
                else:
                    p.alignment = align
                f = p.font
                f.name = FONT
                f.size = Pt(header_size or font_size) if ri == 0 else Pt(font_size)
                f.bold = ri == 0 or ri in highlight_rows
                f.color.rgb = WHITE if ri == 0 else GREY
                if line:
                    p.add_run().text = line
    return gf


# --------------------------------------------------------------------------
# Slide scaffolding
# --------------------------------------------------------------------------
def content_slide(prs, title, number, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, WHITE)
    rect(slide, 0, 0, SLIDE_W, HEADER_H, NAVY)
    textbox(
        slide, 0.60, 0.13, 11.30, 0.52,
        [{"t": title, "size": 28, "bold": True, "color": WHITE, "line": 1.05}],
    )
    if subtitle:
        textbox(
            slide, 0.60, 0.68, 11.30, 0.36,
            [{"t": subtitle, "size": 15, "color": PALEBLUE, "line": 1.0}],
        )
    textbox(
        slide, 12.25, 0.15, 0.85, 0.40,
        [{"t": str(number), "size": 14, "color": WHITE, "align": PP_ALIGN.RIGHT}],
    )
    return slide


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def bullets(size=16, color=GREY, space=7, line=1.24):
    def mk(*items):
        out = []
        for it in items:
            if isinstance(it, dict):
                d = {"size": size, "color": color, "space_after": space, "line": line}
                d.update(it)
                out.append(d)
            else:
                out.append(
                    {"t": it, "size": size, "color": color,
                     "space_after": space, "line": line}
                )
        return out
    return mk


# ==========================================================================
# DATA — transcribed from SEBL2026_paper_Dam_Cong_Danh.docx
# ==========================================================================

# Table 1
T_DIST = [
    ["Debt group", "Total", "Share (%)", "Train", "Test"],
    ["1 — Standard", "20,965", "77.65", "16,771", "4,194"],
    ["2 — Special mention", "1,287", "4.77", "1,030", "257"],
    ["3 — Sub-standard", "604", "2.24", "483", "121"],
    ["4 — Doubtful", "784", "2.90", "627", "157"],
    ["5 — Loss", "3,361", "12.45", "2,689", "672"],
    ["Total", "27,001", "100.00", "21,600", "5,401"],
]

# Table 2
T_IV = [
    ["Feature", "IV", "Level / decision"],
    ["DAYS_TO_MATURITY", "4.554", "High IV — reviewed, retained (available at scoring time)"],
    ["CURRMIACCTTYPCD", "1.992", "High IV — reviewed, retained"],
    ["LOAN_TENURE_DAYS", "1.941", "High IV — reviewed, retained"],
    ["MUCDICHVAY", "1.414", "High IV — reviewed, retained"],
    ["UTIL_RATE", "1.333", "High IV — reviewed, retained"],
    ["MJACCTTYPCD", "0.970", "Very strong"],
    ["LAISUAT (interest rate)", "0.785", "Very strong"],
    ["BASE_BAL", "0.536", "Very strong"],
    ["PARENTORGNBR", "0.320", "Strong"],
    ["CURR_BAL", "0.259", "Medium"],
    ["ORGNBR", "0.182", "Medium"],
    ["SEX", "0.008", "Useless — dropped from 11-feature config"],
    ["LOAIKH (customer type)", "0.000", "Useless — dropped from 11-feature config"],
]

# Table 2b
T_DTM = [
    ["Debt group", "Mean", "Median", "Min", "Max", "Past\nmaturity"],
    ["1 — Standard", "1,112.0", "224", "−3,252", "16,798", "35.1%"],
    ["2 — Special mention", "733.2", "172", "−724", "6,549", "44.2%"],
    ["3 — Sub-standard", "327.8", "−418", "−727", "6,547", "60.3%"],
    ["4 — Doubtful", "54.4", "−517", "−1,085", "6,520", "68.1%"],
    ["5 — Loss", "−1,884.8", "−2,236", "−5,513", "6,495", "92.1%"],
]

T_DTM_SMALL = [
    ["Group", "Median\nDTM", "Past\nmaturity"],
    ["1 — Standard", "224", "35.1%"],
    ["2 — Special mention", "172", "44.2%"],
    ["3 — Sub-standard", "−418", "60.3%"],
    ["4 — Doubtful", "−517", "68.1%"],
    ["5 — Loss", "−2,236", "92.1%"],
]

# Table 3  <-- the table that was wrong in the previous deck
T_MODELS = [
    ["Model", "Imbalance\ntreatment", "Accuracy", "Macro-F1", "Weighted-F1", "ROC-AUC\n(macro)"],
    ["Logistic Regression", "Class weighting", "0.6091", "0.3946", "0.6886", "0.7744"],
    ["Logistic Regression", "No weighting", "0.8752", "0.3628", "0.8298", "0.8083"],
    ["Decision Tree", "Class weighting", "0.6441", "0.4962", "0.7243", "0.8647"],
    ["Decision Tree", "No weighting", "0.9082", "0.6245", "0.8907", "0.8921"],
    ["Random Forest", "Class weighting", "0.7784", "0.5699", "0.8229", "0.9071"],
    ["Random Forest", "No weighting", "0.9080", "0.6149", "0.8908", "0.9239"],
    ["XGBoost", "Class weighting", "0.8298", "0.5966", "0.8556", "0.9147"],
    ["XGBoost", "No weighting", "0.9113", "0.6197", "0.8942", "0.9282"],
    ["LightGBM", "Class weighting", "0.8548", "0.6044", "0.8713", "0.9128"],
    ["LightGBM", "No weighting", "0.9113", "0.6248", "0.8954", "0.9246"],
    ["CatBoost", "Class weighting", "0.7730", "0.5460", "0.8192", "0.9062"],
    ["CatBoost", "No weighting", "0.9093", "0.6074", "0.8911", "0.9235"],
]

# Table 5
T_PERCLASS = [
    ["Debt group", "Prec.\n(none)", "Recall\n(none)", "F1\n(none)",
     "Prec.\n(wt.)", "Recall\n(wt.)", "F1\n(wt.)"],
    ["1 — Standard", "0.921", "0.990", "0.954", "0.958", "0.895", "0.926"],
    ["2 — Special mention", "0.603", "0.183", "0.281", "0.245", "0.471", "0.323"],
    ["3 — Sub-standard", "0.613", "0.314", "0.415", "0.350", "0.413", "0.379"],
    ["4 — Doubtful", "0.652", "0.465", "0.543", "0.411", "0.548", "0.470"],
    ["5 — Loss", "0.953", "0.909", "0.931", "0.951", "0.900", "0.925"],
    ["Macro average", "0.748", "0.572", "0.625", "0.583", "0.645", "0.604"],
]

# Table 4 / Table B1
T_CONF_NONE = [
    ["True \\ Pred", "1", "2", "3", "4", "5", "Prec.", "Recall", "F1", "n"],
    ["1", "4,153", "22", "6", "3", "10", "0.921", "0.990", "0.954", "4,194"],
    ["2", "195", "47", "10", "3", "2", "0.603", "0.183", "0.281", "257"],
    ["3", "68", "3", "38", "10", "2", "0.613", "0.314", "0.415", "121"],
    ["4", "59", "2", "7", "73", "16", "0.652", "0.465", "0.543", "157"],
    ["5", "33", "4", "1", "23", "611", "0.953", "0.909", "0.931", "672"],
]
T_CONF_WT = [
    ["True \\ Pred", "1", "2", "3", "4", "5", "Prec.", "Recall", "F1", "n"],
    ["1", "3,755", "323", "54", "47", "15", "0.958", "0.895", "0.926", "4,194"],
    ["2", "99", "121", "21", "15", "1", "0.245", "0.471", "0.323", "257"],
    ["3", "24", "27", "50", "18", "2", "0.350", "0.413", "0.379", "121"],
    ["4", "31", "13", "14", "86", "13", "0.411", "0.548", "0.470", "157"],
    ["5", "11", "9", "4", "43", "605", "0.951", "0.900", "0.925", "672"],
]

# Table 6
T_ABLATION = [
    ["Model", "Treatment", "Macro-F1\n(11 feat.)", "Macro-F1\n(10 feat., no DTM)",
     "Change", "ROC-AUC\n(10 feat.)"],
    ["XGBoost", "No weighting", "0.6197", "0.4776", "−0.1421  (−22.9%)", "0.8892"],
    ["XGBoost", "Class weighting", "0.5966", "0.4995", "−0.0971  (−16.3%)", "0.8693"],
    ["LightGBM", "No weighting", "0.6248", "0.4823", "−0.1424  (−22.8%)", "0.8826"],
    ["LightGBM", "Class weighting", "0.6044", "0.5052", "−0.0992  (−16.4%)", "0.8673"],
]

# Table 7
T_TOPK = [
    ["k", "XGB Macro-F1", "XGB ROC-AUC", "LGBM Macro-F1", "LGBM ROC-AUC"],
    ["3", "0.6043", "0.8885", "0.5909", "0.9008"],
    ["5", "0.6061", "0.9121", "0.6205", "0.9211"],
    ["7", "0.6241", "0.9227", "0.6119", "0.9215"],
    ["9", "0.6139", "0.9287", "0.6147", "0.9245"],
    ["11", "0.6197", "0.9282", "0.6234", "0.9260"],
    ["13 (full pool)", "0.6194", "0.9288", "0.6167", "0.9264"],
]

# Table 8 — NOTE: the Demographics row was missing from the previous deck
T_GAIN = [
    ["Business group", "XGB\nnone", "LGBM\nnone", "XGB\nwt.", "LGBM\nwt."],
    ["Loan tenure / maturity", "43.1%", "60.0%", "31.7%", "49.3%"],
    ["Balance & utilization", "11.2%", "18.3%", "16.8%", "24.6%"],
    ["Branch / org. unit", "7.6%", "9.1%", "11.2%", "12.5%"],
    ["Product & loan purpose", "24.7%", "7.1%", "21.7%", "5.7%"],
    ["Interest rate", "7.7%", "4.4%", "9.0%", "6.2%"],
    ["Demographics (low IV)", "5.6%", "1.1%", "9.7%", "1.6%"],
]

# Table 9
T_SHAP = [
    ["Business group", "XGB\nnone", "LGBM\nnone", "XGB\nwt.", "LGBM\nwt."],
    ["Loan tenure / maturity", "40.8%", "41.2%", "42.0%", "42.8%"],
    ["Balance & utilization", "26.1%", "27.9%", "28.0%", "28.7%"],
    ["Branch / org. unit", "14.8%", "14.1%", "13.9%", "13.7%"],
    ["Interest rate", "9.8%", "8.8%", "8.3%", "6.7%"],
    ["Product & loan purpose", "8.5%", "7.9%", "7.8%", "8.1%"],
    ["Demographics", "—", "—", "—", "—"],
]

# Table 10
T_CV = [
    ["Configuration", "Macro-F1\n(mean ± sd)", "DT–none\nminus comparator", "t", "Holm p"],
    ["Decision Tree–none", "0.6271 ± 0.0109", "—", "—", "—"],
    ["XGBoost–none", "0.6217 ± 0.0101", "+0.0053", "0.81", "0.7885"],
    ["Random Forest–none", "0.6216 ± 0.0097", "+0.0054", "0.87", "0.7885"],
    ["LightGBM–none", "0.6183 ± 0.0095", "+0.0088", "1.26", "0.6634"],
    ["CatBoost–none", "0.6042 ± 0.0116", "+0.0229", "3.42", "0.0091"],
    ["LightGBM–balanced", "0.5917 ± 0.0089", "+0.0354", "5.51", "<0.0001"],
    ["XGBoost–balanced", "0.5878 ± 0.0083", "+0.0392", "6.50", "<0.0001"],
    ["Random Forest–balanced", "0.5637 ± 0.0072", "+0.0634", "10.98", "<0.0001"],
    ["CatBoost–balanced", "0.5471 ± 0.0091", "+0.0800", "13.11", "<0.0001"],
    ["Decision Tree–balanced", "0.4990 ± 0.0144", "+0.1281", "14.24", "<0.0001"],
    ["Logistic Regression–balanced", "0.3876 ± 0.0055", "+0.2394", "41.55", "<0.0001"],
    ["Logistic Regression–none", "0.3699 ± 0.0089", "+0.2572", "31.54", "<0.0001"],
]

# Section 4.6 — the ranking that flips
T_FLIP = [
    ["Configuration", "Hold-out Macro-F1", "Repeated-CV Macro-F1"],
    ["Decision Tree–none", "0.6245   (3rd)", "0.6271 ± 0.0109   (1st)"],
    ["XGBoost–none", "0.6197   (4th)", "0.6217 ± 0.0101   (2nd)"],
    ["Random Forest–none", "0.6149   (5th)", "0.6216 ± 0.0097   (3rd)"],
    ["LightGBM–none", "0.6248   (1st)", "0.6183 ± 0.0095   (4th)"],
    ["CatBoost–none", "0.6074   (6th)", "0.6042 ± 0.0116   (5th)"],
]

# Table B2
T_SHAP_FEAT = [
    ["Feature", "XGBoost", "LightGBM"],
    ["DAYS_TO_MATURITY", "0.8665", "1.1694"],
    ["UTIL_RATE", "0.3134", "0.4065"],
    ["CURR_BAL", "0.2302", "0.3146"],
    ["LAISUAT", "0.2599", "0.3019"],
    ["ORGNBR", "0.2209", "0.3009"],
    ["LOAN_TENURE_DAYS", "0.2178", "0.2424"],
    ["BASE_BAL", "0.1508", "0.2353"],
    ["CURRMIACCTTYPCD", "0.1520", "0.1806"],
    ["PARENTORGNBR", "0.1728", "0.1803"],
    ["MUCDICHVAY", "0.0598", "0.0756"],
    ["MJACCTTYPCD", "0.0146", "0.0158"],
]

T_HYPER = [
    ["Classifier", "Fixed hyperparameters (no per-configuration tuning)"],
    ["Logistic Regression", "L2 penalty, C = 1.0, lbfgs, max_iter = 1,000"],
    ["Decision Tree", "max_depth = 10, min_samples_leaf = 10  (pruned, not unconstrained)"],
    ["Random Forest", "300 trees, max_depth = 15, min_samples_leaf = 5"],
    ["XGBoost", "400 rounds, max_depth = 6, lr = 0.05, subsample = colsample = 0.8"],
    ["LightGBM", "as XGBoost, max_depth = 8"],
    ["CatBoost", "400 iterations, depth = 6, lr = 0.05"],
]


# ==========================================================================
# Build
# ==========================================================================
def build(out_path):
    """12 presentation slides + 3 backup slides = 15 pages, sized for a 15-minute slot."""
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    blank = prs.slide_layouts[6]
    b = bullets()
    GREY6 = RGBColor(0x66, 0x66, 0x66)

    # ---------------------------------------------------------------- 1
    s = prs.slides.add_slide(blank)
    _set_bg(s, NAVY)
    textbox(s, 1.00, 1.55, 11.30, 1.60, [
        {"t": "Machine Learning for Credit Classification:",
         "size": 36, "bold": True, "color": WHITE,
         "align": PP_ALIGN.CENTER, "line": 1.20, "space_after": 2},
        {"t": "Case Study in a Vietnamese Commercial Bank",
         "size": 36, "bold": True, "color": WHITE,
         "align": PP_ALIGN.CENTER, "line": 1.20},
    ])
    textbox(s, 1.00, 3.55, 11.30, 0.45, [
        {"t": "Lê Thị Thanh An  ·  Lê Kim Thư  ·  Đàm Công Danh",
         "size": 20, "color": PALEBLUE, "align": PP_ALIGN.CENTER},
    ])
    textbox(s, 1.00, 4.10, 11.30, 0.75, [
        {"t": "University of Economics and Law, VNU-HCM", "size": 15,
         "color": RGBColor(0x88, 0xBB, 0xDD), "align": PP_ALIGN.CENTER, "space_after": 2},
        {"t": "Hanoi University of Science and Technology", "size": 15,
         "color": RGBColor(0x88, 0xBB, 0xDD), "align": PP_ALIGN.CENTER},
    ])
    rect(s, 4.00, 5.15, 5.33, 0.03, BLUE)
    textbox(s, 1.00, 5.45, 11.30, 1.00, [
        {"t": "3rd Global Conference on Sustainability in Economics, Business and Law",
         "size": 15, "color": PALEBLUE, "align": PP_ALIGN.CENTER, "space_after": 2},
        {"t": "(UEL SEBL International Conference 2026)",
         "size": 15, "color": PALEBLUE, "align": PP_ALIGN.CENTER, "space_after": 2},
        {"t": "Ho Chi Minh City  ·  October 6, 2026",
         "size": 15, "color": PALEBLUE, "align": PP_ALIGN.CENTER},
    ])
    notes(s, """
[0:25] Good morning. My name is Dam Cong Danh, from Hanoi University of Science and Technology,
presenting joint work with Le Thi Thanh An of the University of Economics and Law, and Le Kim Thu.

Our paper asks what it actually takes to govern a machine-learning credit-classification model
responsibly in a Vietnamese bank — and the answer turned out to be less about which algorithm you
pick than we expected.
""")

    # ---------------------------------------------------------------- 2
    s = content_slide(prs, "Motivation, Research Questions & Gaps", 2)
    textbox(s, 0.60, 1.22, 12.15, 0.62, b(
        {"t": "In Vietnam, debt-group classification is a regulatory obligation, not an internal "
              "modelling choice. Every loan goes into one of five groups; Groups 3–5 are "
              "non-performing; and that classification drives provisioning — and therefore how much "
              "credit reaches private borrowers.", "size": 15, "line": 1.20},
    ))
    textbox(s, 0.60, 2.18, 6.10, 0.32, b(
        {"t": "Research questions", "size": 16, "bold": True, "color": NAVY},
    ))
    textbox(s, 0.60, 2.54, 6.10, 3.15, b(
        {"t": "RQ1   How do standard ML classifiers compare on a multi-class, severely imbalanced "
              "debt-group task — and how far does the ranking depend on the imbalance treatment?",
         "size": 13, "color": NAVY, "space_after": 9, "line": 1.18},
        {"t": "RQ2   Does adding more features always improve out-of-sample performance, or can a "
              "smaller, cheaper-to-maintain set retain near-full performance?",
         "size": 13, "color": NAVY, "space_after": 9, "line": 1.18},
        {"t": "RQ3   Which business groups of information drive prediction — and is that conclusion "
              "stable across two importance measures (Gain and SHAP)?",
         "size": 13, "color": NAVY, "line": 1.18},
    ))
    rect(s, 6.90, 2.20, 0.015, 3.35, RGBColor(0xCC, 0xCC, 0xCC))
    textbox(s, 7.15, 2.18, 5.60, 0.32, b(
        {"t": "Gaps in the prior literature", "size": 16, "bold": True, "color": NAVY},
    ))
    textbox(s, 7.15, 2.54, 5.60, 3.15, b(
        {"t": "1.  Metric.  Comparisons use Accuracy as headline (e.g. 81.84% for debt-group "
              "migration) without an imbalance-robust primary metric.",
         "size": 13, "space_after": 9, "line": 1.18},
        {"t": "2.  Stability of importance.  No study checks whether its feature-importance "
              "conclusions hold across more than one algorithm or measure.",
         "size": 13, "space_after": 9, "line": 1.18},
        {"t": "3.  Verification.  None reports a leakage diagnostic for a dominant predictor, an "
              "ablation, or a sensitivity check on an arbitrary convention.",
         "size": 13, "line": 1.18},
    ))
    callout(s, 0.60, 5.82, 12.15, 1.22, b(
        {"t": "Key design principle", "size": 15, "bold": True, "color": NAVY, "space_after": 3},
        {"t": "Every non-trivial claim is cross-checked against an independent test — leakage "
              "diagnostic, ablation, reference-date check, corrected paired t-test, Gain vs. SHAP — "
              "so that governance conclusions rest on tested, not asserted, evidence.",
         "size": 14.5, "color": NAVY, "line": 1.20},
    ))
    notes(s, """
[1:20] In Vietnam this is a regulatory obligation, and it drives provisioning. So a model that is
accurate but opaque, or that has quietly learned the labelling rule rather than genuine risk
signal, is unusable in a regulated process.

Three questions. Note how RQ1 is phrased: not "which algorithm wins", but how far the ranking
depends on the imbalance treatment. That phrasing turned out to matter a great deal.

On the right, the three gaps that motivated the design. The 81.84 percent figure is from the study
closest in subject to ours — an early-warning system for debt-group migration. It is an Accuracy
figure, on data with this kind of imbalance. Frame this as "checks we thought were missing", not
as criticism — those authors may be in the room.

Read the box at the bottom out loud. It is the real contribution.
""")

    # ---------------------------------------------------------------- 3
    s = content_slide(prs, "Data & Methodology", 3,
                      "27,001 loan contracts, one Vietnamese commercial bank, 21 raw variables")
    table(s, T_DIST, 0.60, 1.26, 5.55, [1.75, 0.98, 0.98, 0.92, 0.92],
          row_height=0.33, font_size=11.5, first_col_align=PP_ALIGN.LEFT,
          highlight_rows=(6,))
    textbox(s, 0.60, 3.66, 5.55, 1.95, b(
        {"t": "•  Imbalance ≈ 35 : 1 (G1 vs. G3); 80/20 stratified split, random_state = 42",
         "size": 12, "space_after": 6, "line": 1.18},
        {"t": "•  Leakage control — NHOMNO (r = 0.98 with target) and NHOMNO_TCBS excluded as "
              "restatements of the label", "size": 12, "space_after": 6, "line": 1.18},
        {"t": "•  IV screening — 5 variables above the IV = 1.0 flag, reviewed manually and retained",
         "size": 12, "space_after": 6, "line": 1.18},
        {"t": "•  Engineered — LOAN_TENURE_DAYS, DAYS_TO_MATURITY, UTIL_RATE (d₀ = 31 Dec 2021); "
              "11 features in the main configuration", "size": 12, "line": 1.18},
    ))
    rect(s, 6.42, 1.28, 0.015, 4.40, RGBColor(0xCC, 0xCC, 0xCC))
    textbox(s, 6.68, 1.26, 6.05, 0.32, b(
        {"t": "Modelling & evaluation", "size": 15, "bold": True, "color": NAVY},
    ))
    textbox(s, 6.68, 1.62, 6.05, 4.05, b(
        {"t": "•  6 classifiers × 2 imbalance treatments = 12 symmetric configurations",
         "size": 13, "space_after": 3, "line": 1.18},
        {"t": "     LogReg, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost — each under "
              "inverse-frequency class weighting and under no weighting",
         "size": 11.5, "color": GREY6, "space_after": 7, "line": 1.15},
        {"t": "•  Hyperparameters fixed a priori — no per-configuration tuning, so no evaluation "
              "partition is consumed by model selection", "size": 13, "space_after": 7, "line": 1.18},
        {"t": "•  SMOTE rejected — interpolating between encoded category codes has no contractual "
              "meaning", "size": 13, "space_after": 7, "line": 1.18},
        {"t": "•  Metrics — Accuracy, Macro-F1 (primary), Weighted-F1, macro OvR ROC-AUC",
         "size": 13, "space_after": 7, "line": 1.18},
        {"t": "•  20% hold-out  +  5-fold × 5-repeat CV, all 12 configurations on the identical "
              "25 partitions", "size": 13, "space_after": 7, "line": 1.18},
        {"t": "•  Nadeau–Bengio corrected resampled paired t-test, with Holm adjustment",
         "size": 13, "line": 1.18},
    ))
    callout(s, 0.60, 5.82, 12.15, 1.32, b(
        {"t": "Read Group 5 with care", "size": 14.5, "bold": True, "color": RED, "space_after": 3},
        {"t": "Group 5 alone (12.45%) exceeds Groups 2 + 3 + 4 combined (9.91%) — not the shape of "
              "a normal active book. The median Group-5 loan is about six years past original "
              "maturity and 92.1% are past maturity, against 44.2% in Group 2. This is a legacy "
              "loss stock, not a flow of newly deteriorating credits — so strong Group-5 performance "
              "later is not evidence the model anticipates deterioration. The demanding groups are 2–4.",
         "size": 13, "color": GREY, "line": 1.18},
    ), fill=WASHRED)
    notes(s, """
[1:35] 27,001 loans, 21 raw variables, one bank, stratified 80/20.

Two things on the left worth pausing on. NHOMNO correlates 0.98 with the target — an alternate
debt-group field. We treat removing it as leakage control, not feature selection, because
retaining it would invalidate every downstream comparison. And IV screening flagged five variables
above 1.0: we did NOT auto-exclude them, we reviewed each against a stated criterion and documented
the decision. That distinction comes back on slide 7.

On the right the key word is symmetric: six algorithms, both treatments, all twelve, on the same
folds. Hyperparameters fixed in advance — a real limitation I own at the end, but it means no
evaluation partition was consumed by model selection.

Now the red box, and raise it here rather than let someone find it in the results. Group 5 alone
is bigger than Groups 2, 3 and 4 combined. The median Group-5 loan is six years past maturity.
This is a legacy loss stock. So when you see F1 of 0.93 on Group 5 later, that is NOT early
warning — it is a settled end state that is easy to separate. The groups that matter operationally
are 2 to 4, and that is exactly where the model is weakest.
""")

    # ---------------------------------------------------------------- 4
    s = content_slide(prs, "RQ1: Model Comparison — Hold-out Results", 4,
                      "Table 3 — 11-feature main configuration, stratified 20% hold-out")
    table(s, T_MODELS, 0.60, 1.20, 12.10, [2.45, 2.00, 1.75, 1.90, 2.00, 2.00],
          row_height=0.34, font_size=12.5, first_col_align=PP_ALIGN.LEFT,
          highlight_rows=(10,))
    textbox(s, 0.60, 5.72, 12.10, 1.55, b(
        {"t": "Turning class weighting off raises Macro-F1 for all five tree-based models, by "
              "0.0204 to 0.1283 points. Logistic Regression is the sole exception — and trails "
              "every tree configuration either way.",
         "size": 15, "bold": True, "color": BLUE, "space_after": 6, "line": 1.20},
        {"t": "LightGBM–none leads at 0.6248 — but Decision Tree–none reaches 0.6245 on the same "
              "split, too close for one split to resolve. Accuracy misleads badly here: CatBoost "
              "is 0.9093 unweighted against 0.7730 weighted, a gap that reflects the 77.65% Group-1 "
              "majority more than any genuine gain on the risky groups.",
         "size": 14, "line": 1.20},
    ))
    notes(s, """
[1:55] The core comparison. Twelve rows: six algorithms, each under both treatments.

First, the headline: the imbalance treatment matters more than the choice of algorithm. Every
tree-based model does better with weighting OFF. Counter-intuitive — class weighting is the
standard reflex for imbalanced data — and here it costs Macro-F1 in five cases out of five.

Second, Logistic Regression is the exception: weighting helps it, 0.3946 against 0.3628. But it
trails every tree configuration either way, so the honest statement is that weighting is dominated
for the tree-based models, not for all six without qualification.

Third — top of the Macro-F1 column. LightGBM-none 0.6248. Decision Tree-none 0.6245. Three
ten-thousandths apart, on one split. I am not going to call that a win; slide 9 shows why.

And compare Accuracy against Macro-F1 for CatBoost: 0.9093 against 0.6074. That is Gap 1 from the
previous slide, visible on our own data.
""")

    # ---------------------------------------------------------------- 5
    s = content_slide(prs, "RQ1: The Precision–Recall Trade-off", 5,
                      "Table 5 — LightGBM, no weighting vs. class weighting (hold-out set)")
    table(s, T_PERCLASS, 0.60, 1.30, 12.10, [2.90, 1.53, 1.53, 1.53, 1.53, 1.54, 1.54],
          row_height=0.44, font_size=13, first_col_align=PP_ALIGN.LEFT,
          highlight_rows=(6,))
    textbox(s, 0.60, 4.60, 12.10, 2.10, b(
        {"t": "•  The two configurations reach similar Macro-F1 (0.625 vs. 0.604) by opposite routes.",
         "size": 15.5, "space_after": 8, "line": 1.20},
        {"t": "•  No weighting — macro Precision 0.748 against macro Recall 0.572. When it flags a "
              "loan it is usually right, but Group-2 Recall is 0.183: roughly four in five "
              "special-mention loans are classified as standard.",
         "size": 15.5, "space_after": 8, "line": 1.20},
        {"t": "•  Class weighting — Group-2 Recall rises 0.183 → 0.471, but Group-2 Precision falls "
              "0.603 → 0.245, so about three in four flags are false.",
         "size": 15.5, "space_after": 8, "line": 1.20},
        {"t": "•  Neither is better in the abstract. This is a policy choice for the bank's "
              "credit-risk committee — and Macro-F1 alone would have concealed it entirely.",
         "size": 15.5, "bold": True, "color": NAVY, "line": 1.20},
    ))
    notes(s, """
[1:25] The most directly useful slide for any practitioner here.

Two configurations of the SAME algorithm. Macro-F1 0.625 against 0.604 — close. But look at how
they get there. Unweighted: precision 0.748, recall 0.572. It rarely raises a false alarm but
misses most of what it should catch — Group-2 recall 0.183, four out of five special-mention loans
called standard.

Weighted: the mirror image. Group-2 recall jumps to 0.471, but precision collapses to 0.245 —
three out of four flags now false.

Which is better? Neither. [pause] It depends on whether a flag triggers costly manual review, in
which case false positives burn analyst time, or whether missing a deteriorating loan is the
expensive error. That is a credit-risk committee decision, not a modelling decision — and a single
Macro-F1 number hides it completely.
""")

    # ---------------------------------------------------------------- 6
    s = content_slide(prs, "RQ2: Feature Efficiency", 6,
                      "Table 7 — top-k curve, 13-variable pool, no class weighting")
    table(s, T_TOPK, 0.60, 1.30, 7.30, [1.70, 1.40, 1.40, 1.40, 1.40],
          row_height=0.44, font_size=13, first_col_align=PP_ALIGN.LEFT)
    textbox(s, 8.15, 1.32, 4.60, 3.30, b(
        {"t": "•  9 of 13 features recover 98–99% of each model's best Macro-F1 — and 98% under "
              "class weighting too.", "size": 14.5, "space_after": 9, "line": 1.20},
        {"t": "•  Even 3 features reach 95–97%: DAYS_TO_MATURITY, LOAN_TENURE_DAYS and one product "
              "or utilization variable.", "size": 14.5, "space_after": 9, "line": 1.20},
        {"t": "•  Honest in-fold selection costs under 1% — 0.6147 ± 0.0102 against 0.6183 ± 0.0095 "
              "for the fixed 11-feature configuration.", "size": 14.5, "line": 1.20},
    ))
    callout(s, 0.60, 4.70, 12.10, 1.10, b(
        {"t": "Two qualifications belong with that number.", "size": 14.5, "bold": True,
         "color": NAVY, "space_after": 3},
        {"t": "The curve is not monotone, because each k refits a different model rather than "
              "pruning one — so small differences between adjacent k are noise, not an optimum. "
              "And embedded in-fold verification was run for LightGBM only. The curve identifies a "
              "broad efficient region, not a unique best k.",
         "size": 14, "color": NAVY, "line": 1.20},
    ))
    textbox(s, 0.60, 5.95, 12.10, 0.80, b(
        {"t": "Governance reading:  each feature dropped is one fewer data feed to source, document, "
              "monitor for drift and explain to a supervisor. The argument is for parsimony where "
              "it is free — not for a specific k.",
         "size": 15, "bold": True, "color": BLUE, "line": 1.20},
    ))
    notes(s, """
[0:55] Does more data always help? No. Thirteen features gives XGBoost 0.6194; nine gives 0.6139 —
98 and a half percent. Three features still recovers 95 to 97 percent.

Two caveats, in the blue box. The curve is not monotone — k=7 beats k=9 — because each k refits a
different model, so those wiggles are noise. Do not read an optimum off this curve; there isn't
one. And the in-fold number matters: selecting the top nine using a ranking computed over the whole
dataset means you have peeked. Recomputing inside each training fold costs under one percent.

Why a bank cares: each dropped feature is one fewer feed to source, document, monitor and explain.
""")

    # ---------------------------------------------------------------- 7
    s = content_slide(prs, "Is DAYS_TO_MATURITY Leakage? Four Tests", 7,
                      "Sections 3.5 and 4.2 — each test could have failed, and did not")
    table(s, T_DTM_SMALL, 0.60, 1.32, 4.05, [2.05, 1.00, 1.00],
          row_height=0.46, font_size=12.5, first_col_align=PP_ALIGN.LEFT)
    textbox(s, 0.60, 4.22, 4.05, 1.32, b(
        {"t": "DAYS_TO_MATURITY comes from NGAYDENHAN — the original contractual maturity date — "
              "not from a payment due date. The raw file has no days-past-due field at all. "
              "The monotone median is what raises the concern; IV = 4.554 is far above any other "
              "admissible variable.",
         "size": 12, "color": GREY6, "line": 1.18},
    ))
    textbox(s, 5.00, 1.32, 7.75, 4.30, b(
        {"t": "1.  Univariate rule.   “Predict non-performing iff DAYS_TO_MATURITY < 0” "
              "reconstructs the binary good/bad split at only 67.3% accuracy, F1 = 0.508 on the "
              "non-performing class — far from the near-perfect reconstruction that would signal "
              "algebraic equivalence. Compare NHOMNO, excluded at r = 0.98.",
         "size": 14.5, "space_after": 10, "line": 1.20},
        {"t": "2.  Boundary behaviour.   35.1% of Group-1 loans are already past original maturity "
              "yet remain classified standard — plausibly rolled-over working capital. And 7.9% of "
              "Group-5 loans are not past maturity. Not a step function at any group boundary.",
         "size": 14.5, "space_after": 10, "line": 1.20},
        {"t": "3.  Ablation.   Removing it costs 16–23% of Macro-F1 across four model × treatment "
              "combinations — a non-redundant contribution no other feature reproduces, but not the "
              "collapse a label copy would produce.",
         "size": 14.5, "space_after": 10, "line": 1.20},
        {"t": "4.  Reference-date invariance.   Macro-F1 identical to four decimal places across "
              "five values of d₀ from 31 Dec 2020 to 31 Dec 2023. The convention cannot have been "
              "tuned to flatter the model.",
         "size": 14.5, "line": 1.20},
    ))
    callout(s, 0.60, 5.62, 12.20, 1.58, b(
        {"t": "None of this proves the absence of leakage.", "size": 14.5, "bold": True,
         "color": RED, "space_after": 3},
        {"t": "The raw 21-column file contains no “days past due” field of any kind, so a residual "
              "possibility that part of the variable's power reflects an unobserved overdue field "
              "correlated with it cannot be excluded from the data alone. Confirming this against "
              "the bank's actual classification methodology is required before any operational use.",
         "size": 13.5, "color": GREY, "line": 1.20},
    ), fill=WASHRED)
    notes(s, """
[1:55] This slide exists because it is the single most likely question in this room.

DAYS_TO_MATURITY dominates every importance ranking, and its IV is 4.554 — more than four times
the leakage flag. So: is it just a restatement of the label, the way NHOMNO was?

Four tests, and each one could have gone the other way.

One. The dumbest possible classifier: call a loan non-performing if it is past maturity. If the
variable encoded the labelling rule that would reconstruct the label almost perfectly. It gets
67.3 percent, F1 0.508. Not a label copy.

Two. The boundaries. If it WERE the overdue criterion, essentially no Group-1 loan would be past
maturity. In fact 35 percent are — rolled-over working capital, common in Vietnamese lending. And
8 percent of Group-5 loans are NOT past maturity, so the label reflects restructuring history too.

Three. Ablation. Take it out: Macro-F1 falls 16 to 23 percent. A genuine strong predictor — if it
were the label, the model would collapse, not degrade.

Four. We chose d-zero arbitrarily. Shifting it adds a constant to every record, so tree partitions
are unchanged. Identical to four decimals across five dates.

Now the red box, and deliver it rather than skip it. None of this PROVES there is no leakage. The
file has no days-past-due column at all. We say so in the paper and I say so here. What we claim
is narrower: four tests that could have failed did not — materially stronger than a plausible
narrative on its own.
""")

    # ---------------------------------------------------------------- 8
    s = content_slide(prs, "RQ3: Gain vs. SHAP Feature-Group Importance", 8,
                      "Tables 8–9 — XGBoost and LightGBM, under both imbalance treatments")
    textbox(s, 0.60, 1.18, 5.50, 0.34, b(
        {"t": "Gain-based  —  unstable", "size": 16, "bold": True, "color": RED},
    ))
    table(s, T_GAIN, 0.60, 1.58, 5.90, [2.14, 0.94, 0.94, 0.94, 0.94],
          row_height=0.42, font_size=12, header_fill=DARKRED,
          first_col_align=PP_ALIGN.LEFT)
    textbox(s, 6.90, 1.18, 5.50, 0.34, b(
        {"t": "SHAP-based  —  stable", "size": 16, "bold": True, "color": GREEN},
    ))
    table(s, T_SHAP, 6.90, 1.58, 5.90, [2.14, 0.94, 0.94, 0.94, 0.94],
          row_height=0.42, font_size=12, header_fill=DARKGREEN,
          first_col_align=PP_ALIGN.LEFT)
    textbox(s, 0.60, 4.55, 12.20, 0.42, b(
        {"t": "Feature sets differ by design: Gain is computed on the 13-variable pool "
              "(includes SEX and LOAIKH); SHAP on the 11-feature main configuration (excludes them). "
              "The stability contrast is read within each table, across the four model × treatment "
              "combinations.",
         "size": 12, "italic": True, "color": GREY6, "line": 1.15},
    ))
    textbox(s, 0.60, 5.10, 12.20, 1.95, b(
        {"t": "•  Gain agrees on the leader — loan tenure/maturity — but on nothing below it. "
              "Product & loan purpose ranges from 5.7% to 24.7% across the four combinations: wide "
              "enough to reverse its rank.", "size": 15, "space_after": 8, "line": 1.20},
        {"t": "•  SHAP holds tenure/maturity at 40.8–42.8% and balance/utilization at 26.1–28.7%, "
              "preserving the same ordering in every combination.",
         "size": 15, "space_after": 8, "line": 1.20},
        {"t": "•  Gain is therefore not a stable property of the data — it depends on which "
              "tree-boosting library computed it and which treatment was in force.",
         "size": 15, "space_after": 8, "line": 1.20},
        {"t": "•  Practical implication: never base a data-investment decision on a single model's "
              "Gain ranking.", "size": 15, "bold": True, "color": NAVY, "line": 1.20},
    ))
    notes(s, """
[1:10] Two tables, same four model-and-treatment combinations, two importance measures. We set
LightGBM's importance_type explicitly to "gain" so both libraries report the same quantity.

Left: everyone agrees tenure/maturity leads. Below that, disagreement. Follow the "Product & loan
purpose" row: 24.7, 7.1, 21.7, 5.7. A factor of four — enough to move it from second to fifth
depending only on which library computed it.

Right: tenure/maturity 40.8, 41.2, 42.0, 42.8. Balance 26.1 to 28.7. Identical ordering in all four.

Say the footnote out loud rather than hide it: the two tables sit on different feature sets, so do
not compare a Gain cell against a SHAP cell. Read across the four columns WITHIN each table.

Implication: if you are deciding which data to invest in collecting and you base it on one model's
Gain ranking, you may be buying the wrong data.
""")

    # ---------------------------------------------------------------- 9
    s = content_slide(prs, "Statistical Robustness: A Corrected Paired Test", 9,
                      "Table 10 — 5-fold × 5-repeat CV on identical folds, Nadeau–Bengio "
                      "correction, Holm adjustment")
    table(s, T_CV, 0.60, 1.20, 12.10, [3.50, 2.45, 2.75, 1.65, 1.75],
          row_height=0.34, font_size=12.5, first_col_align=PP_ALIGN.LEFT,
          highlight_rows=(1,))
    textbox(s, 0.60, 5.72, 12.10, 1.55, b(
        {"t": "Nadeau–Bengio replaces the naive variance factor 1/25 = 0.04 with 1/25 + 1/4 = 0.29, "
              "because folds in repeated k-fold CV share data. Holm controls the family of eleven "
              "comparisons.",
         "size": 13.5, "color": GREY6, "space_after": 7, "line": 1.20},
        {"t": "Three of eleven comparisons fail to reject: the four leading unweighted tree-based "
              "configurations (Macro-F1 0.6183–0.6271) are statistically indistinguishable, so this "
              "paper names no best algorithm. What survives correction is a treatment effect — every "
              "tree-based model beats its own class-weighted counterpart, all five contrasts rejecting.",
         "size": 15, "bold": True, "color": BLUE, "line": 1.20},
    ))
    notes(s, """
[1:35] The proper test. Twelve configurations, twenty-five identical partitions, properly paired.

Why the correction: folds in repeated k-fold CV overlap — they share data. A naive paired t-test
treats them as independent, understating variance. Nadeau-Bengio replaces a variance factor of
0.04 with 0.29 — a factor of seven. Then Holm, for the family of eleven comparisons.

Highest point estimate: Decision Tree-none, 0.6271. A single depth-ten pruned tree, ahead of three
gradient-boosting libraries. But read the Holm column: 0.789, 0.789, 0.663. Nowhere near rejection.

So the correct statement is not "Decision Tree wins" — it is that these four cannot be separated
on this evidence. We name no winning algorithm. Choosing among them has to rest on other grounds —
training cost, interpretability, or the precision-recall profile from slide 5 — and stating which
ground you used is itself part of the governance record.

What does survive: the bottom eight rows, every class-weighted configuration. The treatment effect
is roughly an order of magnitude larger than any algorithm effect. This is also exactly what Brown
and Mues report: the best algorithm changes with the degree of imbalance.
""")

    # ---------------------------------------------------------------- 10
    s = content_slide(prs, "When a Ranking Flips", 10,
                      "Section 4.6 — the same discipline applied to our own model ranking")
    table(s, T_FLIP, 1.30, 1.35, 10.70, [3.60, 3.40, 3.70],
          row_height=0.46, font_size=14, first_col_align=PP_ALIGN.LEFT,
          highlight_rows=(4,))
    textbox(s, 0.60, 4.40, 12.20, 2.05, b(
        {"t": "•  On the single hold-out split, LightGBM–none leads XGBoost–none (0.6248 against "
              "0.6197) — which invites the conclusion that LightGBM is the better algorithm here.",
         "size": 15, "space_after": 8, "line": 1.20},
        {"t": "•  Under repeated cross-validation the ordering reverses (0.6183 against 0.6217), "
              "and the corrected paired test gives p = 0.3844 — not close to rejecting in either "
              "direction. The apparent ranking was an artefact of one split.",
         "size": 15, "space_after": 8, "line": 1.20},
        {"t": "•  We re-tested one other claim the same way. Comparing two non-reference rows of "
              "Table 10 is not valid, and an earlier version of this paper did exactly that. Tested "
              "directly, CatBoost–none over LightGBM–balanced gives Δ = 0.0125, t = 2.29, "
              "p = 0.0313 unadjusted but 0.0626 after Holm — direction reproduced, evidence weaker "
              "than the bare p-value suggested.", "size": 15, "space_after": 8, "line": 1.20},
        {"t": "•  A test that fails to reject is as much a governance result as one that rejects.",
         "size": 15, "bold": True, "color": NAVY, "line": 1.20},
    ))
    notes(s, """
[1:10] The same discipline applied to ourselves — the part of the paper I am most confident about.

On the hold-out split LightGBM-none was first. That is the natural headline: "LightGBM is best for
Vietnamese debt-group classification." A perfectly publishable sentence.

Under repeated CV it drops to fourth, behind XGBoost. The corrected test on that pair: p = 0.3844.
No evidence either way. The ranking was an artefact of one random split.

Third bullet is an erratum, stated deliberately. Every row of the Table 10 test is against the same
reference configuration, so you cannot read a comparison between two non-reference rows off it. An
earlier version of our paper did. Tested directly: the direction holds, but after Holm it is
p = 0.063, not significant. We corrected the claim rather than leaving it standing.

The takeaway generalises: a ranking that flips between two honest evaluation designs is a signal to
report both, not to pick the flattering one.
""")

    # ---------------------------------------------------------------- 11
    s = content_slide(prs, "Conclusion & Governance Implications", 11)
    textbox(s, 0.60, 1.24, 6.05, 0.32, b(
        {"t": "What we found", "size": 16, "bold": True, "color": NAVY},
    ))
    textbox(s, 0.60, 1.62, 6.05, 4.20, b(
        {"t": "RQ1 — The imbalance treatment is better identified than the algorithm.",
         "size": 14, "bold": True, "color": NAVY, "space_after": 2, "line": 1.15},
        {"t": "For every tree-based classifier, dropping class weighting improves repeated-CV "
              "Macro-F1 and survives Holm. The four leading configurations (0.6183–0.6271) cannot "
              "be separated. Highest point estimate: Decision Tree–none (0.6271 ± 0.0109); best "
              "hold-out: LightGBM–none (0.6248). We report both and claim neither.",
         "size": 13, "space_after": 11, "line": 1.18},
        {"t": "RQ2 — Nine of thirteen features recover 98–99% of peak.",
         "size": 14, "bold": True, "color": NAVY, "space_after": 2, "line": 1.15},
        {"t": "DAYS_TO_MATURITY is dominant (ablation cost 16–23%), and honest in-fold selection "
              "costs under 1% — supporting a more parsimonious, auditable specification.",
         "size": 13, "space_after": 11, "line": 1.18},
        {"t": "RQ3 — SHAP is stable across algorithms and treatments; Gain is not.",
         "size": 14, "bold": True, "color": NAVY, "space_after": 2, "line": 1.15},
        {"t": "Consistently defined Gain agrees on the leading group but not on the ranks below it. "
              "Triangulate importance measures before a data-investment decision.",
         "size": 13, "line": 1.18},
    ))
    rect(s, 6.88, 1.26, 0.015, 4.55, RGBColor(0xCC, 0xCC, 0xCC))
    textbox(s, 7.12, 1.24, 5.60, 0.32, b(
        {"t": "What it means for governance", "size": 16, "bold": True, "color": NAVY},
    ))
    textbox(s, 7.12, 1.62, 5.60, 4.20, b(
        {"t": "1.  Imbalance-aware, multi-metric evaluation is a safeguard.  Two configurations "
              "with comparable Macro-F1 behave in opposite ways; choosing on Macro-F1 alone is "
              "choosing blind.", "size": 13, "space_after": 10, "line": 1.18},
        {"t": "2.  Feature efficiency lowers governance cost.  Each feature dropped is one fewer "
              "feed to source, document, monitor for drift and explain to a supervisor.",
         "size": 13, "space_after": 10, "line": 1.18},
        {"t": "3.  No single measure should be trusted until checked against an independent one.  "
              "Applied to our own ranking, it overturned a claim we would otherwise have made.",
         "size": 13, "space_after": 10, "line": 1.18},
        {"t": "4.  An automated flag should trigger a documented, falsifiable check — not a "
              "plausible justification.  IV > 1.0 is a prompt for human review, not automatic "
              "exclusion.", "size": 13, "bold": True, "color": NAVY, "line": 1.18},
    ))
    callout(s, 0.60, 6.05, 12.15, 1.05, b(
        {"t": "Stated as conditional recommendations.  Every result comes from one institution and "
              "one classification snapshot, resampled randomly; none has been validated out-of-time. "
              "What transfers is the ordering of the checks, not the specific coefficients.",
         "size": 14, "bold": True, "color": NAVY, "line": 1.20},
    ))
    notes(s, """
[0:55] Three findings on the left, four governance implications on the right.

RQ1: the treatment is better identified than the algorithm. Say it plainly — we deliberately name
no winning algorithm, and that refusal is a result, not a failure to reach one. Note the highest
point estimate belongs to a single pruned decision tree.

RQ2: nine of thirteen features, 98 to 99 percent, surviving honest in-fold selection.
RQ3: SHAP stable, Gain not. Triangulate before you spend money on data.

On the right, point 4 is the one I would most want you to take away. An automated screen flagged
five of our variables. The wrong response is to auto-drop them. The wrong response is ALSO to write
a paragraph explaining why they are fine. The right response is a check that could fail.

And the box: these are conditional. One bank, one snapshot. I will not overclaim.
""")

    # ---------------------------------------------------------------- 12
    s = content_slide(prs, "Limitations & Future Work", 12)
    textbox(s, 0.60, 1.26, 12.15, 4.55, b(
        {"t": "1.   Small minority-group samples.   Only 121 Group-3 loans in the test set, so "
              "per-class estimates carry real uncertainty. Model comparisons should rely on the "
              "repeated-CV figures — and even there several configurations are indistinguishable.",
         "size": 15, "space_after": 12, "line": 1.20},
        {"t": "2.   Single institution, single snapshot.   The file records no classification date, "
              "so all resampling is random rather than temporal. An out-of-time evaluation is the "
              "necessary next step, and the Group-5 legacy stock suggests in-sample performance may "
              "be flattering relative to what such a test would show.",
         "size": 15, "space_after": 12, "line": 1.20},
        {"t": "3.   Label encoding.   Categorical variables are label-encoded for the non-CatBoost "
              "models, imposing an arbitrary ordering on nominal codes such as branch identifiers. "
              "Native categorical handling was verified for CatBoost only.",
         "size": 15, "space_after": 12, "line": 1.20},
        {"t": "4.   Hyperparameters fixed a priori, not tuned.   This keeps the comparison clean, "
              "but compares algorithm families at sensible defaults rather than at their optima. A "
              "tuning budget could reorder the four inseparable configurations; it is less likely to "
              "overturn the treatment effect, which is an order of magnitude larger.",
         "size": 15, "space_after": 12, "line": 1.20},
        {"t": "Throughout: the temporal availability of DAYS_TO_MATURITY cannot be confirmed from "
              "the dataset alone. Every governance claim is conditional on out-of-time validation "
              "this single-snapshot dataset cannot supply.",
         "size": 14, "bold": True, "color": RED, "line": 1.20},
    ))
    rect(s, 0.00, 6.15, SLIDE_W, 1.35, NAVY)
    textbox(s, 0.60, 6.28, 12.15, 0.45, [
        {"t": "Thank you  —  Questions & Discussion", "size": 21, "bold": True,
         "color": WHITE, "align": PP_ALIGN.CENTER},
    ])
    textbox(s, 0.60, 6.80, 12.15, 0.45, [
        {"t": "Đàm Công Danh  ·  danh.dc251317m@sis.hust.edu.vn  ·  +84 966 565 830",
         "size": 15, "color": PALEBLUE, "align": PP_ALIGN.CENTER},
    ])
    notes(s, """
[0:40] Four limitations, stated plainly — I would rather raise them than be asked.

One. 121 Group-3 loans in the test set; the minority-group numbers are noisy. Trust repeated-CV.

Two. One bank, one snapshot, no classification date, so all resampling is random not temporal. And
I will go further than the paper: given the Group-5 legacy stock, I expect out-of-time numbers to
be WORSE than these, not better.

Three. Label encoding imposes a fake ordering on branch codes; native handling verified for
CatBoost only.

Four. No tuning. A genuine trade-off, not an oversight — tuning would have consumed an evaluation
partition. It could reorder the four configurations the test already cannot separate; it is much
less likely to overturn the treatment effect.

Thank you. I have three backup slides — the leakage diagnostics and ablation, both confusion
matrices, and the exact hyperparameters — if any would be useful.
""")

    # ================================================================ BACKUP
    # ---- B1
    s = content_slide(prs, "B1 — Leakage Diagnostics & Ablation", "B1",
                      "Table 2 (flagged variables) and Table 6 — supporting slide 7")
    textbox(s, 0.60, 1.22, 5.90, 0.32, b(
        {"t": "Variables above the IV = 1.0 leakage flag", "size": 14.5, "bold": True, "color": NAVY},
    ))
    table(s, [T_IV[0]] + T_IV[1:6], 0.60, 1.60, 5.90, [2.30, 0.85, 2.75],
          row_height=0.46, font_size=11, first_col_align=PP_ALIGN.LEFT)
    textbox(s, 6.85, 1.22, 5.90, 0.32, b(
        {"t": "Removing DAYS_TO_MATURITY (Table 6)", "size": 14.5, "bold": True, "color": NAVY},
    ))
    table(s, [["Model", "Treat.", "11 feat.", "10 feat.", "Change"],
              ["XGBoost", "none", "0.6197", "0.4776", "−22.9%"],
              ["XGBoost", "wt.", "0.5966", "0.4995", "−16.3%"],
              ["LightGBM", "none", "0.6248", "0.4823", "−22.8%"],
              ["LightGBM", "wt.", "0.6044", "0.5052", "−16.4%"]],
          6.85, 1.60, 5.90, [1.35, 0.95, 1.15, 1.15, 1.30],
          row_height=0.52, font_size=11.5, first_col_align=PP_ALIGN.LEFT)
    textbox(s, 0.60, 4.55, 12.15, 2.40, b(
        {"t": "•  Each flagged variable was reviewed against a two-part admissibility criterion — "
              "available no later than the classification point, and not a direct or derived "
              "restatement of the target. CURRMIACCTTYPCD (31-category product sub-type) and "
              "MUCDICHVAY (79-category loan purpose) are contractual attributes fixed at "
              "origination, not outcomes of the classification process. That is what distinguishes "
              "them from NHOMNO and NHOMNO_TCBS, which were excluded.",
         "size": 14, "space_after": 9, "line": 1.20},
        {"t": "•  The ablation is run under both treatments, so it is not itself conditional on the "
              "treatment the model comparison rejects. It establishes a non-redundant contribution — "
              "it does not establish temporal admissibility or rule out proxy leakage.",
         "size": 14, "space_after": 9, "line": 1.20},
        {"t": "•  Reference-date check: only DAYS_TO_MATURITY depends on d₀, and shifting d₀ adds "
              "one constant to every record, leaving tree partitions unchanged. Across five dates "
              "from 31 Dec 2020 to 31 Dec 2023, Macro-F1 is identical to four decimal places "
              "(0.6248 unweighted; ROC-AUC 0.9246).", "size": 14, "line": 1.20},
    ))
    notes(s, "Backup for Q1 (days past due vs. days to maturity), Q2 (leakage) and Q3 (why the flagged variables were kept).")

    # ---- B2
    s = content_slide(prs, "B2 — Confusion Matrices, Both Treatments", "B2",
                      "Tables 4 and B1 — LightGBM, 11-feature configuration, hold-out set")
    textbox(s, 0.60, 1.22, 12.10, 0.30, b(
        {"t": "No weighting  —  Macro-F1 0.6248", "size": 15, "bold": True, "color": NAVY},
    ))
    table(s, T_CONF_NONE, 0.60, 1.58, 12.10, [1.60, 1.05, 1.05, 1.05, 1.05, 1.05,
                                              1.20, 1.20, 1.20, 1.65],
          row_height=0.36, font_size=12, first_col_align=PP_ALIGN.LEFT)
    textbox(s, 0.60, 3.90, 12.10, 0.30, b(
        {"t": "Class weighting  —  Macro-F1 0.6044", "size": 15, "bold": True, "color": NAVY},
    ))
    table(s, T_CONF_WT, 0.60, 4.26, 12.10, [1.60, 1.05, 1.05, 1.05, 1.05, 1.05,
                                            1.20, 1.20, 1.20, 1.65],
          row_height=0.36, font_size=12, first_col_align=PP_ALIGN.LEFT)
    textbox(s, 0.60, 6.60, 12.10, 0.70, b(
        {"t": "Rows are true debt groups, columns predicted. Unweighted: errors are concentrated, "
              "not spread across adjacent groups — 68 true Group-3 and 59 true Group-4 loans go to "
              "Group 1, which is under-classification a provisioning process cannot absorb quietly. "
              "Weighted: 323 Group-1 loans are pushed into Group 2, which is where its Group-2 "
              "precision of 0.245 comes from.",
         "size": 13, "color": NAVY, "line": 1.20},
    ))
    notes(s, "Backup for Q4 (is Macro-F1 0.62 usable?) and questions on exactly where the "
             "errors fall.")

    # ---- B3
    s = content_slide(prs, "B3 — Hyperparameters & Reproducibility", "B3",
                      "Section 3.6 and the Reproducibility Statement")
    table(s, T_HYPER, 0.60, 1.35, 12.10, [3.10, 9.00],
          row_height=0.42, font_size=13, first_col_align=PP_ALIGN.LEFT,
          align=PP_ALIGN.LEFT)
    textbox(s, 0.60, 4.30, 12.10, 2.60, b(
        {"t": "•  Fixed a priori and not tuned per configuration, so neither the hold-out set nor "
              "any CV fold is consumed by model selection. The Decision Tree is therefore a pruned "
              "tree, not an unconstrained one — which matters for reading its position in Table 10.",
         "size": 14.5, "space_after": 9, "line": 1.20},
        {"t": "•  Single software environment for every number in Tables 3–10 and Appendix B: "
              "scikit-learn 1.9.0, xgboost 3.3.0, lightgbm 4.7.0, catboost 1.2.10, shap 0.52.0. "
              "random_state = 42 for every split, resample and model fit.",
         "size": 14.5, "space_after": 9, "line": 1.20},
        {"t": "•  All twelve configurations were evaluated on the identical 25 cross-validation "
              "partitions, so every reported difference is properly paired.",
         "size": 14.5, "space_after": 9, "line": 1.20},
        {"t": "•  Five of the eleven features are categorical, including the nominal branch "
              "identifiers ORGNBR and PARENTORGNBR. For non-CatBoost models these remain "
              "label-encoded — a stated limitation. SMOTE was rejected because interpolating "
              "between encoded category values has no contractual meaning.",
         "size": 14.5, "line": 1.20},
    ))
    notes(s, "Backup for Q3 (no tuning), Q5 (SMOTE), Q6 (Decision Tree) and Q10 (label encoding).")

    prs.save(out_path)
    return prs


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "SEBL2026_presentation_Dam_Cong_Danh.pptx"
    p = build(str(out))
    msg = f"Wrote {out.name}  ({len(p.slides._sldIdLst)} slides)"
    sys.stdout.buffer.write(msg.encode("utf-8", "replace") + b"\n")
