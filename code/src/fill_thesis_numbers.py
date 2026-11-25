"""Điền các chỗ giữ chỗ @@KEY@@ trong luanvan_latex/sections/*.tex bằng số liệu
của code/results/thesis_final/ (kết quả của src/rerun_thesis_final.py).

Chạy:  cd code && python3 -m src.fill_thesis_numbers [--check]
  --check  chỉ liệt kê chỗ giữ chỗ còn thiếu số liệu, không ghi tệp.
"""
import re
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).parent.parent / "results" / "thesis_final"
TEX = Path(__file__).parent.parent.parent / "luanvan_latex" / "sections"
BASE = 265 / 98968


def vn(x, d=4):
    """Định dạng số kiểu Việt Nam: dấu phẩy thập phân, dấu chấm hàng nghìn."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pval(p):
    return "<0,001" if p < 0.001 else vn(p, 3)


def build():
    v = {}
    s = pd.read_csv(R / "t2_cv_summary.csv")
    best = s.loc[s.groupby("model")["pr_auc_mean"].idxmax()].set_index("model")
    tests = pd.read_csv(R / "t2_tests.csv")
    tests["m"] = tests["comparator"].str.replace(r" \(.*\)", "", regex=True)
    tests = tests.set_index("m")
    for key, name in [("XGB", "XGBoost"), ("LGB", "LightGBM"), ("RF", "Random Forest"), ("CB", "CatBoost"),
                      ("DT", "Decision Tree"), ("LR", "Logistic Regression")]:
        r = best.loc[name]
        v[f"T_{key}_AP"] = f"{vn(r.pr_auc_mean)}$\\pm${vn(r.pr_auc_std)}"
        v[f"T_{key}_APM"] = vn(r.pr_auc_mean)
        v[f"T_{key}_ROC"] = f"{vn(r.roc_auc_mean)}$\\pm${vn(r.roc_auc_std)}"
        v[f"T_{key}_P"] = f"{vn(r.precision_pos_mean, 3)}$\\pm${vn(r.precision_pos_std, 3)}"
        v[f"T_{key}_R"] = f"{vn(r.recall_pos_mean, 3)}$\\pm${vn(r.recall_pos_std, 3)}"
        v[f"T_{key}_STRAT"] = r.strategy
        if name in tests.index:
            v[f"T_{key}_PV"] = pval(tests.loc[name, "p_holm"])
    v["T_LIFT"] = f"{best.loc['XGBoost'].pr_auc_mean / BASE:.0f}"
    x = s[s.model == "XGBoost"].set_index("strategy")
    v.update({"T_S_NONE": vn(x.loc["none"].pr_auc_mean), "T_S_SMOTE": vn(x.loc["custom"].pr_auc_mean),
              "T_S_COMB": vn(x.loc["smote_class_weight"].pr_auc_mean), "T_S_CW": vn(x.loc["class_weight"].pr_auc_mean),
              "T_R_NONE": vn(x.loc["none"].recall_pos_mean, 3), "T_R_CW": vn(x.loc["class_weight"].recall_pos_mean, 3),
              "T_P_NONE": vn(x.loc["none"].precision_pos_mean, 3), "T_P_CW": vn(x.loc["class_weight"].precision_pos_mean, 3)})

    rk = pd.read_csv(R / "t2_recall_at_k.csv")
    rk2 = pd.read_csv(R / "t2_recall_at_k_no_interest.csv").set_index("k_pct")
    v["T_SPLIT_AP"], v["T_SPLIT_ROC"] = vn(rk.ap.iloc[0]), vn(rk.roc.iloc[0])
    rows = []
    for _, r in rk.iterrows():
        k = int(r.k_pct)
        rows.append(f"{k}\\% & {vn(r.n_reviewed, 0)} & {int(r.found)} & {vn(r.recall * 100, 1)}\\% & "
                    f"{vn(r.precision * 100, 1)}\\% & {vn(r.lift, 1)} & {int(rk2.loc[k, 'found'])} & "
                    f"{vn(rk2.loc[k, 'recall'] * 100, 1)}\\%\\\\")
    v["T_RK_ROWS"] = "\n".join(rows)
    r5 = rk.set_index("k_pct").loc[5]
    v.update({"T_RK5_FOUND": str(int(r5.found)), "T_RK5_REC": f"{vn(r5.recall * 100, 1)}\\%",
              "T_RK5_PREC": f"{vn(r5.precision * 100, 1)}\\%", "T_RK5_LIFT": f"{r5.found / 2.65:.0f}".replace(".", ","),
              "T_RK5_REC_NI": f"{vn(rk2.loc[5, 'recall'] * 100, 1)}\\%",
              "T_RK5_REC_EN": f"{r5.recall * 100:.1f}\\%",
              "T_XGB_APM_EN": f"{best.loc['XGBoost'].pr_auc_mean:.4f}"})

    ty = pd.read_csv(R / "t2_ap_by_type.csv").set_index("type")
    v.update({"T_TY_12": vn(ty.loc["1to2", "ap"]), "T_TY_12L": f"{ty.loc['1to2', 'ap'] / ty.loc['1to2', 'base']:.0f}",
              "T_TY_R": vn(ty.loc["rarer", "ap"]), "T_TY_RL": f"{ty.loc['rarer', 'ap'] / ty.loc['rarer', 'base']:.0f}"})

    ab = pd.read_csv(R / "t2_interest_ablation.csv")
    rows = []
    for name in ["XGBoost", "LightGBM", "Random Forest"]:
        a = ab[ab.model == name].set_index("config")
        cells = [f"{vn(a.iloc[0].ap_full)}"]
        for c in ["no_offbalance", "no_both_interest"]:
            pv = pval(a.loc[c, "p_corrected"])
            pv = pv if pv.startswith("<") else "=" + pv
            cells.append(f"{vn(a.loc[c, 'pr_auc_mean'])}$\\pm${vn(a.loc[c, 'pr_auc_std'])} ($p{pv}$)")
        rows.append(f"{name} & " + " & ".join(cells) + "\\\\")
    v["T_AB_ROWS"] = "\n".join(rows)

    sh = pd.read_csv(R / "t2_shap.csv")
    piv = sh.pivot_table(index="feature", columns="sample", values="mean_abs_shap")
    piv = piv.sort_values("stratified_500", ascending=False).head(8)
    rows = []
    for f, r in piv.iterrows():
        nm = f"\\texttt{{\\seqsplit{{{f.replace('_', chr(92) + '_')}}}}}" if f.startswith("ENG_") else f"\\texttt{{{f}}}"
        rows.append(f"{nm} & {vn(r.stratified_500, 3)} & {vn(r.events, 3)} & {vn(r.non_events, 3)}\\\\")
    v["T_SHAP_ROWS"] = "\n".join(rows)

    if (R / "x_exits_sensitivity.csv").exists():
        e = pd.read_csv(R / "x_exits_sensitivity.csv").iloc[0]
        v["T_EXIT_AP"] = f"{vn(e.ap_mean)}$\\pm${vn(e.ap_std)}"
    if (R / "x_loan_age.csv").exists():
        la = pd.read_csv(R / "x_loan_age.csv")
        v["AGE_APR"], v["AGE_MAY"] = (vn(x, 0) for x in la.tuoi_khoan_vay_trung_vi)
    return v


def main():
    check = "--check" in sys.argv
    try:
        v = build()
    except FileNotFoundError as e:
        v = {}
        print(f"Chưa đủ kết quả: {e}")
    missing = set()
    for f in sorted(TEX.glob("*.tex")):
        s = f.read_text()
        keys = set(re.findall(r"@@([A-Z0-9_]+)@@", s))
        missing |= keys - set(v)
        if not check and keys & set(v):
            for k in keys & set(v):
                s = s.replace(f"@@{k}@@", v[k])
            f.write_text(s)
            print(f"{f.name}: điền {len(keys & set(v))} chỗ")
    print("Còn thiếu:", sorted(missing) or "không")


if __name__ == "__main__":
    main()
