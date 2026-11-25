"""Phân tích đa kỳ cho bài ICEIR 2026. Tệp 20260630 là kỳ MÔ PHỎNG (không phải dữ liệu ngân hàng);
mọi kết quả cặp 5->6 (p2_*, tr_transfer) phải được ghi là dữ liệu giả định trong bài.

  P2   Bài toán 2 trên cặp 5->6 với cùng các phương pháp và baseline như cặp 4->5
       (5x10 CV), gồm AP trên ca không cơ học; độ nhạy hợp đồng rời danh mục.
  TR   Chuyển giao hai chiều giữa hai cặp (huấn luyện trên toàn bộ một cặp,
       chấm điểm toàn bộ cặp kia) cho XGBoost 20 biến và XGBoost + trường quá hạn.
  T1J  Bài toán 1: mô hình huấn luyện trên tháng 4, đánh giá trên tháng 6 theo tập con.
  OFF  Tỷ lệ lãi ngoại bảng > 0 theo nhóm nợ cho cả ba tháng.
  SH   Biểu đồ SHAP beeswarm cho mô hình chính (cặp 4->5).
  CB   Chi phí - lợi ích: phần tăng dự phòng (theo tỷ lệ 0/5/20/50/100%) của các
       ca bắt được theo ngân sách rà soát, từ điểm out-of-fold.

Chạy:  cd code && python3 -m src.review_iceir2 [P2] [TR] [T1J] [OFF] [SH] [CB]
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from xgboost import XGBRanker

from config.config import (RANDOM_STATE, TEST_SIZE, DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST,
                           DATA_CREDIT_INFO_TEST2, CREDIT_INFO_SMOTE_STRATEGY)
from src.credit_info_preprocessing import (load_credit_info, build_raw_feature_pool, engineer_business_features,
                                           prepare_credit_info, build_transition_dataset, TRANSITION_LABEL_COL as LAB)
from src.models import build_pipeline
from src.review_iceir import overdue_features, rank_metrics, woe_transform, KEY, TGT, OFF, ON

OUT = Path(__file__).parent.parent / "results" / "iceir"
FIG = Path(__file__).parent.parent.parent / "iceir_paper" / "figures"
OVC = ["DPD_PRINCIPAL", "DPD_INTEREST", "OVERDUE_PRINCIPAL", "OVERDUE_INTEREST"]
DELTA_PROV = {1: 0.05, 2: 0.15, 3: 0.30, 4: 0.50}  # tăng tỷ lệ trích lập khi chuyển lên nhóm kế tiếp (NĐ 86/2024)
BAL = "Số dư nợ theo nguyên tệ"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def build_pair(t, t1):
    dm = build_transition_dataset(t, t1, id_col=KEY, target_col=TGT)
    nr, cr = build_raw_feature_pool(dm); de, eng = engineer_business_features(dm); num, cat = nr + eng, cr
    X, y = prepare_credit_info(de, LAB, num, cat, label_offset=0)
    d = de.loc[X.index]; ov = overdue_features(d, d["ETL_DATE"])
    Xd = pd.concat([X, ov[OVC]], axis=1)
    return dict(X=X, Xd=Xd, y=y, num=num, cat=cat, ov=ov, g0=d[TGT].astype(int).values, bal=d[BAL].fillna(0).values,
                ids=d[KEY].values)


def xgb(cat, num):
    return build_pipeline("xgboost", cat, num, random_state=RANDOM_STATE, imbalance_strategy="none", eval_metric="logloss")


def cv_methods(P, tag):
    X, Xd, y, num, cat, ov, g0 = P["X"], P["Xd"], P["y"], P["num"], P["cat"], P["ov"], P["g0"]
    mech = ov["MECHANICAL"].values
    log(f"{tag}: {len(y)} khoản vay, {int(y.sum())} sự kiện, cơ học {int(mech[y == 1].sum())}")
    rows = []
    for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(X, y)):
        ytr, yte = y[tr], y[te]; sc = {}
        p = xgb(cat, num).fit(X.iloc[tr], ytr); sc["XGBoost (20 features)"] = p.predict_proba(X.iloc[te])[:, 1]
        prep = p.named_steps["preprocessor"]
        rk = XGBRanker(objective="rank:pairwise", n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)
        rk.fit(prep.transform(X.iloc[tr]), ytr, qid=np.zeros(len(ytr), int)); sc["XGBoost rank:pairwise"] = rk.predict(prep.transform(X.iloc[te]))
        sc["XGBoost + overdue fields"] = xgb(cat, num + OVC).fit(Xd.iloc[tr], ytr).predict_proba(Xd.iloc[te])[:, 1]
        Wtr, Wte = woe_transform(X.iloc[tr], ytr, X.iloc[te], cat)
        sc["WOE-logistic scorecard"] = LogisticRegression(C=1.0, max_iter=2000).fit(Wtr, ytr).predict_proba(Wte)[:, 1]
        sc["DPD at T"] = ov["DPD_MAX"].values[te]
        sc["Distance to next threshold"] = -ov["DIST_NEXT"].replace(np.inf, 1e6).values[te]
        for name, col in [("On-balance interest receivable", ON), ("Off-balance unpaid interest", OFF)]:
            v = X[col].fillna(0).values; sign = 1 if roc_auc_score(ytr, v[tr]) >= 0.5 else -1; sc[name] = sign * v[te]
        nm = (mech[te] == 0) | (yte == 0)
        for name, s in sc.items():
            m = rank_metrics(yte, s, g0[te])
            rows.append({"fold": f, "method": name, **m, "events_nonmech": int(yte[nm].sum()),
                         "ap_nonmechanical": average_precision_score(yte[nm], s[nm]) if yte[nm].sum() else np.nan})
        if f % 10 == 9:
            log(f"{tag} fold {f + 1}/50"); save(pd.DataFrame(rows), f"{tag}_folds.csv")
    df = pd.DataFrame(rows); save(df, f"{tag}_folds.csv")
    agg = df.drop(columns="fold").groupby("method").agg(["mean", "std"]); agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index().sort_values("ap_mean", ascending=False); save(agg, f"{tag}_baselines.csv")
    g = lambda mth: df[df.method == mth].sort_values("fold")["ap"].values
    tests = []
    for a_, b_ in [("XGBoost + overdue fields", "Distance to next threshold"), ("XGBoost (20 features)", "Distance to next threshold"),
                   ("XGBoost rank:pairwise", "XGBoost (20 features)")]:
        d = g(a_) - g(b_); k = len(d); t = d.mean() / np.sqrt((1 / k + 0.25) * d.var(ddof=1))
        tests.append({"a": a_, "b": b_, "mean_diff": d.mean(), "p_corrected": 2 * stats.t.sf(abs(t), k - 1)})
    save(pd.DataFrame(tests), f"{tag}_tests.csv")
    log(f"{tag}:\n{agg[['method','ap_mean','ap_std','roc_mean','recall@1_mean','recall@5_mean','precision@5_mean','ndcg@5_mean','ap_nonmechanical_mean']].round(4)}\n{pd.DataFrame(tests).round(4)}")


def transfer(P1, P2):
    rows = []
    for src, dst, lab in [(P1, P2, "Apr->May to May->Jun"), (P2, P1, "May->Jun to Apr->May")]:
        for name, feats in [("XGBoost (20 features)", "X"), ("XGBoost + overdue fields", "Xd")]:
            num = src["num"] + (OVC if feats == "Xd" else [])
            s = xgb(src["cat"], num).fit(src[feats], src["y"]).predict_proba(dst[feats])[:, 1]
            rows.append({"direction": lab, "method": name, **rank_metrics(dst["y"], s, dst["g0"])})
        s = -dst["ov"]["DIST_NEXT"].replace(np.inf, 1e6).values
        rows.append({"direction": lab, "method": "Distance to next threshold (no training)", **rank_metrics(dst["y"], s, dst["g0"])})
    df = pd.DataFrame(rows); save(df, "tr_transfer.csv"); log(f"TR:\n{df[['direction','method','ap','roc','recall@1','recall@5']].round(4)}")


def exits_sensitivity(t, t1, P, tag):
    exits = t[~t[KEY].isin(t1[KEY])].copy(); de_e, _ = engineer_business_features(exits)
    Xa = pd.concat([P["X"], de_e[P["num"] + P["cat"]]]); ya = np.r_[P["y"], (de_e[TGT].astype(int) >= 2).astype(int).values]
    aps = [average_precision_score(ya[te], xgb(P["cat"], P["num"]).fit(Xa.iloc[tr], ya[tr]).predict_proba(Xa.iloc[te])[:, 1])
           for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(Xa, ya)]
    r = {"pair": tag, "exits": len(exits), "added_events": int(ya.sum() - P["y"].sum()), "events": int(ya.sum()),
         "ap_mean": float(np.mean(aps)), "ap_std": float(np.std(aps, ddof=1))}
    save(pd.DataFrame([r]), f"{tag}_exits.csv"); log(f"M6 {tag}: {r}")


def task1_june(a, c):
    nr, cr = build_raw_feature_pool(a); de4, eng = engineer_business_features(a); num, cat = nr + eng, cr
    X, y = prepare_credit_info(de4, TGT, num, cat, label_offset=1); ids = de4.loc[X.index, KEY].values
    Xtr, Xva, ytr, yva, idtr, idva = train_test_split(X, y, ids, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    de6, _ = engineer_business_features(c); Xj, yj = prepare_credit_info(de6, TGT, num, cat, label_offset=1); idj = de6.loc[Xj.index, KEY].values
    sets = {"June: seen (Apr train)": np.isin(idj, idtr), "June: unseen (Apr test)": np.isin(idj, idva),
            "June: not in April": ~np.isin(idj, ids), "June: all": np.ones(len(yj), bool)}
    rows = []
    for model in ["lightgbm", "xgboost"]:
        for cfg, drop in [("full", []), ("conservative", [OFF, ON])]:
            nn = [x for x in num if x not in drop]
            p = build_pipeline(model, cat, nn, random_state=RANDOM_STATE, imbalance_strategy="custom",
                               custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY).fit(Xtr[nn + cat], ytr)
            pred = np.asarray(p.predict(Xj[nn + cat])).ravel().astype(int)
            for sn, msk in sets.items():
                yy, pp = yj[msk], pred[msk]; present = sorted(set(yy))
                rows.append({"model": model, "config": cfg, "subset": sn, "n": int(msk.sum()), "groups": "".join(str(g + 1) for g in present),
                             "macro_f1": f1_score(yy, pp, labels=present, average="macro"), "accuracy": float((yy == pp).mean())})
    df = pd.DataFrame(rows); save(df, "t1_june_subsets.csv"); log(f"T1J:\n{df.round(4)}")


def offbalance(a, b, c):
    rows = [{"snapshot": n, "group": g, "pct": float((d.loc[d[TGT].astype(int) == g, OFF] > 0).mean() * 100)}
            for n, d in [("2026-04-30", a), ("2026-05-31", b), ("2026-06-30", c)] for g in range(1, 6)]
    df = pd.DataFrame(rows).pivot(index="group", columns="snapshot", values="pct").reset_index()
    save(df, "off_by_group_3months.csv"); log(f"OFF:\n{df.round(2)}")


def shap_beeswarm(P):
    import shap
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names_en = {"DPD_INTEREST": "Interest DPD", "DPD_PRINCIPAL": "Principal DPD", "OVERDUE_INTEREST": "Overdue interest amount",
                "OVERDUE_PRINCIPAL": "Overdue principal amount", "Số dư nợ theo nguyên tệ": "Outstanding balance",
                "ENG_TENOR_ACTUAL_DAYS": "Actual tenor", "Lãi suất": "Interest rate", "ENG_DAYS_TO_MATURITY": "Days to maturity",
                "ENG_LOAN_AGE_DAYS": "Loan age", "Mã chi nhánh TCTD": "Branch code", ON: "On-balance interest receivable",
                OFF: "Off-balance unpaid interest", "Mô tả mục đích sử dụng tiền vay": "Loan purpose description"}
    Xtr, Xva, ytr, yva = train_test_split(P["Xd"], P["y"], test_size=TEST_SIZE, stratify=P["y"], random_state=RANDOM_STATE)
    p = xgb(P["cat"], P["num"] + OVC).fit(Xtr, ytr); prep, clf = p.named_steps["preprocessor"], p.named_steps["classifier"]
    names = prep.get_feature_names(); Xt = pd.DataFrame(prep.transform(Xva), columns=names)
    rng = np.random.RandomState(RANDOM_STATE); pos = np.where(yva == 1)[0]
    idx = np.r_[pos, rng.choice(np.where(yva == 0)[0], 500 - len(pos), replace=False)]
    ex = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    sv = ex.shap_values(Xt.iloc[idx])
    Xs = Xt.iloc[idx].rename(columns=lambda c: names_en.get(c, c))
    plt.figure()
    shap.summary_plot(sv, Xs, max_display=10, show=False, plot_size=(6.0, 3.6), color_bar_label="Feature value (low to high)")
    plt.xlabel("SHAP value (impact on log-odds of migration)", fontsize=8)
    plt.gcf().axes[0].tick_params(labelsize=7.5)
    plt.tight_layout(); plt.savefig(FIG / "fig3_shap_beeswarm.pdf", bbox_inches="tight"); plt.savefig(FIG / "fig3_shap_beeswarm.png", dpi=200, bbox_inches="tight")
    log("SH: đã vẽ beeswarm")


def cost_benefit(P, tag):
    """Điểm out-of-fold (5 fold, 1 lần) -> phần tăng dự phòng của các ca bắt được theo ngân sách k."""
    y, g0, bal = P["y"], P["g0"], P["bal"]
    at_stake = np.array([DELTA_PROV.get(g, 0) for g in g0]) * bal * y
    scores = {"XGBoost + overdue fields": np.zeros(len(y)),
              "Distance to next threshold": -P["ov"]["DIST_NEXT"].replace(np.inf, 1e6).values}
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(P["X"], y):
        scores["XGBoost + overdue fields"][te] = xgb(P["cat"], P["num"] + OVC).fit(P["Xd"].iloc[tr], y[tr]).predict_proba(P["Xd"].iloc[te])[:, 1]
    rows = []
    for name, s in scores.items():
        o = np.argsort(-s, kind="mergesort")
        for k in [1, 2, 5, 10]:
            n = int(round(len(y) * k / 100)); top = o[:n]
            cap = at_stake[top].sum()
            rows.append({"pair": tag, "method": name, "k_pct": k, "n_reviewed": n, "events_found": int(y[top].sum()),
                         "provision_increase_captured": cap, "share_of_total": cap / at_stake.sum(),
                         "breakeven_cost_per_review": cap / n})
    df = pd.DataFrame(rows); save(df, f"{tag}_cost_benefit.csv")
    log(f"CB {tag} (tổng phần tăng dự phòng {at_stake.sum():,.0f}):\n{df[['method','k_pct','events_found','share_of_total','breakeven_cost_per_review']].round(3)}")


if __name__ == "__main__":
    want = {x.upper() for x in sys.argv[1:]} or {"OFF", "T1J", "P2", "TR", "SH", "CB"}
    a, b, c = (load_credit_info(p) for p in (DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST, DATA_CREDIT_INFO_TEST2))
    P1 = build_pair(a, b) if want & {"TR", "SH", "CB"} else None
    P2 = build_pair(b, c) if want & {"P2", "TR", "CB"} else None
    if "OFF" in want: offbalance(a, b, c)
    if "T1J" in want: task1_june(a, c)
    if "SH" in want: shap_beeswarm(P1)
    if "CB" in want: cost_benefit(P1, "p1"); cost_benefit(P2, "p2")
    if "TR" in want: transfer(P1, P2)
    if "P2" in want: cv_methods(P2, "p2"); exits_sensitivity(b, c, P2, "p2")
    log("HOÀN TẤT")
