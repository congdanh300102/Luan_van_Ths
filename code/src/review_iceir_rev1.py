"""Phân tích bổ sung theo phản biện ICEIR 2026 (Paper 282, 22/11/2026).

Chỉ dùng HAI kỳ dữ liệu thật (20260430, 20260531).

  IV   IV theo hai cách chia khoảng: thập phân vị (cách cũ) và có khoảng riêng
       cho giá trị khối (0); "bad" = nhóm 2-5, "good" = nhóm 1 (phản biện 2).
  T2G  Bài toán 2 với cùng thông tin: thêm nhóm nợ hiện tại G_T vào XGBoost,
       và thêm điểm của luật (khoảng cách tới ngưỡng) làm biến (phản biện 1).
  GRP  Rò rỉ theo khách hàng: số hợp đồng/khách hàng, độ nhất quán nhóm nợ trong
       khách hàng; chia theo khách hàng cho Bài toán 1 và 2 (phản biện 4).
  ROB  Các kiểm tra độ bền của Bài toán 1 trên cấu hình bảo thủ (phản biện 3).
  BOOT Khoảng tin cậy bootstrap theo cụm khách hàng cho Bảng Macro-F1 (phản biện 3).

Chạy:  cd code && python3 -m src.review_iceir_rev1 [IV] [T2G] [GRP] [ROB] [BOOT]
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, train_test_split

from config.config import (
    RANDOM_STATE, TEST_SIZE, DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST,
    CREDIT_INFO_TARGET_COL as TGT, CREDIT_INFO_SMOTE_STRATEGY, TRANSITION_ID_COL as KEY,
)
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info, build_transition_dataset, feature_to_group, TRANSITION_LABEL_COL as LAB,
)
from src.iv_analysis import compute_iv_table
from src.models import build_pipeline
from src.rerun_review_fixes import evaluate, load_b
from src.review_iceir import overdue_features, rank_metrics, OFF, ON

OUT = Path(__file__).parent.parent / "results" / "iceir" / "rev1"
OUT.mkdir(parents=True, exist_ok=True)
CUST = "Mã khách hàng do TCTD cấp"
OVC = ["DPD_PRINCIPAL", "DPD_INTEREST", "OVERDUE_PRINCIPAL", "OVERDUE_INTEREST"]
SMOTE_KW = {"imbalance_strategy": "custom", "custom_smote_strategy": CREDIT_INFO_SMOTE_STRATEGY}
XGB_KW = {"random_state": RANDOM_STATE, "imbalance_strategy": "none", "eval_metric": "logloss"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def corrected_ttest(a, b, ratio=0.25):
    d = np.asarray(a) - np.asarray(b); k = len(d)
    t = d.mean() / np.sqrt((1 / k + ratio) * d.var(ddof=1))
    return d.mean(), 2 * stats.t.sf(abs(t), df=k - 1)


def task2_data():
    a, b = load_credit_info(DATA_CREDIT_INFO_TRAIN), load_credit_info(DATA_CREDIT_INFO_TEST)
    dm = build_transition_dataset(a, b, id_col=KEY, target_col=TGT)
    nr, cr = build_raw_feature_pool(dm); de, eng = engineer_business_features(dm); num, cat = nr + eng, cr
    X, y = prepare_credit_info(de, LAB, num, cat, label_offset=0)
    d = de.loc[X.index]; ov = overdue_features(d, d["ETL_DATE"])
    X = X.copy()
    X["G_T"] = d[TGT].astype(int).values
    X["RULE_SCORE"] = -ov["DIST_NEXT"].replace(np.inf, 1e6).values
    for c in OVC:
        X[c] = ov[c].values
    return X, y, num, cat, d, ov


def task2_scores(X, tr, te, y, num, cat, configs):
    out = {}
    for name, extra in configs.items():
        p = build_pipeline("xgboost", cat, num + extra, **XGB_KW)
        p.fit(X.iloc[tr][num + extra + cat], y[tr])
        out[name] = p.predict_proba(X.iloc[te][num + extra + cat])[:, 1]
    out["Rule: distance to next threshold"] = X["RULE_SCORE"].values[te]
    return out


T2_CONFIGS = {
    "XGB, 20 features": [],
    "XGB, 20 features + G_T": ["G_T"],
    "XGB + overdue fields": OVC,
    "XGB + overdue fields + G_T": OVC + ["G_T"],
    "XGB + overdue fields + G_T + rule score": OVC + ["G_T", "RULE_SCORE"],
}


def summarise(rows, ref_pairs, prefix):
    df = pd.DataFrame(rows); save(df, f"{prefix}_folds.csv")
    agg = df.drop(columns="fold").groupby("method").agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index().sort_values("ap_mean", ascending=False); save(agg, f"{prefix}_summary.csv")
    tests = []
    for a_, b_ in ref_pairs:
        va = df[df.method == a_].sort_values("fold")["ap"].values
        vb = df[df.method == b_].sort_values("fold")["ap"].values
        md, p = corrected_ttest(va, vb)
        tests.append({"a": a_, "b": b_, "mean_diff": md, "p_corrected": p})
    save(pd.DataFrame(tests), f"{prefix}_tests.csv")
    log(f"{prefix}:\n{agg[['method', 'ap_mean', 'ap_std', 'recall@1_mean', 'recall@5_mean', 'ndcg@5_mean']].round(4)}"
        f"\n{pd.DataFrame(tests).round(4)}")


TEST_PAIRS = [
    ("XGB, 20 features + G_T", "Rule: distance to next threshold"),
    ("XGB, 20 features + G_T", "XGB, 20 features"),
    ("XGB + overdue fields + G_T", "XGB + overdue fields"),
    ("XGB + overdue fields + G_T", "Rule: distance to next threshold"),
    ("XGB + overdue fields + G_T + rule score", "XGB + overdue fields + G_T"),
    ("XGB, 20 features", "Rule: distance to next threshold"),
    ("XGB + overdue fields", "Rule: distance to next threshold"),
]


# ── IV ───────────────────────────────────────────────────────────────────────
def part_iv():
    a = load_credit_info(DATA_CREDIT_INFO_TRAIN); de, eng = engineer_business_features(a)
    nr, cr = build_raw_feature_pool(a); feats = nr + eng + cr
    new = compute_iv_table(de[feats + [TGT]], TGT).set_index("feature")["iv"]
    bad = (de[TGT] != 1).astype(int)

    def iv_decile(x):
        if x.dtype.kind not in "biuf":
            k = x.fillna("MISSING").astype(str)
        else:
            try:
                k = pd.qcut(x, 10, duplicates="drop").astype(str)
            except Exception:
                k = x.astype(str)
        g = pd.DataFrame({"k": k, "b": bad}).groupby("k")["b"].agg(["sum", "count"])
        pb, pg = g["sum"] / bad.sum(), (g["count"] - g["sum"]) / (bad == 0).sum()
        g = (pb > 0) & (pg > 0)
        return float(((pb - pg) * np.log(pb / pg))[g].sum())
    rows = [{"feature": f, "iv_decile": iv_decile(de[f]), "iv_mass_point_bins": float(new.get(f, np.nan))} for f in feats]
    df = pd.DataFrame(rows).sort_values("iv_mass_point_bins", ascending=False)
    save(df, "iv_decile_vs_masspoint.csv")
    log(f"IV (bad = nhóm 2-5; {int(bad.sum())} bad / {int((bad == 0).sum())} good):\n{df.head(8).round(3)}")


# ── T2G ──────────────────────────────────────────────────────────────────────
def part_t2g():
    X, y, num, cat, d, ov = task2_data()
    log(f"T2G: {len(y)} khoản vay, {y.sum()} sự kiện")
    rows = []
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)  # cùng fold với m2_folds.csv
    for f, (tr, te) in enumerate(rcv.split(X, y)):
        sev = X["G_T"].values[te]; mech = ov["MECHANICAL"].values[te]; yte = y[te]
        nm = (mech == 0) | (yte == 0)
        for name, s in task2_scores(X, tr, te, y, num, cat, T2_CONFIGS).items():
            m = rank_metrics(yte, s, sev)
            rows.append({"fold": f, "method": name, **m, "ap_nonmechanical": average_precision_score(yte[nm], s[nm])})
        if f % 10 == 9:
            log(f"T2G fold {f + 1}/50")
    summarise(rows, TEST_PAIRS, "t2g")


# ── GRP ──────────────────────────────────────────────────────────────────────
def part_grp():
    a = load_credit_info(DATA_CREDIT_INFO_TRAIN); b = load_credit_info(DATA_CREDIT_INFO_TEST)
    n = a.groupby(CUST).size(); g = a.groupby(CUST)[TGT].nunique()
    st = {"contracts": len(a), "customers": len(n), "contracts_per_customer_mean": n.mean(), "contracts_per_customer_max": int(n.max()),
          "share_customers_multi": float((n > 1).mean()), "share_contracts_in_multi_customers": float((a[CUST].map(n) > 1).mean()),
          "multi_customers_with_mixed_groups": int((g[n > 1] > 1).sum())}

    # Bài toán 1: chia 80/20 theo khách hàng
    nr, cr = build_raw_feature_pool(a); de4, eng = engineer_business_features(a); num, cat = nr + eng, cr
    X, y = prepare_credit_info(de4, TGT, num, cat, label_offset=1); cust4 = de4.loc[X.index, CUST].values
    de5, _ = engineer_business_features(b); Xh, yh = prepare_credit_info(de5, TGT, num, cat, label_offset=1)
    cust5 = de5.loc[Xh.index, CUST].values; id4 = de4.loc[X.index, KEY].values; id5 = de5.loc[Xh.index, KEY].values
    # theo hợp đồng (như bài): bao nhiêu hợp đồng test có khách hàng nằm trong tập train?
    _, _, _, _, ctr, cva = train_test_split(X, y, cust4, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    st["contract_split_test_share_customer_in_train"] = float(np.isin(cva, ctr).mean())
    _, _, _, _, idtr_c, idva_c = train_test_split(X, y, id4, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    unseen_c = np.isin(id5, idva_c)
    st["may_unseen_contracts_customer_in_train"] = float(np.isin(cust5[unseen_c], ctr).mean())
    st["may_contracts_new_customers"] = int((~np.isin(cust5, cust4)).sum())

    tr, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X, y, cust4))
    Xtr, ytr, Xva, yva = X.iloc[tr], y[tr], X.iloc[te], y[te]
    ctr_g = set(cust4[tr])
    seen = np.isin(cust5, list(ctr_g))
    sets = {"internal test (customer split)": (Xva, yva), "May: customers in training": (Xh[seen], yh[seen]),
            "May: customers not in training": (Xh[~seen], yh[~seen]), "May: all": (Xh, yh)}
    st["grouped_split_train_n"], st["grouped_split_test_n"] = len(tr), len(te)
    rows = []
    for model in ["lightgbm", "xgboost"]:
        for cfg, drop in [("full", []), ("conservative", [OFF, ON])]:
            nn = [c for c in num if c not in drop]
            p = build_pipeline(model, cat, nn, random_state=RANDOM_STATE, **SMOTE_KW).fit(Xtr[nn + cat], ytr)
            for sname, (XX, yy) in sets.items():
                pred = np.asarray(p.predict(XX[nn + cat])).ravel().astype(int)
                rows.append({"model": model, "config": cfg, "subset": sname, "n": len(yy),
                             "macro_f1": f1_score(yy, pred, labels=sorted(set(yy)), average="macro"),
                             "accuracy": float((pred == yy).mean())})
            log(f"GRP T1 {model} {cfg}: {[round(r['macro_f1'], 4) for r in rows[-4:]]}")
    save(pd.DataFrame(rows), "grp_task1.csv")

    # Bài toán 2: CV 5x10 theo khách hàng
    X2, y2, num2, cat2, d2, ov2 = task2_data(); c2 = d2[CUST].values
    ev = pd.Series(y2, index=c2)
    st["task2_event_customers"] = int(ev[ev == 1].index.nunique())
    st["task2_events_in_customers_with_2plus_events"] = int(ev[ev == 1].groupby(level=0).size().pipe(lambda s: s[s > 1].sum()))
    save(pd.DataFrame([st]).T.reset_index().rename(columns={"index": "stat", 0: "value"}), "grp_stats.csv")
    log(f"GRP thống kê: {st}")
    cfg2 = {k: T2_CONFIGS[k] for k in ["XGB, 20 features", "XGB + overdue fields", "XGB + overdue fields + G_T"]}
    rows, f = [], 0
    for r in range(10):
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + r).split(X2, y2, c2):
            yte = y2[te]
            for name, s in task2_scores(X2, tr, te, y2, num2, cat2, cfg2).items():
                rows.append({"fold": f, "method": name, **rank_metrics(yte, s, X2["G_T"].values[te])})
            f += 1
        log(f"GRP T2 lặp {r + 1}/10")
    summarise(rows, [("XGB, 20 features", "Rule: distance to next threshold"),
                     ("XGB + overdue fields", "Rule: distance to next threshold"),
                     ("XGB + overdue fields + G_T", "Rule: distance to next threshold"),
                     ("XGB + overdue fields + G_T", "XGB + overdue fields")], "grp_task2")


# ── ROB ──────────────────────────────────────────────────────────────────────
def part_rob():
    X, y, Xh, yh, num_p, cat_p = load_b()
    num_c = [c for c in num_p if c not in (OFF, ON)]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    rows = []

    def add(check, variant, model, p, cols):
        for split, XX, yy in [("validation", Xva, yva), ("holdout", Xh, yh)]:
            m = evaluate(p, XX[cols], yy)[0]
            rows.append({"check": check, "variant": variant, "model": model, "split": split, "f1_macro": m["f1_macro"]})
        log(f"ROB {check} {variant} {model}: {rows[-2]['f1_macro']:.4f} / {rows[-1]['f1_macro']:.4f}")

    cols = num_c + cat_p
    base = build_pipeline("lightgbm", cat_p, num_c, random_state=RANDOM_STATE, **SMOTE_KW).fit(Xtr[cols], ytr)
    add("baseline", "moderate SMOTE", "LightGBM", base, cols)
    for label, strat in [("none", "none"), ("class_weight", "class_weight"), ("SMOTE+class_weight", "smote_class_weight")]:
        p = build_pipeline("lightgbm", cat_p, num_c, random_state=RANDOM_STATE, imbalance_strategy=strat,
                           custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY).fit(Xtr[cols], ytr)
        add("imbalance", label, "LightGBM", p, cols)
    for model, key in [("LightGBM", "lightgbm"), ("XGBoost", "xgboost")]:
        p = build_pipeline(key, cat_p, num_c, random_state=RANDOM_STATE, imbalance_strategy="smote_nc",
                           custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY).fit(Xtr[cols], ytr)
        add("smote_nc", "SMOTE-NC", model, p, cols)
    Xb, yb, Xhb, yhb, num_b, cat_b = load_b(branch_as_categorical=True)
    num_bc = [c for c in num_b if c not in (OFF, ON)]
    Xbtr, Xbva, ybtr, ybva = train_test_split(Xb, yb, test_size=TEST_SIZE, stratify=yb, random_state=RANDOM_STATE)
    for model, key in [("LightGBM", "lightgbm"), ("XGBoost", "xgboost")]:
        p = build_pipeline(key, cat_b, num_bc, random_state=RANDOM_STATE, **SMOTE_KW).fit(Xbtr[num_bc + cat_b], ybtr)
        for split, XX, yy in [("validation", Xbva, ybva), ("holdout", Xhb, yhb)]:
            rows.append({"check": "branch_categorical", "variant": "branch as category", "model": model, "split": split,
                         "f1_macro": evaluate(p, XX[num_bc + cat_b], yy)[0]["f1_macro"]})
        log(f"ROB branch {model}: {rows[-2]['f1_macro']:.4f}")
    # Gain / SHAP theo nhóm biến, Top-8
    prep, clf = base.named_steps["preprocessor"], base.named_steps["classifier"]
    names = prep.get_feature_names()
    gain = pd.Series(clf.booster_.feature_importance(importance_type="gain"), index=names); gain = gain / gain.sum() * 100
    Xs = pd.DataFrame(prep.transform(Xva[cols]), columns=names).sample(5000, random_state=RANDOM_STATE)
    sv = np.asarray(shap.TreeExplainer(clf).shap_values(Xs))
    imp = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
    dd = pd.DataFrame({"feature": names, "gain_pct": gain.values, "shap_pct": imp / imp.sum() * 100})
    dd["group"] = [feature_to_group(f) for f in names]
    save(dd.sort_values("gain_pct", ascending=False), "rob_gain_shap_feature.csv")
    gg = dd.groupby("group")[["gain_pct", "shap_pct"]].sum()
    sp = stats.spearmanr(gg.gain_pct, gg.shap_pct)
    rows.append({"check": "gain_vs_shap_groups", "variant": f"spearman rho (n_groups={len(gg)})", "model": "LightGBM",
                 "split": f"p={sp.pvalue:.4f}", "f1_macro": sp.statistic})
    order = dd.sort_values("gain_pct", ascending=False)["feature"].tolist()[:8]
    ck, nk = [c for c in cat_p if c in order], [c for c in num_c if c in order]
    p8 = build_pipeline("lightgbm", ck, nk, random_state=RANDOM_STATE, **SMOTE_KW).fit(Xtr[nk + ck], ytr)
    add("top8", "8 features by Gain: " + "; ".join(order), "LightGBM", p8, nk + ck)
    # theo nhóm (recall nhóm 2) và sai lệch >= 2 nhóm
    m, cm, (pr, rc, f1, sup) = evaluate(base, Xva[cols], yva)
    off2 = (cm * (np.abs(np.arange(5)[:, None] - np.arange(5)) >= 2)).sum() / cm.sum()
    rows.append({"check": "per_group", "variant": "recall G1..G5 | share off by >=2", "model": "LightGBM", "split": "validation",
                 "f1_macro": np.nan, "detail": " ".join(f"{v:.3f}" for v in rc) + f" | {off2:.4f}"})
    save(pd.DataFrame(rows), "rob_conservative.csv")


# ── BOOT ─────────────────────────────────────────────────────────────────────
def part_boot(n_boot=1000):
    a = load_credit_info(DATA_CREDIT_INFO_TRAIN); b = load_credit_info(DATA_CREDIT_INFO_TEST)
    nr, cr = build_raw_feature_pool(a); de4, eng = engineer_business_features(a); num, cat = nr + eng, cr
    X, y = prepare_credit_info(de4, TGT, num, cat, label_offset=1); ids = de4.loc[X.index, KEY].values
    c4 = de4.loc[X.index, CUST].values
    Xtr, Xva, ytr, yva, idtr, idva, ctr, cva = train_test_split(X, y, ids, c4, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    de5, _ = engineer_business_features(b); Xh, yh = prepare_credit_info(de5, TGT, num, cat, label_offset=1)
    id5, c5 = de5.loc[Xh.index, KEY].values, de5.loc[Xh.index, CUST].values
    sets = {"internal test": (Xva, yva, cva), "May seen": (Xh[np.isin(id5, idtr)], yh[np.isin(id5, idtr)], c5[np.isin(id5, idtr)]),
            "May unseen": (Xh[np.isin(id5, idva)], yh[np.isin(id5, idva)], c5[np.isin(id5, idva)]), "May all": (Xh, yh, c5)}
    rng = np.random.RandomState(RANDOM_STATE); rows = []
    for model in ["lightgbm", "xgboost"]:
        for cfg, drop in [("full", []), ("conservative", [OFF, ON])]:
            nn = [c for c in num if c not in drop]
            p = build_pipeline(model, cat, nn, random_state=RANDOM_STATE, **SMOTE_KW).fit(Xtr[nn + cat], ytr)
            for sname, (XX, yy, cc) in sets.items():
                pred = np.asarray(p.predict(XX[nn + cat])).ravel().astype(int)
                codes, inv = np.unique(cc, return_inverse=True)
                labels = sorted(set(yy)); bs = []
                for _ in range(n_boot):  # bootstrap theo cụm khách hàng: trọng số = số lần khách hàng được chọn
                    w = rng.multinomial(len(codes), np.full(len(codes), 1 / len(codes)))[inv]
                    bs.append(f1_score(yy, pred, labels=labels, average="macro", sample_weight=w))
                rows.append({"model": model, "config": cfg, "subset": sname, "n": len(yy), "customers": len(codes),
                             "macro_f1": f1_score(yy, pred, labels=labels, average="macro"),
                             "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5)})
                log(f"BOOT {model} {cfg} {sname}: {rows[-1]['macro_f1']:.4f} [{rows[-1]['ci_low']:.4f}, {rows[-1]['ci_high']:.4f}]")
            save(pd.DataFrame(rows), "boot_task1.csv")


# ── Bài toán 2, đánh giá chính theo khách hàng (StratifiedGroupKFold 5x10) ──────
MAIN = "XGB + overdue fields + G_T"
FULL = OVC + ["G_T"]  # thông tin đầy đủ tại T: 20 trường + quá hạn + nhóm nợ hiện tại
DELTA_PROV = {1: 0.05, 2: 0.15, 3: 0.30, 4: 0.50}  # tăng tỷ lệ trích lập khi lên nhóm kế tiếp (NĐ 86/2024)


def grouped_folds(X, y, groups, repeats=10):
    for r in range(repeats):
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + r).split(X, y, groups):
            yield tr, te


def part_main():
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBRanker
    from src.review_iceir import woe_transform
    X, y, num, cat, d, ov = task2_data(); c = d[CUST].values; mech = ov["MECHANICAL"].values
    cfg = {"XGB, 20 features": [], "XGB, 20 features + G_T": ["G_T"], "XGB + overdue fields": OVC,
           MAIN: FULL, MAIN + " + rule score": FULL + ["RULE_SCORE"]}
    rows, done = [], 0
    if (OUT / "main_folds.csv").exists():  # chạy tiếp từ fold đã lưu (các fold là tất định)
        prev = pd.read_csv(OUT / "main_folds.csv", encoding="utf-8-sig")
        done = int(prev["fold"].max()) + 1; rows = prev.to_dict("records")
        log(f"MAIN: tiếp tục từ fold {done}")
    for f, (tr, te) in enumerate(grouped_folds(X, y, c)):
        if f < done:
            continue
        yte = y[te]; sev = X["G_T"].values[te]; nm = (mech[te] == 0) | (yte == 0)
        s = task2_scores(X, tr, te, y, num, cat, cfg)
        cols = num + FULL + cat
        p = build_pipeline("xgboost", cat, num + FULL, **XGB_KW).fit(X.iloc[tr][cols], y[tr])
        prep = p.named_steps["preprocessor"]
        rk = XGBRanker(objective="rank:pairwise", n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)
        rk.fit(prep.transform(X.iloc[tr][cols]), y[tr], qid=np.zeros(len(tr), int))
        s["XGB pairwise ranking (full information)"] = rk.predict(prep.transform(X.iloc[te][cols]))
        Wtr, Wte = woe_transform(X.iloc[tr][cols], y[tr], X.iloc[te][cols], cat)
        s["WOE-logistic scorecard (full information)"] = LogisticRegression(C=1.0, max_iter=2000).fit(Wtr, y[tr]).predict_proba(Wte)[:, 1]
        s["DPD at T"] = np.maximum(X["DPD_PRINCIPAL"].values, X["DPD_INTEREST"].values)[te]
        for name, col in [("On-balance interest", ON), ("Off-balance interest", OFF)]:
            v = X[col].fillna(0).values
            from sklearn.metrics import roc_auc_score
            s[name] = (1 if roc_auc_score(y[tr], v[tr]) >= 0.5 else -1) * v[te]
        for name, sc in s.items():
            m = rank_metrics(yte, sc, sev)
            rows.append({"fold": f, "method": name, **m,
                         "ap_nonmechanical": average_precision_score(yte[nm], sc[nm]), "events_nonmech": int(yte[nm].sum()),
                         "events": int(yte.sum())})
        if f % 5 == 4:
            log(f"MAIN fold {f + 1}/50"); pd.DataFrame(rows).to_csv(OUT / "main_folds.csv", index=False, encoding="utf-8-sig")
    R = "Rule: distance to next threshold"
    summarise(rows, [(MAIN, R), ("XGB + overdue fields", R), ("XGB, 20 features + G_T", R), ("XGB, 20 features", R),
                     (MAIN, "XGB + overdue fields"), (MAIN + " + rule score", MAIN),
                     (MAIN, "XGB pairwise ranking (full information)"), (MAIN, "WOE-logistic scorecard (full information)")],
              "main")


def part_learners():
    from src.rerun_iceir import _kw, holm
    X, y, num, cat, d, ov = task2_data(); c = d[CUST].values; cols = num + FULL + cat
    specs = [("XGBoost", "xgboost", "none"), ("LightGBM", "lightgbm", "none"), ("LightGBM", "lightgbm", "custom"),
             ("Random Forest", "random_forest", "none"), ("Decision Tree", "decision_tree", "custom"),
             ("Logistic Regression", "logistic", "custom")]
    rows = []
    for f, (tr, te) in enumerate(grouped_folds(X, y, c)):
        for name, key, strat in specs:
            p = build_pipeline(key, cat, num + FULL, **_kw(strat, key)).fit(X.iloc[tr][cols], y[tr])
            rows.append({"fold": f, "model": name, "strategy": strat,
                         "ap": average_precision_score(y[te], p.predict_proba(X.iloc[te][cols])[:, 1])})
        if f % 5 == 4:
            log(f"LRN fold {f + 1}/50"); save(pd.DataFrame(rows), "learners_folds.csv")
    df = pd.DataFrame(rows); save(df, "learners_folds.csv")
    summ = df.groupby(["model", "strategy"])["ap"].agg(["mean", "std"]).reset_index()
    best = summ.loc[summ.groupby("model")["mean"].idxmax()].sort_values("mean", ascending=False)
    ref = best.iloc[0]; va = df[(df.model == ref.model) & (df.strategy == ref.strategy)].sort_values("fold")["ap"].values
    tests = []
    for _, r in best.iloc[1:].iterrows():
        vb = df[(df.model == r.model) & (df.strategy == r.strategy)].sort_values("fold")["ap"].values
        md, p = corrected_ttest(va, vb)
        tests.append({"reference": f"{ref.model} ({ref.strategy})", "comparator": f"{r.model} ({r.strategy})",
                      "comparator_ap": r["mean"], "mean_diff": md, "p_corrected": p, "p_wilcoxon_naive": stats.wilcoxon(va, vb).pvalue})
    tests = pd.DataFrame(tests); tests["p_holm"] = holm(tests["p_corrected"].values)
    save(best, "learners_best.csv"); save(tests, "learners_tests.csv")
    log(f"LRN:\n{best.round(4)}\n{tests.round(4)}")


def part_extra():
    X, y, num, cat, d, ov = task2_data(); c = d[CUST].values
    # (a) bỏ biến lãi khỏi mô hình chính
    base = pd.read_csv(OUT / "main_folds.csv", encoding="utf-8-sig")
    base = base[base.method == MAIN].sort_values("fold")["ap"].values
    rows = []
    for label, drop in [("no_offbalance", [OFF]), ("no_both_interest", [OFF, ON])]:
        nn = [v for v in num if v not in drop]; aps = []
        for tr, te in grouped_folds(X, y, c):
            p = build_pipeline("xgboost", cat, nn + FULL, **XGB_KW).fit(X.iloc[tr][nn + FULL + cat], y[tr])
            aps.append(average_precision_score(y[te], p.predict_proba(X.iloc[te][nn + FULL + cat])[:, 1]))
        md, pv = corrected_ttest(base, aps)
        rows.append({"config": label, "ap_main": base.mean(), "ap": np.mean(aps), "ap_std": np.std(aps, ddof=1), "p_corrected": pv})
        log(f"EXTRA ablation {label}: {np.mean(aps):.4f}")
    save(pd.DataFrame(rows), "extra_interest_ablation.csv")

    # (b) hợp đồng rời danh mục từ nhóm >= 2 được coi là sự kiện
    a, b = load_credit_info(DATA_CREDIT_INFO_TRAIN), load_credit_info(DATA_CREDIT_INFO_TEST)
    ex = a[~a[KEY].isin(b[KEY])].copy(); de_e, _ = engineer_business_features(ex)
    ove = overdue_features(de_e, de_e["ETL_DATE"])
    Xe = de_e[num + cat].copy(); Xe["G_T"] = de_e[TGT].astype(int).values
    for v in OVC:
        Xe[v] = ove[v].values
    ye = (de_e[TGT].astype(int) >= 2).astype(int).values
    cols = num + FULL + cat
    Xa = pd.concat([X[cols], Xe[cols]], ignore_index=True); ya = np.r_[y, ye]; ca = np.r_[c, de_e[CUST].values]
    aps = []
    for tr, te in grouped_folds(Xa, ya, ca):
        p = build_pipeline("xgboost", cat, num + FULL, **XGB_KW).fit(Xa.iloc[tr], ya[tr])
        aps.append(average_precision_score(ya[te], p.predict_proba(Xa.iloc[te])[:, 1]))
    m6 = {"events": int(ya.sum()), "added_exit_events": int(ye.sum()), "ap_mean": np.mean(aps), "ap_std": np.std(aps, ddof=1)}
    save(pd.DataFrame([m6]), "extra_exits.csv"); log(f"EXTRA exits: {m6}")

    # (c) chi phí - lợi ích: điểm out-of-fold (1 lần 5 fold theo khách hàng)
    bal = d["Số dư nợ theo nguyên tệ"].fillna(0).values
    at_stake = np.array([DELTA_PROV.get(g, 0) for g in X["G_T"].values]) * bal * y
    sc = {MAIN: np.zeros(len(y)), "Rule: distance to next threshold": X["RULE_SCORE"].values}
    cols = num + FULL + cat
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X, y, c):
        sc[MAIN][te] = build_pipeline("xgboost", cat, num + FULL, **XGB_KW).fit(X.iloc[tr][cols], y[tr]).predict_proba(X.iloc[te][cols])[:, 1]
    rows = []
    for name, s in sc.items():
        o = np.argsort(-s, kind="mergesort")
        for k in [1, 2, 5, 10]:
            n = int(round(len(y) * k / 100)); top = o[:n]; cap = at_stake[top].sum()
            rows.append({"method": name, "k_pct": k, "n_reviewed": n, "events_found": int(y[top].sum()),
                         "provision_increase_captured": cap, "share_of_total": cap / at_stake.sum(), "per_reviewed_loan": cap / n})
    save(pd.DataFrame(rows), "extra_cost_benefit.csv"); log(f"EXTRA CB:\n{pd.DataFrame(rows).round(3)}")

    # (d) SHAP của mô hình chính trên phép chia 80/20 theo khách hàng
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tr, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X, y, c))
    p = build_pipeline("xgboost", cat, num + FULL, **XGB_KW).fit(X.iloc[tr][cols], y[tr])
    prep, clf = p.named_steps["preprocessor"], p.named_steps["classifier"]
    names = prep.get_feature_names(); Xt = pd.DataFrame(prep.transform(X.iloc[te][cols]), columns=names); yva = y[te]
    rng = np.random.RandomState(RANDOM_STATE); pos = np.where(yva == 1)[0]
    idx = np.r_[pos, rng.choice(np.where(yva == 0)[0], 500 - len(pos), replace=False)]
    sv = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent").shap_values(Xt.iloc[idx])
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    save(imp, "extra_shap.csv"); save(pd.DataFrame([{"events": len(pos), "non_events": 500 - len(pos)}]), "extra_shap_sample.csv")
    en = {"DPD_INTEREST": "Interest DPD", "DPD_PRINCIPAL": "Principal DPD", "OVERDUE_INTEREST": "Overdue interest amount",
          "OVERDUE_PRINCIPAL": "Overdue principal amount", "Số dư nợ theo nguyên tệ": "Outstanding balance", "G_T": "Current debt group",
          "ENG_TENOR_ACTUAL_DAYS": "Actual tenor", "Lãi suất": "Interest rate", "ENG_DAYS_TO_MATURITY": "Days to maturity",
          "ENG_LOAN_AGE_DAYS": "Loan age", "Mã chi nhánh TCTD": "Branch code", ON: "On-balance interest receivable",
          OFF: "Off-balance unpaid interest", "Mô tả mục đích sử dụng tiền vay": "Loan purpose description"}
    fig_dir = Path(__file__).parent.parent.parent / "iceir_paper" / "figures"
    plt.figure()
    shap.summary_plot(sv, Xt.iloc[idx].rename(columns=lambda v: en.get(v, v)), max_display=10, show=False,
                      plot_size=(6.0, 3.6), color_bar_label="Feature value (low to high)")
    plt.xlabel("SHAP value (impact on log-odds of migration)", fontsize=8); plt.gcf().axes[0].tick_params(labelsize=7.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_shap_beeswarm_rev1.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "fig3_shap_beeswarm_rev1.png", dpi=200, bbox_inches="tight")
    log(f"EXTRA SHAP ({len(pos)} sự kiện):\n{imp.head(10)}")


if __name__ == "__main__":
    want = {x.upper() for x in sys.argv[1:]} or {"IV", "T2G", "GRP", "ROB", "BOOT"}
    for k, fn in [("IV", part_iv), ("T2G", part_t2g), ("GRP", part_grp), ("ROB", part_rob), ("BOOT", part_boot),
                  ("MAIN", part_main), ("LRN", part_learners), ("EXTRA", part_extra)]:
        if k in want:
            fn()
    log("HOÀN TẤT")
