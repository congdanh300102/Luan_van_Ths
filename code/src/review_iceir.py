"""Phân tích bổ sung theo góp ý bản thảo ICEIR 2026 (bản 14/10/2026).

  M1  Tách tập ngoài thời gian (kỳ 5) thành ba tập con: thuộc tập train tháng 4,
      thuộc tập test tháng 4, hợp đồng mới; tỷ lệ lãi ngoại bảng > 0 theo nhóm
      nợ ở hai kỳ (m0_offbalance_by_group.csv).
  M2  Baseline nghiệp vụ cho cảnh báo sớm (DPD, khoảng cách tới ngưỡng chuyển
      nhóm, đơn biến theo hai trường lãi, scorecard WOE-logistic) và XGBoost có
      thêm các trường quá hạn tại T; tách ca chuyển nhóm cơ học / không cơ học.
  M4  Recall@k, Precision@k trung bình ± độ lệch chuẩn trên 50 fold.
  M5  Learning-to-rank (XGBoost rank:pairwise) và NDCG@5% có trọng số mức độ.
  M6  Độ nhạy: coi hợp đồng rời danh mục từ nhóm >= 2 là sự kiện.

Chỉ dùng hai kỳ dữ liệu thật 20260430 và 20260531.

Chạy:  cd code && python3 -m src.review_iceir
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from xgboost import XGBRanker

from config.config import (
    RANDOM_STATE, TEST_SIZE, DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST,
    CREDIT_INFO_TARGET_COL as TGT, CREDIT_INFO_SMOTE_STRATEGY, TRANSITION_ID_COL as KEY,
)
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info, build_transition_dataset, TRANSITION_LABEL_COL as LAB,
)
from src.models import build_pipeline
from src.rerun_review_fixes import evaluate

OUT = Path(__file__).parent.parent / "results" / "iceir"
OFF, ON = "Lãi chưa thu hạch toán ngoại bảng", "Lãi phải thu hạch toán nội bảng"
LATE_P, LATE_I = "Ngày chậm trả nợ gốc", "Ngày chậm trả nợ lãi"
AMT_P, AMT_I = "Dư nợ gốc chậm trả thực tế", "Số tiền lãi chậm trả thực tế"
NEXT_THRESHOLD = {1: 10, 2: 91, 3: 181, 4: 361}  # ngưỡng DPD của nhóm kế tiếp (TT 31/2024)
KS = [1, 2, 5, 10]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def ymd(s):
    v = pd.to_numeric(s, errors="coerce")
    return pd.to_datetime(v.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def overdue_features(df, etl):
    """DPD gốc/lãi tại ngày chốt số liệu, số tiền quá hạn, khoảng cách tới ngưỡng nhóm kế tiếp."""
    out = pd.DataFrame(index=df.index)
    for name, col in [("DPD_PRINCIPAL", LATE_P), ("DPD_INTEREST", LATE_I)]:
        d = (etl - ymd(df[col])).dt.days
        out[name] = d.clip(lower=0).fillna(0)
    out["OVERDUE_PRINCIPAL"] = pd.to_numeric(df[AMT_P], errors="coerce").fillna(0)
    out["OVERDUE_INTEREST"] = pd.to_numeric(df[AMT_I], errors="coerce").fillna(0)
    out["DPD_MAX"] = out[["DPD_PRINCIPAL", "DPD_INTEREST"]].max(axis=1)
    g = df[TGT].astype(int)
    thr = g.map(NEXT_THRESHOLD).astype(float)
    out["DIST_NEXT"] = (thr - out["DPD_MAX"]).where(g < 5, np.inf)
    # cơ học: đã quá hạn tại T và sẽ vượt ngưỡng nhóm kế tiếp trong 31 ngày nếu không trả
    out["MECHANICAL"] = ((out["DPD_MAX"] > 0) & (out["DIST_NEXT"] <= 31)).astype(int)
    return out


def rank_metrics(y, s, sev=None):
    y = np.asarray(y); s = np.asarray(s, float)
    o = np.argsort(-s, kind="mergesort"); yo = y[o]; P = y.sum()
    m = {"ap": average_precision_score(y, s), "roc": roc_auc_score(y, s)}
    for k in KS:
        n = int(round(len(y) * k / 100)); h = yo[:n].sum()
        m[f"recall@{k}"] = h / P; m[f"precision@{k}"] = h / n
    if sev is not None:  # NDCG@5% với gain = mức độ nghiêm trọng (nhóm xuất phát)
        g = (np.asarray(sev, float) * y)[o]; n = int(round(len(y) * 0.05))
        disc = 1 / np.log2(np.arange(2, n + 2))
        ideal = np.sort(np.asarray(sev, float) * y)[::-1][:n]
        m["ndcg@5"] = float((g[:n] * disc).sum() / (ideal * disc).sum())
    return m


def woe_transform(Xtr, ytr, Xte, cats, bins=10, eps=0.5):
    Ztr, Zte = pd.DataFrame(index=Xtr.index), pd.DataFrame(index=Xte.index)
    for c in Xtr.columns:
        a, b = Xtr[c], Xte[c]
        if c in cats or a.nunique() <= bins:
            ka, kb = a.astype(str), b.astype(str)
        else:
            edges = np.unique(np.quantile(a.dropna(), np.linspace(0, 1, bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            ka = pd.cut(a, edges).astype(str); kb = pd.cut(b, edges).astype(str)
        t = pd.DataFrame({"k": ka.values, "y": ytr})
        g = t.groupby("k")["y"].agg(["sum", "count"])
        bad, good = g["sum"] + eps, g["count"] - g["sum"] + eps
        woe = np.log((bad / bad.sum()) / (good / good.sum()))
        Ztr[c] = ka.map(woe).fillna(0).values; Zte[c] = kb.map(woe).fillna(0).values
    return Ztr, Zte


def offbalance_by_group(a, b):
    """Tỷ lệ khoản vay có lãi ngoại bảng > 0 theo nhóm nợ, kỳ 30/4 và 31/5/2026."""
    rows = []
    for name, d in [("2026-04-30", a), ("2026-05-31", b)]:
        for g in range(1, 6):
            sub = d[d[TGT].astype(int) == g]
            rows.append({"snapshot": name, "group": g, "n": len(sub), "pct_offbalance_pos": (sub[OFF] > 0).mean() * 100})
    save(pd.DataFrame(rows), "m0_offbalance_by_group.csv")


# ── M1 ───────────────────────────────────────────────────────────────────────
def m1(a, b):
    nr, cr = build_raw_feature_pool(a); de4, eng = engineer_business_features(a); num, cat = nr + eng, cr
    X, y = prepare_credit_info(de4, TGT, num, cat, label_offset=1); ids = de4.loc[X.index, KEY].values
    Xtr, Xva, ytr, yva, idtr, idva = train_test_split(X, y, ids, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    de5, _ = engineer_business_features(b); Xh, yh = prepare_credit_info(de5, TGT, num, cat, label_offset=1)
    id5 = de5.loc[Xh.index, KEY].values
    sets = {"internal test (Apr, 20%)": (Xva, yva), "May: in April training set": (Xh[np.isin(id5, idtr)], yh[np.isin(id5, idtr)]),
            "May: in April internal test": (Xh[np.isin(id5, idva)], yh[np.isin(id5, idva)]),
            "May: new contracts": (Xh[~np.isin(id5, ids)], yh[~np.isin(id5, ids)]), "May: all": (Xh, yh)}
    rows = []
    for model in ["lightgbm", "xgboost"]:
        for cfg, drop in [("full", []), ("conservative", [OFF, ON])]:
            nn = [c for c in num if c not in drop]
            p = build_pipeline(model, cat, nn, random_state=RANDOM_STATE, imbalance_strategy="custom",
                               custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY).fit(Xtr[nn + cat], ytr)
            for sname, (XX, yy) in sets.items():
                pred = np.asarray(p.predict(XX[nn + cat])).ravel().astype(int)
                present = sorted(set(yy))
                rows.append({"model": model, "config": cfg, "subset": sname, "n": len(yy),
                             "groups_present": "".join(str(g + 1) for g in present),
                             "macro_f1_present": f1_score(yy, pred, labels=present, average="macro"),
                             "macro_f1_5class": f1_score(yy, pred, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0),
                             "accuracy": float((pred == yy).mean()),
                             **{f"n_g{g+1}": int((yy == g).sum()) for g in range(5)}})
            log(f"M1 {model} {cfg} xong")
    save(pd.DataFrame(rows), "m1_oot_subsets.csv")
    log(f"M1:\n{pd.DataFrame(rows)[['model','config','subset','n','macro_f1_present','macro_f1_5class']].round(4)}")


# ── M2/M4/M5 ─────────────────────────────────────────────────────────────────
def task2_review(a, b):
    dm = build_transition_dataset(a, b, id_col=KEY, target_col=TGT)
    nr, cr = build_raw_feature_pool(dm); de, eng = engineer_business_features(dm); num, cat = nr + eng, cr
    X, y = prepare_credit_info(de, LAB, num, cat, label_offset=0)
    d = de.loc[X.index]
    ov = overdue_features(d, d["ETL_DATE"]); g0 = d[TGT].astype(int).values; mech = ov["MECHANICAL"].values
    log(f"T2-review: {len(y)} khoản vay, {y.sum()} sự kiện; sự kiện cơ học {int(mech[y == 1].sum())}/{int(y.sum())}; "
        f"khoản vay đang quá hạn tại T: {int((ov['DPD_MAX'] > 0).sum())}")
    OVC = ["DPD_PRINCIPAL", "DPD_INTEREST", "OVERDUE_PRINCIPAL", "OVERDUE_INTEREST"]
    Xd = pd.concat([X, ov[OVC]], axis=1)
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    rows = []
    for f, (tr, te) in enumerate(rcv.split(X, y)):
        ytr, yte = y[tr], y[te]; sev = g0[te]
        scores = {}
        p = build_pipeline("xgboost", cat, num, random_state=RANDOM_STATE, imbalance_strategy="none", eval_metric="logloss")
        p.fit(X.iloc[tr], ytr); scores["XGBoost (20 features)"] = p.predict_proba(X.iloc[te])[:, 1]
        prep = p.named_steps["preprocessor"]
        Ztr, Zte = prep.transform(X.iloc[tr]), prep.transform(X.iloc[te])
        rk = XGBRanker(objective="rank:pairwise", n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)
        rk.fit(Ztr, ytr, qid=np.zeros(len(ytr), int)); scores["XGBoost rank:pairwise"] = rk.predict(Zte)
        pd_ = build_pipeline("xgboost", cat, num + OVC, random_state=RANDOM_STATE, imbalance_strategy="none", eval_metric="logloss")
        pd_.fit(Xd.iloc[tr], ytr); scores["XGBoost + overdue fields"] = pd_.predict_proba(Xd.iloc[te])[:, 1]
        Wtr, Wte = woe_transform(X.iloc[tr], ytr, X.iloc[te], cat)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Wtr, ytr); scores["WOE-logistic scorecard"] = lr.predict_proba(Wte)[:, 1]
        scores["DPD at T"] = ov["DPD_MAX"].values[te]
        scores["Distance to next threshold"] = -ov["DIST_NEXT"].replace(np.inf, 1e6).values[te]
        for name, col in [("On-balance interest receivable", ON), ("Off-balance unpaid interest", OFF)]:
            v = X[col].fillna(0).values
            sign = 1 if roc_auc_score(ytr, v[tr]) >= 0.5 else -1
            scores[name] = sign * v[te]
        nm = (mech[te] == 0) | (yte == 0)  # loại các sự kiện cơ học
        for name, s in scores.items():
            m = rank_metrics(yte, s, sev)
            m_nm = average_precision_score(yte[nm], s[nm]) if yte[nm].sum() > 0 else np.nan
            rows.append({"fold": f, "method": name, **m, "ap_nonmechanical": m_nm, "events_nonmech": int(yte[nm].sum())})
        if f % 10 == 9:
            log(f"T2-review fold {f + 1}/50")
            save(pd.DataFrame(rows), "m2_folds.csv")
    df = pd.DataFrame(rows); save(df, "m2_folds.csv")
    agg = df.drop(columns="fold").groupby("method").agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index().sort_values("ap_mean", ascending=False)
    save(agg, "m2_baselines.csv")
    log(f"M2/M4/M5:\n{agg[['method','ap_mean','ap_std','recall@5_mean','recall@5_std','precision@5_mean','ndcg@5_mean','ap_nonmechanical_mean']].round(4)}")

    # M6: hợp đồng rời danh mục từ nhóm >= 2 được coi là sự kiện
    exits = a[~a[KEY].isin(b[KEY])].copy()
    de_e, _ = engineer_business_features(exits)
    Xe = de_e[num + cat]; ye = (de_e[TGT].astype(int) >= 2).astype(int).values
    Xa = pd.concat([X, Xe]); ya = np.r_[y, ye]
    aps = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(Xa, ya):
        p = build_pipeline("xgboost", cat, num, random_state=RANDOM_STATE, imbalance_strategy="none", eval_metric="logloss")
        p.fit(Xa.iloc[tr], ya[tr]); aps.append(average_precision_score(ya[te], p.predict_proba(Xa.iloc[te])[:, 1]))
    m6 = {"n": len(ya), "events": int(ya.sum()), "added_exit_events": int(ye.sum()), "exits": len(ye),
          "base_rate": ya.mean(), "ap_mean": float(np.mean(aps)), "ap_std": float(np.std(aps, ddof=1))}
    save(pd.DataFrame([m6]), "m6_exits_sensitivity.csv")
    log(f"M6: {m6}")


if __name__ == "__main__" and "EXTRA" not in {x.upper() for x in sys.argv[1:]}:
    a, b = load_credit_info(DATA_CREDIT_INFO_TRAIN), load_credit_info(DATA_CREDIT_INFO_TEST)
    want = {x.upper() for x in sys.argv[1:]} or {"M1", "T2"}
    if "M1" in want:
        offbalance_by_group(a, b)
        m1(a, b)
    if "T2" in want:
        task2_review(a, b)
    log("HOÀN TẤT")


# ── Bổ sung: mô hình chính mới của Bài toán 2 (XGBoost + các trường quá hạn tại T) ──
def main_model_extras(a, b):
    import shap
    from scipy import stats
    dm = build_transition_dataset(a, b, id_col=KEY, target_col=TGT)
    nr, cr = build_raw_feature_pool(dm); de, eng = engineer_business_features(dm); num, cat = nr + eng, cr
    X, y = prepare_credit_info(de, LAB, num, cat, label_offset=0)
    d = de.loc[X.index]; ov = overdue_features(d, d["ETL_DATE"])
    OVC = ["DPD_PRINCIPAL", "DPD_INTEREST", "OVERDUE_PRINCIPAL", "OVERDUE_INTEREST"]
    Xd = pd.concat([X, ov[OVC]], axis=1); numd = num + OVC
    full = pd.read_csv(OUT / "m2_folds.csv", encoding="utf-8-sig")
    full = full[full.method == "XGBoost + overdue fields"].sort_values("fold")["ap"].values
    rows = []
    for label, drop in [("no_offbalance", [OFF]), ("no_both_interest", [OFF, ON])]:
        nn = [c for c in numd if c not in drop]; aps = []
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(Xd, y):
            p = build_pipeline("xgboost", cat, nn, random_state=RANDOM_STATE, imbalance_strategy="none", eval_metric="logloss")
            p.fit(Xd.iloc[tr][nn + cat], y[tr]); aps.append(average_precision_score(y[te], p.predict_proba(Xd.iloc[te][nn + cat])[:, 1]))
        dd = full - np.asarray(aps); k = len(dd); t = dd.mean() / np.sqrt((1 / k + 0.25) * dd.var(ddof=1))
        rows.append({"config": label, "ap_full": full.mean(), "ap_mean": np.mean(aps), "ap_std": np.std(aps, ddof=1),
                     "p_corrected": 2 * stats.t.sf(abs(t), k - 1)})
        log(f"EXTRA ablation {label}: AP {np.mean(aps):.4f} (đầy đủ {full.mean():.4f})")
    save(pd.DataFrame(rows), "m2_main_interest_ablation.csv")
    Xtr, Xva, ytr, yva = train_test_split(Xd, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    p = build_pipeline("xgboost", cat, numd, random_state=RANDOM_STATE, imbalance_strategy="none", eval_metric="logloss").fit(Xtr, ytr)
    prep, clf = p.named_steps["preprocessor"], p.named_steps["classifier"]
    names = prep.get_feature_names(); Xt = pd.DataFrame(prep.transform(Xva), columns=names)
    sv = np.asarray(shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent").shap_values(Xt))
    rng = np.random.RandomState(RANDOM_STATE); pos = np.where(yva == 1)[0]
    idx = np.r_[pos, rng.choice(np.where(yva == 0)[0], 500 - len(pos), replace=False)]
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv[idx]).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    save(imp, "m2_main_shap.csv"); log(f"EXTRA SHAP:\n{imp.head(10)}")


if __name__ == "__main__" and "EXTRA" in {x.upper() for x in sys.argv[1:]}:
    main_model_extras(load_credit_info(DATA_CREDIT_INFO_TRAIN), load_credit_info(DATA_CREDIT_INFO_TEST))
