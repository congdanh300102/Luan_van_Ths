"""Tái tạo toàn bộ kết quả của bài báo ICEIR 2026 từ HAI kỳ dữ liệu thật
(20260430, 20260531), sau khi chuẩn hóa ngày của tệp 20260531 về yyyymmdd.

Không dùng bộ dữ liệu A.
Kết quả ghi vào code/results/iceir/ (không ghi đè kết quả cũ của luận văn).

  T1  Bài toán 5 nhóm: 6 mô hình (nội bộ + ngoài thời gian), theo nhóm, ma trận
      nhầm lẫn, loại biến lãi, 4 chiến lược mất cân bằng, SMOTE-NC, mã chi nhánh,
      Top-k, Gain/SHAP theo nhóm, lưới siêu tham số (XGBoost, LightGBM), IV, PSI.
  T2  Bài toán chuyển nhóm 31 ngày: 6 mô hình x 4 chiến lược (5x10 CV), kiểm định
      t hiệu chỉnh + Holm, Recall@k, AP theo loại chuyển nhóm, SHAP phân tầng, IV,
      và PHÂN TÍCH MỚI: độ vững khi bỏ biến lãi (giả định tính sẵn có tại T).

Chạy:  cd code && python3 -m src.rerun_iceir [T1] [T2]
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from config.config import (
    RANDOM_STATE, TEST_SIZE, DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST,
    CREDIT_INFO_TARGET_COL, CREDIT_INFO_SMOTE_STRATEGY, TRANSITION_ID_COL,
    TRANSITION_SMOTE_STRATEGY, TRANSITION_CV_SPLITS, TRANSITION_CV_REPEATS,
)
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info, build_transition_dataset, feature_to_group, TRANSITION_LABEL_COL,
)
from src.models import build_pipeline, compute_sample_weights
from src.iv_analysis import compute_iv_table
from src.evaluation import repeated_stratified_binary_eval
from src.rerun_review_fixes import load_b, evaluate, GRID_B, MODELS

OUT = Path(__file__).parent.parent / "results" / "iceir"
OUT.mkdir(parents=True, exist_ok=True)
OFF, ON = "Lãi chưa thu hạch toán ngoại bảng", "Lãi phải thu hạch toán nội bảng"
SMOTE_KW = {"imbalance_strategy": "custom", "custom_smote_strategy": CREDIT_INFO_SMOTE_STRATEGY}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def psi(a, b, bins=10):
    """Population Stability Index giữa hai mẫu (biến số: 10 khoảng phân vị của a)."""
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if a.dtype.kind in "biuf" and a.nunique() > bins:
        edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        pa = pd.cut(a, edges).value_counts(normalize=True, sort=False)
        pb = pd.cut(b, edges).value_counts(normalize=True, sort=False)
    else:
        cats = sorted(set(a.astype(str)) | set(b.astype(str)))
        pa = a.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
        pb = b.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    pa, pb = pa.values + 1e-6, pb.values + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))


# ── T1: bài toán 5 nhóm ──────────────────────────────────────────────────────
def task1():
    X, y, Xh, yh, num_p, cat_p = load_b()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    log(f"T1 dữ liệu: train {len(ytr)}, nội bộ {len(yva)}, ngoài thời gian {len(yh)}; "
        f"thiếu ENG ngoài TG: {Xh[[c for c in num_p if c.startswith('ENG_')]].isna().mean().round(4).to_dict()}")

    rows, byc, cms, fitted = [], [], [], {}
    for name, key in MODELS.items():
        p = build_pipeline(key, cat_p, num_p, random_state=RANDOM_STATE, **SMOTE_KW).fit(Xtr, ytr)
        fitted[name] = p
        for split, XX, yy in [("validation", Xva, yva), ("holdout", Xh, yh)]:
            m, cm, (pr, rc, f1, sup) = evaluate(p, XX, yy)
            off2 = (cm * (np.abs(np.arange(5)[:, None] - np.arange(5)) >= 2)).sum() / cm.sum()
            rows.append({"model": name, "split": split, **m, "share_off_by_2plus": off2})
            byc += [{"model": name, "split": split, "nhom": i + 1, "support": int(sup[i]),
                     "precision": pr[i], "recall": rc[i], "f1": f1[i]} for i in range(5)]
            cms.append(pd.DataFrame(cm, index=[f"true_{i}" for i in range(1, 6)],
                                    columns=[f"pred_{i}" for i in range(1, 6)]).assign(model=name, split=split))
        log(f"T1 {name}: nội bộ {rows[-2]['f1_macro']:.4f} | ngoài TG {rows[-1]['f1_macro']:.4f}")
    save(pd.DataFrame(rows), "t1_models.csv"); save(pd.DataFrame(byc), "t1_by_class.csv")
    pd.concat(cms).to_csv(OUT / "t1_confusion.csv", encoding="utf-8-sig")

    # Gain & SHAP (LightGBM, 5.000 quan sát nội bộ)
    p = fitted["LightGBM"]; prep, clf = p.named_steps["preprocessor"], p.named_steps["classifier"]
    names = prep.get_feature_names()
    gain = pd.Series(clf.booster_.feature_importance(importance_type="gain"), index=names)
    gain = gain / gain.sum() * 100
    Xs = pd.DataFrame(prep.transform(Xva), columns=names).sample(5000, random_state=RANDOM_STATE)
    sv = np.asarray(shap.TreeExplainer(clf).shap_values(Xs))
    imp = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
    shp = pd.Series(imp, index=names)
    d = pd.DataFrame({"feature": names, "gain_pct": gain.values, "shap_pct": (shp / shp.sum() * 100).values})
    d["group"] = [feature_to_group(f) for f in names]
    save(d.sort_values("gain_pct", ascending=False), "t1_gain_shap_feature.csv")
    g = d.groupby("group")[["gain_pct", "shap_pct"]].sum().reset_index()
    save(g.sort_values("gain_pct", ascending=False), "t1_gain_shap_group.csv")
    sf, sg = stats.spearmanr(d.gain_pct, d.shap_pct), stats.spearmanr(g.gain_pct, g.shap_pct)
    save(pd.DataFrame([{"level": "feature", "rho": sf.statistic, "p": sf.pvalue},
                       {"level": "group", "rho": sg.statistic, "p": sg.pvalue}]), "t1_spearman.csv")
    log(f"T1 Gain/SHAP: ρ biến {sf.statistic:.3f}, ρ nhóm {sg.statistic:.3f} (p={sg.pvalue:.3f})")

    # Top-k theo Gain
    order = d.sort_values("gain_pct", ascending=False)["feature"].tolist()
    rows = []
    for k in [3, 5, 8, 11, 14, 17, 20]:
        sel = order[:k]; ck, nk = [c for c in cat_p if c in sel], [c for c in num_p if c in sel]
        pk = build_pipeline("lightgbm", ck, nk, random_state=RANDOM_STATE, **SMOTE_KW).fit(Xtr[nk + ck], ytr)
        rows.append({"k": k, "q_k_pct": d.set_index("feature").loc[sel, "gain_pct"].sum(),
                     **evaluate(pk, Xva[nk + ck], yva)[0]})
    save(pd.DataFrame(rows), "t1_topk.csv")

    # Loại biến lãi
    rows = []
    for label, drop in [("full", []), ("no_offbalance", [OFF]), ("no_both_interest", [OFF, ON])]:
        num = [c for c in num_p if c not in drop]
        pl = build_pipeline("lightgbm", cat_p, num, random_state=RANDOM_STATE, **SMOTE_KW).fit(Xtr[num + cat_p], ytr)
        for split, XX, yy in [("validation", Xva, yva), ("holdout", Xh, yh)]:
            rows.append({"config": label, "split": split, **evaluate(pl, XX[num + cat_p], yy)[0]})
    save(pd.DataFrame(rows), "t1_leak.csv")

    # Chiến lược mất cân bằng (LightGBM, nội bộ)
    rows = []
    for label, strat in [("SMOTE", "custom"), ("none", "none"), ("combined", "smote_class_weight"), ("class_weight", "class_weight")]:
        pi = build_pipeline("lightgbm", cat_p, num_p, random_state=RANDOM_STATE, imbalance_strategy=strat,
                            custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY).fit(Xtr, ytr)
        m, _c, (pr, rc, f1, sup) = evaluate(pi, Xva, yva)
        rows.append({"strategy": label, **m, **{f"recall_n{i+1}": rc[i] for i in range(5)}})
    save(pd.DataFrame(rows), "t1_imbalance.csv")

    # SMOTE-NC và mã chi nhánh
    rows = []
    for name in ["LightGBM", "XGBoost"]:
        ps = build_pipeline(MODELS[name], cat_p, num_p, random_state=RANDOM_STATE, imbalance_strategy="smote_nc",
                            custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY).fit(Xtr, ytr)
        for split, XX, yy in [("validation", Xva, yva), ("holdout", Xh, yh)]:
            rows.append({"variant": "smote_nc", "model": name, "split": split, **evaluate(ps, XX, yy)[0]})
    Xb, yb, Xhb, yhb, num_b, cat_b = load_b(branch_as_categorical=True)
    Xbtr, Xbva, ybtr, ybva = train_test_split(Xb, yb, test_size=TEST_SIZE, stratify=yb, random_state=RANDOM_STATE)
    for name in ["LightGBM", "XGBoost", "Logistic Regression"]:
        pb = build_pipeline(MODELS[name], cat_b, num_b, random_state=RANDOM_STATE, **SMOTE_KW).fit(Xbtr, ybtr)
        for split, XX, yy in [("validation", Xbva, ybva), ("holdout", Xhb, yhb)]:
            rows.append({"variant": "branch_categorical", "model": name, "split": split, **evaluate(pb, XX, yy)[0]})
    save(pd.DataFrame(rows), "t1_encoding.csv")

    # Lưới siêu tham số (XGBoost, LightGBM)
    rows = []
    for name in ["XGBoost", "LightGBM"]:
        sc = [evaluate(build_pipeline(MODELS[name], cat_p, num_p, random_state=RANDOM_STATE, model_params=prm,
                                      **SMOTE_KW).fit(Xtr, ytr), Xva, yva)[0]["f1_macro"] for prm in GRID_B[name]]
        rows.append({"model": name, "n": len(sc), "best": max(sc), "worst": min(sc), "range": max(sc) - min(sc)})
        log(f"T1 lưới {name}: biên độ {max(sc) - min(sc):.4f}")
    save(pd.DataFrame(rows), "t1_grid.csv")

    # IV (toàn mẫu tháng 4), tỷ lệ dương của hai biến lãi, PSI tháng 4 -> tháng 5
    df4 = load_credit_info(DATA_CREDIT_INFO_TRAIN); de4, eng = engineer_business_features(df4)
    df5 = load_credit_info(DATA_CREDIT_INFO_TEST); de5, _ = engineer_business_features(df5)
    feats = num_p + cat_p
    save(compute_iv_table(de4[feats + [CREDIT_INFO_TARGET_COL]], CREDIT_INFO_TARGET_COL), "t1_iv.csv")
    pos = [{"var": v, "group": gname, "pct_positive": float((de4.loc[de4[CREDIT_INFO_TARGET_COL] == gname, v] > 0).mean() * 100)}
           for v in [OFF, ON] for gname in range(1, 6)]
    pos += [{"var": OFF, "group": "share_zero", "pct_positive": float((de4[OFF] == 0).mean() * 100)}]
    save(pd.DataFrame(pos), "t1_interest_positive.csv")
    ps_rows = [{"feature": f, "psi_apr_may": psi(de4[f], de5[f]),
                "missing_apr": float(de4[f].isna().mean()), "missing_may": float(de5[f].isna().mean())} for f in feats]
    save(pd.DataFrame(ps_rows).sort_values("psi_apr_may", ascending=False), "t1_psi.csv")
    log(f"T1 PSI lớn nhất: {max(r['psi_apr_may'] for r in ps_rows):.4f}")


# ── T2: chuyển nhóm 31 ngày ─────────────────────────────────────────────────
def _kw(strategy, key):
    kw = {"random_state": RANDOM_STATE, "imbalance_strategy": strategy}
    if strategy in ("custom", "smote_class_weight"):
        kw["custom_smote_strategy"] = TRANSITION_SMOTE_STRATEGY
    if key == "catboost":
        kw["loss_function"] = "Logloss"
    if key == "xgboost":
        kw["eval_metric"] = "logloss"
    return kw


def _sw(strategy, key):
    return compute_sample_weights if (strategy == "class_weight" and key == "xgboost") else None


def corrected_ttest(a, b, ratio=0.25):
    d = np.asarray(a) - np.asarray(b); k = len(d)
    t = d.mean() / np.sqrt((1 / k + ratio) * d.var(ddof=1))
    return d.mean(), t, 2 * stats.t.sf(abs(t), df=k - 1)


def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); out = np.empty_like(ps); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i])); out[i] = run
    return out


def task2():
    a, b = load_credit_info(DATA_CREDIT_INFO_TRAIN), load_credit_info(DATA_CREDIT_INFO_TEST)
    dm = build_transition_dataset(a, b, id_col=TRANSITION_ID_COL, target_col=CREDIT_INFO_TARGET_COL)
    nr, cr = build_raw_feature_pool(dm); de, eng = engineer_business_features(dm)
    num_p, cat_p = nr + eng, cr
    X, y = prepare_credit_info(de, TRANSITION_LABEL_COL, num_p, cat_p, label_offset=0)
    log(f"T2: {len(y)} khoản vay, {int(y.sum())} sự kiện")
    save(compute_iv_table(de[num_p + cat_p + [TRANSITION_LABEL_COL]], TRANSITION_LABEL_COL, good_class=0), "t2_iv.csv")

    # Tiếp tục từ kết quả đã lưu (mỗi cấu hình được lưu ngay khi chạy xong).
    # CatBoost chỉ chạy "none": 50 lần huấn luyện mất ~4,6 giờ trên máy hiện tại,
    # và ở lần chạy trước "class_weight" kém hơn "none" (AP 0,119 so với 0,128).
    per, summ = [], []
    if (OUT / "t2_cv_summary.csv").exists():
        summ = pd.read_csv(OUT / "t2_cv_summary.csv").to_dict("records")
        per = [g for _, g in pd.read_csv(OUT / "t2_cv_per_fold.csv").groupby(["model", "strategy"])]
    done = {(r["model"], r["strategy"]) for r in summ}
    for name, key in MODELS.items():
        for strat in (["none"] if key == "catboost" else ["none", "class_weight", "custom", "smote_class_weight"]):
            if (name, strat) in done:
                continue
            t0 = time.time()
            pr, s = repeated_stratified_binary_eval(build_pipeline(key, cat_p, num_p, **_kw(strat, key)), X, y,
                                                    n_splits=TRANSITION_CV_SPLITS, n_repeats=TRANSITION_CV_REPEATS,
                                                    random_state=RANDOM_STATE, sample_weight_fn=_sw(strat, key))
            per.append(pr.assign(model=name, strategy=strat)); summ.append({"model": name, "strategy": strat, **s})
            log(f"T2 {name} ({strat}): AP {s['pr_auc_mean']:.4f}±{s['pr_auc_std']:.4f} ROC {s['roc_auc_mean']:.4f} ({time.time()-t0:.0f}s)")
            save(pd.DataFrame(summ), "t2_cv_summary.csv"); pd.concat(per).to_csv(OUT / "t2_cv_per_fold.csv", index=False, encoding="utf-8-sig")
    summ = pd.DataFrame(summ); per = pd.concat(per)
    best = summ.loc[summ.groupby("model")["pr_auc_mean"].idxmax()].sort_values("pr_auc_mean", ascending=False)
    top = best.iloc[0]
    ref = per[(per.model == top.model) & (per.strategy == top.strategy)]["pr_auc"].values
    rows = []
    for _, r in best.iloc[1:].iterrows():
        oth = per[(per.model == r.model) & (per.strategy == r.strategy)]["pr_auc"].values
        md, t, p = corrected_ttest(ref, oth)
        rows.append({"reference": f"{top.model} ({top.strategy})", "comparator": f"{r.model} ({r.strategy})",
                     "mean_diff": md, "t": t, "p_corrected": p, "p_wilcoxon_naive": stats.wilcoxon(ref, oth).pvalue})
    tests = pd.DataFrame(rows); tests["p_holm"] = holm(tests["p_corrected"]); save(tests, "t2_tests.csv")
    log(f"T2 kiểm định:\n{tests}")

    # Chia 80/20: Recall@k, AP theo loại chuyển nhóm, SHAP phân tầng (mô hình tốt nhất)
    key = MODELS[top.model]
    Xtr, Xva, ytr, yva, dtr, dva = train_test_split(X, y, de.loc[X.index], test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    p = build_pipeline(key, cat_p, num_p, **_kw(top.strategy, key))
    sw = _sw(top.strategy, key)
    p.fit(Xtr, ytr, **({"classifier__sample_weight": sw(ytr)} if sw else {}))
    s = p.predict_proba(Xva)[:, 1]
    rk = []
    order = np.argsort(-s); yo = yva[order]
    for k in [1, 2, 5, 10]:
        n = int(round(len(yva) * k / 100)); h = int(yo[:n].sum())
        rk.append({"k_pct": k, "n_reviewed": n, "found": h, "recall": h / yva.sum(), "precision": h / n, "lift": h / yva.sum() / (k / 100)})
    rk = pd.DataFrame(rk); rk.attrs = {}
    save(rk.assign(model=top.model, strategy=top.strategy, ap=average_precision_score(yva, s),
                   roc=roc_auc_score(yva, s), n=len(yva), events=int(yva.sum())), "t2_recall_at_k.csv")
    g0 = dva["Nhóm nợ tự phân loại"].astype(int).values
    types = []
    for label, mask in [("all", np.ones(len(yva), bool)), ("1to2", (g0 == 1) | (yva == 0)), ("rarer", (g0 != 1) | (yva == 0))]:
        yy = yva[mask] if label == "all" else np.where(mask, yva, 0)[mask]
        if label == "1to2":
            yy = ((yva == 1) & (g0 == 1))[mask].astype(int)
        if label == "rarer":
            yy = ((yva == 1) & (g0 != 1))[mask].astype(int)
        types.append({"type": label, "n": int(mask.sum()), "events": int(yy.sum()), "ap": average_precision_score(yy, s[mask]),
                      "base": yy.mean(), "roc": roc_auc_score(yy, s[mask])})
    save(pd.DataFrame(types), "t2_ap_by_type.csv")
    log(f"T2 Recall@k:\n{rk}\nAP theo loại: {pd.DataFrame(types)[['type','events','ap','base']].to_dict('records')}")

    prep, clf = p.named_steps["preprocessor"], p.named_steps["classifier"]
    names = prep.get_feature_names(); Xt = pd.DataFrame(prep.transform(Xva), columns=names)
    sv = np.asarray(shap.TreeExplainer(clf).shap_values(Xt))
    if sv.ndim == 3:
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
    rng = np.random.RandomState(RANDOM_STATE)
    pos_idx = np.where(yva == 1)[0]; neg_idx = rng.choice(np.where(yva == 0)[0], 500 - len(pos_idx), replace=False)
    rnd_idx = rng.choice(len(yva), 500, replace=False)
    rows = []
    for lab, idx in [("events", pos_idx), ("non_events", neg_idx), ("stratified_500", np.r_[pos_idx, neg_idx]), ("random_500", rnd_idx)]:
        rows += [{"sample": lab, "n": len(idx), "feature": f, "mean_abs_shap": v} for f, v in zip(names, np.abs(sv[idx]).mean(axis=0))]
    save(pd.DataFrame(rows), "t2_shap.csv")

    # PHÂN TÍCH MỚI: bỏ biến lãi (giả định tính sẵn có tại T)
    rows, per_ab = [], []
    for name in ["XGBoost", "LightGBM", "Random Forest"]:
        strat = best.set_index("model").loc[name, "strategy"]; key = MODELS[name]
        full = per[(per.model == name) & (per.strategy == strat)]["pr_auc"].values
        for label, drop in [("no_offbalance", [OFF]), ("no_both_interest", [OFF, ON])]:
            num = [c for c in num_p if c not in drop]
            pr, s2 = repeated_stratified_binary_eval(build_pipeline(key, cat_p, num, **_kw(strat, key)), X[num + cat_p], y,
                                                     n_splits=TRANSITION_CV_SPLITS, n_repeats=TRANSITION_CV_REPEATS,
                                                     random_state=RANDOM_STATE, sample_weight_fn=_sw(strat, key))
            md, t, pv = corrected_ttest(full, pr["pr_auc"].values)
            rows.append({"model": name, "strategy": strat, "config": label, "ap_full": full.mean(), **s2,
                         "ap_drop": md, "t": t, "p_corrected": pv})
            per_ab.append(pr.assign(model=name, strategy=strat, config=label))
            log(f"T2 ablation {name} {label}: AP {s2['pr_auc_mean']:.4f} (đầy đủ {full.mean():.4f}, p={pv:.3g})")
            save(pd.DataFrame(rows), "t2_interest_ablation.csv")
    # Recall@k khi bỏ cả hai biến lãi (mô hình tốt nhất, cùng phép chia 80/20)
    num = [c for c in num_p if c not in (OFF, ON)]
    p2 = build_pipeline(key := MODELS[top.model], cat_p, num, **_kw(top.strategy, key))
    p2.fit(Xtr[num + cat_p], ytr, **({"classifier__sample_weight": sw(ytr)} if sw else {}))
    s2 = p2.predict_proba(Xva[num + cat_p])[:, 1]; yo2 = yva[np.argsort(-s2)]
    rk2 = [{"k_pct": k, "found": int(yo2[:int(round(len(yva) * k / 100))].sum()),
            "recall": yo2[:int(round(len(yva) * k / 100))].sum() / yva.sum()} for k in [1, 2, 5, 10]]
    save(pd.DataFrame(rk2).assign(ap=average_precision_score(yva, s2)), "t2_recall_at_k_no_interest.csv")
    log(f"T2 Recall@k không biến lãi: {rk2}")


if __name__ == "__main__":
    want = {a.upper() for a in sys.argv[1:]} or {"T1", "T2"}
    if "T1" in want:
        task1()
    if "T2" in want:
        task2()
    log("HOÀN TẤT")
