"""
Tạo synthetic data giữ nguyên tỉ lệ CLASSIFICATION (1-5).

Phương pháp: Stratified Bootstrap + Gaussian Noise
  1. Giữ toàn bộ records gốc
  2. Với mỗi class: bootstrap sample từ records cùng class
     - Numerical : cộng thêm Gaussian noise (±noise_pct × std của cột)
     - Categorical: giữ nguyên giá trị (sample with replacement)
     - Datetime   : giữ nguyên giá trị (sample with replacement)
  3. Tạo CONTRACT ID mới cho synthetic records
  4. Combine gốc + synthetic, shuffle, lưu ra file

Tỉ lệ class ban đầu (sau khi drop NaN target):
  1: 97.17% | 2: 1.61% | 3: 0.20% | 4: 0.24% | 5: 0.43%
"""
import numpy as np
import pandas as pd
from pathlib import Path


TARGET_COL  = "CLASSIFICATION"
ID_COL      = "CONTRACT"
TOTAL_TARGET = 30_000   # tổng số bản ghi đầu ra
NOISE_PCT    = 0.02     # Gaussian noise = ±2% × std (nhỏ để giữ tính thực tế)
RANDOM_SEED  = 42


def _new_contract_ids(prefix_weights: dict, n: int, existing: set, rng) -> list:
    """Tạo CONTRACT ID mới không trùng với existing."""
    prefixes = list(prefix_weights.keys())
    weights  = np.array(list(prefix_weights.values()), dtype=float)
    weights /= weights.sum()

    ids = []
    counter = 900_000_000
    while len(ids) < n:
        pfx = rng.choice(prefixes, p=weights)
        new_id = f"{pfx}{counter:010d}"
        if new_id not in existing:
            ids.append(new_id)
            existing.add(new_id)
        counter += 1
    return ids


def augment(df_orig: pd.DataFrame,
            total_target: int = TOTAL_TARGET,
            noise_pct: float = NOISE_PCT,
            random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Trả về DataFrame tổng hợp (gốc + synthetic) với ~total_target rows,
    giữ nguyên tỉ lệ CLASSIFICATION.
    """
    rng = np.random.default_rng(random_seed)

    # Bỏ rows không có target
    df = df_orig.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    n_orig = len(df)
    n_synthetic_needed = max(0, total_target - n_orig)
    if n_synthetic_needed == 0:
        print(f"Đã đủ {n_orig} bản ghi, không cần tạo thêm.")
        return df.reset_index(drop=True)

    # Tỉ lệ từng class
    class_counts = df[TARGET_COL].value_counts().sort_index()
    total_valid  = class_counts.sum()
    class_ratio  = class_counts / total_valid

    # Số synthetic cần tạo mỗi class
    synth_per_class = (class_ratio * n_synthetic_needed).round().astype(int)
    # Điều chỉnh để tổng đúng
    diff = n_synthetic_needed - synth_per_class.sum()
    if diff != 0:
        synth_per_class.iloc[0] += diff

    # Phân loại cột
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols  = [c for c in num_cols if c != TARGET_COL]
    cat_cols  = df.select_dtypes(include='object').columns.tolist()
    date_cols = df.select_dtypes(include='datetime').columns.tolist()

    # Std cho từng cột numerical (dùng để scale noise)
    col_stds = {c: df[c].std(skipna=True) for c in num_cols}

    # Prefix weights cho CONTRACT ID mới
    prefix_weights = df[ID_COL].str[:4].value_counts().to_dict() if ID_COL in df.columns else {}
    existing_ids   = set(df[ID_COL].tolist()) if ID_COL in df.columns else set()

    print(f"Dữ liệu gốc: {n_orig:,} | Cần tạo thêm: {n_synthetic_needed:,}")
    print(f"Tỉ lệ synthetic mỗi class:")
    for cls, cnt in synth_per_class.items():
        print(f"  Class {cls}: +{cnt:,} (tổng → {class_counts[cls] + cnt:,}, "
              f"ratio {(class_counts[cls]+cnt)/(total_valid+n_synthetic_needed)*100:.2f}%)")

    synthetic_chunks = []

    for cls, n_synth in synth_per_class.items():
        if n_synth <= 0:
            continue

        pool = df[df[TARGET_COL] == cls].copy()

        # Bootstrap: sample with replacement
        idx = rng.choice(len(pool), size=n_synth, replace=True)
        chunk = pool.iloc[idx].copy().reset_index(drop=True)

        # Gaussian noise cho numerical (chỉ cột không phải ID/SK)
        skip_noise = {c for c in num_cols if
                      c.endswith('_SK') or c.endswith('_ID') or
                      c in (ID_COL, TARGET_COL)}
        for col in num_cols:
            if col in skip_noise:
                continue
            std = col_stds.get(col, 0)
            if std > 0 and chunk[col].notna().any():
                noise = rng.normal(0, noise_pct * std, size=n_synth)
                chunk[col] = chunk[col] + noise
                # Giữ giá trị không âm nếu cột gốc không âm
                if df[col].min() >= 0:
                    chunk[col] = chunk[col].clip(lower=0)

        # Tạo CONTRACT ID mới
        if ID_COL in chunk.columns and prefix_weights:
            chunk[ID_COL] = _new_contract_ids(prefix_weights, n_synth, existing_ids, rng)

        synthetic_chunks.append(chunk)

    df_synthetic = pd.concat(synthetic_chunks, ignore_index=True) if synthetic_chunks else pd.DataFrame()
    df_combined  = pd.concat([df, df_synthetic], ignore_index=True)

    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Verify tỉ lệ
    print(f"\nTổng bản ghi đầu ra: {len(df_combined):,}")
    vc_out = df_combined[TARGET_COL].value_counts().sort_index()
    for cls, cnt in vc_out.items():
        orig_cnt = class_counts.get(cls, 0)
        print(f"  Class {cls}: {cnt:,} ({cnt/len(df_combined)*100:.2f}%)  "
              f"[gốc: {orig_cnt} → tổng {cnt}]")

    return df_combined


def run(input_path: str | Path, output_path: str | Path,
        total_target: int = TOTAL_TARGET, **kwargs):
    print(f"Đọc: {input_path}")
    df_orig = pd.read_excel(input_path)
    print(f"  Gốc: {df_orig.shape[0]:,} rows × {df_orig.shape[1]} cols")

    df_out = augment(df_orig, total_target=total_target, **kwargs)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLưu: {out}")
    if str(out).endswith('.csv'):
        df_out.to_csv(out, index=False, encoding='utf-8-sig')
    else:
        df_out.to_excel(out, index=False)
    print("Hoàn tất.")
    return df_out


if __name__ == "__main__":
    BASE = Path(__file__).parent.parent
    run(
        input_path  = BASE / "data" / "raw" / "fct_l.xlsx",
        output_path = BASE / "data" / "processed" / "fct_l_30k.csv",
        total_target = 30_000,
        noise_pct    = 0.02,
    )
