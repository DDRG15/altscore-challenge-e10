# ============================================================
# E10 - Phase 2 only: aggregate + model (batches already in Drive)
# Paste into a NEW cell. Run AFTER stopping the previous script.
# All 35 batch files are already saved in Drive/AltScore challange/batches/
# This script processes ONE batch at a time — never loads all 340M at once.
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

import os, gc
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import subprocess
subprocess.run(["pip", "install", "h3", "-q"])
import h3

DRIVE_DIR  = "/content/drive/MyDrive/AltScore challange"
BATCH_DIR  = os.path.join(DRIVE_DIR, "batches")
OUT_FILE   = os.path.join(DRIVE_DIR, "submission.csv")
HEX_COL    = "hex_id"
TARGET_COL = "cost_of_living"

# Load train/test
train = pd.read_csv(os.path.join(DRIVE_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DRIVE_DIR, "test.csv"))
TRAIN_RES = h3.get_resolution(train[HEX_COL].dropna().iloc[0])
print(f"Train: {train.shape} | Test: {test.shape} | H3 res: {TRAIN_RES}")
print(f"Baseline RMSE: {train[TARGET_COL].std():.4f}")

batch_files = sorted([
    os.path.join(BATCH_DIR, f)
    for f in os.listdir(BATCH_DIR)
    if f.startswith("batch_") and f.endswith(".parquet")
])
print(f"Found {len(batch_files)} batch files in Drive\n")


# ── Aggregate ONE batch at a time ───────────────────────────
def process_batch(batch_path, agg_res):
    df = pd.read_parquet(batch_path)
    df["agg_hex"] = [h3.latlng_to_cell(float(la), float(lo), agg_res)
                     for la, lo in zip(df["lat"].to_numpy(), df["lon"].to_numpy())]
    df["dt"]         = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df["hour"]       = df["dt"].dt.hour.astype("int8")
    df["dow"]        = df["dt"].dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["is_night"]   = ((df["hour"] < 6) | (df["hour"] >= 22)).astype("int8")
    df["is_rush"]    = df["hour"].isin([7,8,9,17,18,19]).astype("int8")
    df["is_biz"]     = ((df["hour"] >= 9) & (df["hour"] < 17) & (df["is_weekend"]==0)).astype("int8")

    # Store SUMS (combinable across batches)
    agg = df.groupby("agg_hex", sort=False).agg(
        visit_count    =("device_id", "count"),
        unique_devices =("device_id", "nunique"),
        sum_weekend    =("is_weekend", "sum"),
        sum_night      =("is_night",   "sum"),
        sum_rush       =("is_rush",    "sum"),
        sum_biz        =("is_biz",     "sum"),
        sum_hour       =("hour",       "sum"),
        sum_hour_sq    =("hour",       lambda x: (x.astype(float)**2).sum()),
    ).reset_index()

    # Hour distribution for entropy (24 columns: h_0 .. h_23)
    hvc = df.groupby(["agg_hex","hour"], sort=False).size().unstack(fill_value=0)
    hvc.columns = [f"h_{c}" for c in hvc.columns]
    agg = agg.merge(hvc.reset_index(), on="agg_hex", how="left").fillna(0)

    del df; gc.collect()
    return agg


def combine_and_finalize(partials):
    # Fill missing hour columns across all partials
    all_h = sorted(set(c for p in partials for c in p.columns if c.startswith("h_")))
    for p in partials:
        for c in all_h:
            if c not in p.columns:
                p[c] = 0

    # Sum everything per hex
    sum_cols = ["visit_count","unique_devices","sum_weekend","sum_night",
                "sum_rush","sum_biz","sum_hour","sum_hour_sq"] + all_h
    combined = (pd.concat(partials, ignore_index=True)
                  .groupby("agg_hex", sort=False)[sum_cols].sum()
                  .reset_index())

    # Compute final features from sums
    n = combined["visit_count"].replace(0, np.nan)
    combined["weekend_ratio"]   = combined["sum_weekend"] / n
    combined["night_ratio"]     = combined["sum_night"]   / n
    combined["rush_ratio"]      = combined["sum_rush"]    / n
    combined["biz_ratio"]       = combined["sum_biz"]     / n
    combined["avg_hour"]        = combined["sum_hour"]    / n
    combined["visits_per_device"] = combined["visit_count"] / combined["unique_devices"].replace(0, np.nan)

    # Std hour via variance formula: Var = E[X²] - E[X]²
    ex  = combined["sum_hour"]    / n
    ex2 = combined["sum_hour_sq"] / n
    combined["std_hour"] = (ex2 - ex**2).clip(lower=0).pow(0.5)

    # Hour entropy from combined hour counts
    if all_h:
        hmat  = combined[all_h].values.astype(float)
        rsums = hmat.sum(axis=1, keepdims=True)
        probs = np.divide(hmat, rsums, where=rsums > 0, out=np.zeros_like(hmat))
        combined["hour_entropy"] = -(probs * np.log(probs + 1e-9)).sum(axis=1)

    combined = combined.fillna(0)
    drop = ["sum_weekend","sum_night","sum_rush","sum_biz","sum_hour","sum_hour_sq"] + all_h
    combined.drop(columns=[c for c in drop if c in combined.columns], inplace=True)
    return combined


def add_centroids(df):
    c = df[HEX_COL].apply(h3.cell_to_latlng)
    df = df.copy()
    df["centroid_lat"] = c.apply(lambda x: x[0])
    df["centroid_lng"] = c.apply(lambda x: x[1])
    return df

def merge_mob(df, mob_agg, agg_res):
    df = df.copy()
    if agg_res == TRAIN_RES:
        join_col = HEX_COL
    else:
        pc = f"parent_r{agg_res}"
        df[pc] = df[HEX_COL].apply(lambda hx: h3.cell_to_parent(hx, agg_res))
        join_col = pc
    mob_cols = [c for c in mob_agg.columns if c != "agg_hex"]
    df = df.merge(mob_agg.rename(columns={"agg_hex": join_col}), on=join_col, how="left")
    for c in mob_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(0)
    return df

def get_feat(df):
    EX = {HEX_COL, TARGET_COL, "agg_hex"}
    return [c for c in df.columns
            if c not in EX and not c.startswith("parent_r")
            and pd.api.types.is_numeric_dtype(df[c])]

def cv_rmse(df, feat):
    m = lgb.LGBMRegressor(objective="regression", n_estimators=500, learning_rate=0.05,
        max_depth=5, num_leaves=31, colsample_bytree=0.8, subsample=0.8,
        reg_alpha=0.5, reg_lambda=0.5, min_child_samples=5, random_state=42, n_jobs=-1, verbose=-1)
    s = cross_val_score(m, df[feat].fillna(0), df[TARGET_COL], cv=5,
                        scoring="neg_root_mean_squared_error")
    return float(-s.mean())


train = add_centroids(train)
test  = add_centroids(test)

# ── Resolution experiment ────────────────────────────────────
results  = {}
mob_aggs = {}

for agg_res in [6, 7, 8]:
    print(f"\n{'='*50}")
    print(f"Processing agg_res={agg_res} — one batch at a time...")
    partials = []
    for i, bf in enumerate(batch_files):
        partial = process_batch(bf, agg_res)
        partials.append(partial)
        if (i+1) % 5 == 0 or (i+1) == len(batch_files):
            print(f"  Batch {i+1}/{len(batch_files)} done")

    print(f"  Combining {len(partials)} partial aggs...")
    mob_agg = combine_and_finalize(partials)
    del partials; gc.collect()
    print(f"  Final cells: {len(mob_agg):,}")

    mob_aggs[agg_res] = mob_agg
    tr = merge_mob(train, mob_agg, agg_res)
    feat = get_feat(tr)
    rmse = cv_rmse(tr, feat)
    results[agg_res] = rmse
    print(f"  CV RMSE (res {agg_res}): {rmse:.5f}")
    gc.collect()

best_res = min(results, key=results.get)
print(f"\n{'='*50}")
print("Resolution results:")
for r, s in sorted(results.items()):
    print(f"  res {r}: {s:.5f}{' <<< BEST' if r==best_res else ''}")

# ── Final model ──────────────────────────────────────────────
print(f"\nTraining final model (agg_res={best_res})...")
train_f = merge_mob(train, mob_aggs[best_res], best_res)
test_f  = merge_mob(test,  mob_aggs[best_res], best_res)
feat    = get_feat(train_f)

model = lgb.LGBMRegressor(objective="regression", n_estimators=2000, learning_rate=0.01,
    max_depth=5, num_leaves=31, colsample_bytree=0.8, subsample=0.8,
    reg_alpha=0.5, reg_lambda=0.5, min_child_samples=5, random_state=42, n_jobs=-1, verbose=-1)
model.fit(train_f[feat].fillna(0), train_f[TARGET_COL])

imp = pd.Series(model.feature_importances_, index=feat).sort_values(ascending=False)
print("\nTop-10 features:")
print(imp.head(10).to_string())

preds = np.clip(model.predict(test_f[feat].fillna(0)), 0.0, 1.0)
sub = test[[HEX_COL]].copy()
sub[TARGET_COL] = preds
sub.to_csv(OUT_FILE, index=False)

print(f"\n{'='*55}")
print(f"DONE — submission.csv saved to Drive: {OUT_FILE}")
print(f"Rows: {len(sub)} | Range: [{preds.min():.4f},{preds.max():.4f}] | Mean: {preds.mean():.4f}")
print(f"\nCV results: {results}")
print(f"Best res: {best_res} → RMSE {results[best_res]:.5f}")
print(f"Baseline: {train[TARGET_COL].std():.4f}")
