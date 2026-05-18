# ============================================================
# E10 - AltScore: Cost of Living Prediction
# SINGLE-CELL Colab script — paste entire file into one cell
#
# BEFORE RUNNING:
#   1. Google Drive must have:
#        Mi unidad/AltScore challange/mobility_data.parquet
#        Mi unidad/AltScore challange/train.csv
#        Mi unidad/AltScore challange/test.csv
#   2. Paste this entire file into ONE Colab cell
#   3. Click Run — authorize Drive when prompted (30 sec)
#   4. Walk away — takes 45-90 min
#   5. submission.csv will be saved to your Drive automatically
#
# IF SESSION DIES: just paste + run again — resumes from last batch
# ============================================================

# Step 1: Mount Drive (requires 30 sec authorization click)
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Install h3 (not pre-installed on Colab)
import subprocess
subprocess.run(["pip", "install", "h3", "-q"], check=True)

# Step 3: Imports
import os, gc
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import h3

# Step 4: Paths — matches your Drive folder exactly
DRIVE_DIR  = "/content/drive/MyDrive/AltScore challange"
BATCH_DIR  = os.path.join(DRIVE_DIR, "batches")
PROGRESS_F = os.path.join(DRIVE_DIR, "progress.txt")
OUT_FILE   = os.path.join(DRIVE_DIR, "submission.csv")
HEX_COL    = "hex_id"
TARGET_COL = "cost_of_living"

os.makedirs(BATCH_DIR, exist_ok=True)
print(f"Drive folder contents: {os.listdir(DRIVE_DIR)}")

# ── PHASE 1: Load train/test ────────────────────────────────
print("\n[1/6] Loading train/test...")
train = pd.read_csv(os.path.join(DRIVE_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DRIVE_DIR, "test.csv"))
print(f"  train: {train.shape} | test: {test.shape}")

sample_hex = train[HEX_COL].dropna().iloc[0]
TRAIN_RES  = h3.get_resolution(sample_hex)
print(f"  Training H3 resolution (from data): {TRAIN_RES}")
print(f"  cost_of_living — mean: {train[TARGET_COL].mean():.4f}  std: {train[TARGET_COL].std():.4f}")
print(f"  Baseline RMSE to beat: {train[TARGET_COL].std():.4f}")

# ── PHASE 2: Bounding box ───────────────────────────────────
print("\n[2/6] Computing LATAM bounding box...")
all_hexes = pd.concat([train[HEX_COL], test[HEX_COL]], ignore_index=True)
centroids = [h3.cell_to_latlng(hx) for hx in all_hexes]
LAT_MIN = min(c[0] for c in centroids) - 1.0
LAT_MAX = max(c[0] for c in centroids) + 1.0
LON_MIN = min(c[1] for c in centroids) - 1.0
LON_MAX = max(c[1] for c in centroids) + 1.0
print(f"  lat=[{LAT_MIN:.2f}, {LAT_MAX:.2f}]  lon=[{LON_MIN:.2f}, {LON_MAX:.2f}]")

# ── PHASE 3: Read parquet in batches, save each to Drive ────
print("\n[3/6] Reading mobility_data.parquet in batches...")
print("  Each batch (~10M rows) is saved to Drive immediately.")
print("  If session dies: re-run — already-saved batches are skipped.\n")

parquet_path = os.path.join(DRIVE_DIR, "mobility_data.parquet")
pf    = pq.ParquetFile(parquet_path)
total = pf.metadata.num_rows
print(f"  Total rows in parquet: {total:,}")

# Check resume point
last_done = -1
if os.path.exists(PROGRESS_F):
    with open(PROGRESS_F) as f:
        val = f.read().strip()
        last_done = int(val) if val.isdigit() else -1
    print(f"  Resuming from batch {last_done + 1} (0–{last_done} already saved to Drive)")
else:
    print("  Starting fresh from batch 0")

BATCH_SIZE  = 10_000_000
kept_total  = 0

for i, batch in enumerate(pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["device_id", "lat", "lon", "timestamp"])):

    batch_path = os.path.join(BATCH_DIR, f"batch_{i:03d}.parquet")

    if i <= last_done:
        if os.path.exists(batch_path):
            kept_total += pd.read_parquet(batch_path, columns=["device_id"]).shape[0]
        continue

    chunk = batch.to_pandas()
    mask  = ((chunk["lat"] >= LAT_MIN) & (chunk["lat"] <= LAT_MAX) &
             (chunk["lon"] >= LON_MIN) & (chunk["lon"] <= LON_MAX))
    filt  = chunk[mask].reset_index(drop=True)
    del chunk

    if len(filt) > 0:
        filt.to_parquet(batch_path, index=False)
        kept_total += len(filt)

    with open(PROGRESS_F, "w") as f:
        f.write(str(i))

    pct = min((i + 1) * BATCH_SIZE, total) / total * 100
    print(f"  Batch {i:02d} | {pct:.0f}% done | kept this batch: {len(filt):,} | total kept: {kept_total:,}")
    del filt
    gc.collect()

print(f"\n  Parquet scan complete. Total rows kept: {kept_total:,} ({kept_total/total*100:.1f}%)")

# ── PHASE 4: Concat all batches ─────────────────────────────
print("\n[4/6] Loading all saved batches into memory...")
batch_files = sorted([
    os.path.join(BATCH_DIR, f)
    for f in os.listdir(BATCH_DIR)
    if f.startswith("batch_") and f.endswith(".parquet")
])
print(f"  Found {len(batch_files)} batch files")
mob = pd.concat([pd.read_parquet(f) for f in batch_files], ignore_index=True)
print(f"  Total filtered rows loaded: {len(mob):,}")
gc.collect()

# ── PHASE 5: Resolution experiment ──────────────────────────
print("\n[5/6] Resolution experiment (CV RMSE for agg_res 6, 7, 8)...")

def add_centroids(df):
    c = df[HEX_COL].apply(h3.cell_to_latlng)
    df = df.copy()
    df["centroid_lat"] = c.apply(lambda x: x[0])
    df["centroid_lng"] = c.apply(lambda x: x[1])
    return df

def aggregate_mobility(mob_df, agg_res):
    print(f"  Converting {len(mob_df):,} records to H3 res {agg_res}...")
    m = mob_df.copy()
    m["agg_hex"]   = [h3.latlng_to_cell(float(la), float(lo), agg_res)
                      for la, lo in zip(m["lat"].to_numpy(), m["lon"].to_numpy())]
    m["dt"]        = pd.to_datetime(m["timestamp"], unit="s", errors="coerce")
    m["hour"]      = m["dt"].dt.hour.astype("int8")
    m["dayofweek"] = m["dt"].dt.dayofweek.astype("int8")
    m["is_weekend"]= (m["dayofweek"] >= 5).astype("int8")
    m["is_night"]  = ((m["hour"] < 6) | (m["hour"] >= 22)).astype("int8")
    m["is_rush"]   = m["hour"].isin([7,8,9,17,18,19]).astype("int8")
    m["is_biz"]    = ((m["hour"] >= 9) & (m["hour"] < 17) & (m["is_weekend"]==0)).astype("int8")

    agg = m.groupby("agg_hex", sort=False).agg(
        visit_count    =("device_id","count"),
        unique_devices =("device_id","nunique"),
        weekend_ratio  =("is_weekend","mean"),
        night_ratio    =("is_night","mean"),
        rush_ratio     =("is_rush","mean"),
        biz_ratio      =("is_biz","mean"),
        avg_hour       =("hour","mean"),
        std_hour       =("hour","std"),
    ).reset_index()
    agg["visits_per_device"] = (agg["visit_count"] /
                                agg["unique_devices"].replace(0,np.nan)).fillna(0)
    agg["std_hour"] = agg["std_hour"].fillna(0)

    hvc     = m.groupby(["agg_hex","hour"],sort=False).size().unstack(fill_value=0)
    probs   = hvc.div(hvc.sum(axis=1), axis=0)
    entropy = -(probs * np.log(probs+1e-9)).sum(axis=1)
    agg     = agg.merge(entropy.rename("hour_entropy").reset_index(), on="agg_hex", how="left")
    agg["hour_entropy"] = agg["hour_entropy"].fillna(0)
    print(f"  Aggregated to {len(agg):,} cells")
    return agg

def merge_mobility(df, mob_agg, train_res, agg_res):
    df = df.copy()
    if agg_res == train_res:
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

def get_feature_cols(df):
    EXCLUDE = {HEX_COL, TARGET_COL, "agg_hex"}
    return [c for c in df.columns
            if c not in EXCLUDE and not c.startswith("parent_r")
            and pd.api.types.is_numeric_dtype(df[c])]

def cv_rmse(df, feat_cols):
    m = lgb.LGBMRegressor(
        objective="regression", n_estimators=500, learning_rate=0.05,
        max_depth=5, num_leaves=31, colsample_bytree=0.8, subsample=0.8,
        reg_alpha=0.5, reg_lambda=0.5, min_child_samples=5,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    scores = cross_val_score(m, df[feat_cols].fillna(0), df[TARGET_COL],
                             cv=5, scoring="neg_root_mean_squared_error")
    return float(-scores.mean())

train = add_centroids(train)
test  = add_centroids(test)

results  = {}
mob_aggs = {}
for agg_res in [6, 7, 8]:
    print(f"\n  --- agg_res = {agg_res} ---")
    mob_agg = aggregate_mobility(mob, agg_res)
    mob_aggs[agg_res] = mob_agg
    tr = merge_mobility(train, mob_agg, TRAIN_RES, agg_res)
    rmse = cv_rmse(tr, get_feature_cols(tr))
    results[agg_res] = rmse
    print(f"  CV RMSE (res {agg_res}): {rmse:.5f}")
    gc.collect()

best_res = min(results, key=results.get)
print(f"\n  Resolution results:")
for r, s in sorted(results.items()):
    print(f"    res {r}: {s:.5f}{' <<< BEST' if r==best_res else ''}")

# ── PHASE 6: Final model + submission ───────────────────────
print(f"\n[6/6] Training final model (agg_res={best_res}, 2000 trees)...")

train_f   = merge_mobility(train, mob_aggs[best_res], TRAIN_RES, best_res)
test_f    = merge_mobility(test,  mob_aggs[best_res], TRAIN_RES, best_res)
feat_cols = get_feature_cols(train_f)

model = lgb.LGBMRegressor(
    objective="regression", n_estimators=2000, learning_rate=0.01,
    max_depth=5, num_leaves=31, colsample_bytree=0.8, subsample=0.8,
    reg_alpha=0.5, reg_lambda=0.5, min_child_samples=5,
    random_state=42, n_jobs=-1, verbose=-1,
)
model.fit(train_f[feat_cols].fillna(0), train_f[TARGET_COL])

imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
print("\n  Top-10 feature importances:")
print(imp.head(10).to_string())

preds = np.clip(model.predict(test_f[feat_cols].fillna(0)), 0.0, 1.0)
submission = test[[HEX_COL]].copy()
submission[TARGET_COL] = preds
submission.to_csv(OUT_FILE, index=False)

print(f"\n{'='*55}")
print(f"DONE. submission.csv saved to Drive:")
print(f"  {OUT_FILE}")
print(f"  Rows: {len(submission)} | Range: [{preds.min():.4f}, {preds.max():.4f}]")
print(f"  Mean prediction: {preds.mean():.4f} (train mean: {train[TARGET_COL].mean():.4f})")
print(f"\nCV RMSE results:")
for r, s in sorted(results.items()):
    print(f"  res {r} → {s:.5f}")
print(f"  Baseline (predict mean): {train[TARGET_COL].std():.4f}")
print(f"\nUpload submission.csv at:")
print("  kaggle.com/competitions/alt-score-data-science-competition/submit")
