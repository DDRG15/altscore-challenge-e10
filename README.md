# AltScore Data Science Competition — Cost of Living Prediction

**Author:** Diego Alonso del Río García | github.com/DDRG15  
**Competition:** [AltScore Data Science Competition on Kaggle](https://www.kaggle.com/competitions/alt-score-data-science-competition)  
**Metric:** RMSE | **Stack:** Python · LightGBM · H3 · PyArrow · Pandas

---

## The Problem

Predict `cost_of_living` (a number between 0 and 1) for 511 geographic zones across LATAM,
using 340 million device location records as the main data source.

| File | Rows | What it contains |
|------|------|-----------------|
| `train.csv` | 510 | Zones with known cost of living |
| `test.csv` | 511 | Zones to predict |
| `mobility_data.parquet` | 340,411,133 | Device positions: lat, lon, timestamp |

---

## How I Approached It

I work in real estate. When I saw "predict cost of living per geographic zone," I did not think
about algorithms first — I thought about zoning reports.

In Peru, every property has a municipal zoning certificate that classifies the lot and its
surrounding area: commercial, residential, industrial, mixed use. That classification is based
on exactly the kind of signals this dataset has — who goes there, at what hours, and how often.
The H3 hexagons in this competition are just zones with a different shape.

The H3 grid was new to me, so I read the documentation. What helped me connect it to something
familiar was the coordinate system. In Peru we use UTM to delimit lots — zones 17, 18, and 19
depending on longitude, southern hemisphere, WGS84 datum. If you use the wrong zone, the
coordinate lands in the wrong country. H3 solves that problem completely: one global index,
no zones, no hemisphere flag. For a dataset that spans all of LATAM across multiple UTM zones,
that is a real advantage.

---

## Features

The mobility data gives raw device positions with timestamps. I converted those into zone-level
signals that describe what kind of place each hexagon is.

| Feature | What it measures | In real estate terms |
|---------|-----------------|---------------------|
| `visit_count` | Total records in the zone | How active the area is overall |
| `unique_devices` | Distinct device IDs | How many different people pass through |
| `visits_per_device` | visit_count / unique_devices | Regulars vs. one-time visitors |
| `business_ratio` | Weekday 9am–5pm fraction | Office or commercial zone |
| `weekend_ratio` | Saturday/Sunday fraction | Recreational or residential area |
| `rush_ratio` | 7–9am and 5–7pm fraction | Commuter corridor |
| `night_ratio` | 10pm–6am fraction | Nightlife area or quiet residential — context tells you which |
| `avg_hour` | Average hour of activity | When the zone peaks |
| `std_hour` | How spread the hours are | Mixed use vs. single purpose |
| `hour_entropy` | How evenly spread visits are across all 24 hours | High = active at all hours (terminal, market); Low = one clear peak (office block, dormitory) |
| `centroid_lat` / `centroid_lng` | Center of the hexagon | Encodes geography: country, coast vs. inland, urban vs. rural |

A zone with high `business_ratio` and high `unique_devices` is a commercial district.
A zone with low everything is rural or industrial. A zone with high `night_ratio` but low
total count is residential — people come home late, not out for nightlife.

These are the same criteria a zoning authority puts in a municipal report when classifying
a sector. The code is computing what that report already describes on paper.

---

## Finding the Right H3 Resolution

The competition includes a code example using `resolution=9` with coordinates from San Francisco.
That is a tutorial snippet — it does not tell you what resolution the training data actually uses.

I tested resolutions in both directions, from 4 up to 12, checking which one produced hex IDs
that matched the training set. Then I confirmed with the library directly:

```python
h3.get_resolution("8866d338abfffff")  # → 8
```

Resolution 8 means each hexagon covers about 0.74 km² — roughly one city block. That is the
right scale for neighborhood-level cost of living: fine enough to distinguish a commercial
street from a residential block two streets away, broad enough to have real foot traffic data.

---

## Choosing the Aggregation Scale

Once I knew the training zones were at resolution 8, I had one more decision: at what resolution
do I aggregate the mobility records before joining them?

Think of it like valuing a specific lot. You look at the immediate block, but you also look at
the surrounding neighborhood. A lot next to a busy commercial street is priced differently than
the same lot two blocks into a residential area.

- **Resolution 8** (same as training): maximum local detail, but sparse in low-density areas
- **Resolution 7** (neighborhood scale): more records per cell, smoother signal
- **Resolution 6** (district scale): broadest context, highest data density per cell

I ran 5-fold cross-validation on the training data for each option and picked the one with
the lowest RMSE. Results from the experiment run:

```
Resolution 6  →  CV RMSE: 0.15705  ← best
Resolution 7  →  CV RMSE: 0.16295
Resolution 8  →  CV RMSE: 0.16607
Baseline (predict mean):  0.1918
```

Resolution 6 won — 100 parent cells for 510 training rows. With so few samples, the coarser
aggregation reduces sparsity and generalizes better than matching the training resolution exactly.

---

## Results

| | RMSE |
|---|---|
| Baseline (predict mean) | 0.1918 |
| CV RMSE — res 6 (best) | 0.15705 |
| CV RMSE — res 7 | 0.16295 |
| CV RMSE — res 8 | 0.16607 |
| **Kaggle Public Score** | **0.15575** |
| **Kaggle Private Score** | **0.16118** |

Prediction range: 0.2592 – 0.8342 across 511 zones. Mean: 0.4311.

The public score (0.15575) matches the CV estimate (0.15705) closely — no overfitting.
The private score (0.16118) is slightly higher, as expected on unseen data.
Both beat the baseline by ~18%.

---

## Model

**LightGBM** — gradient boosting on decision trees.

With only 510 training rows, the model choice matters less than the features. What matters
is not overfitting. I used:

- `min_child_samples=5` — no split on fewer than 5 samples
- `reg_alpha=0.5`, `reg_lambda=0.5` — regularization to avoid memorizing training data
- `5-fold cross-validation` — 510 rows is too small for a held-out validation set to mean anything

The baseline to beat is RMSE ≈ 0.192. That is the score you get by predicting the average
cost of living for every zone. It equals the standard deviation of the target in the training
data. Any model above 0.192 is not learning anything.

---

## Setup

```bash
pip install h3 pandas lightgbm pyarrow scikit-learn

# Place these in e10/data/:
#   train.csv  test.csv  mobility_data.parquet

python e10/solution.py
# Output: e10/submission.csv
```

---

## What I Would Add with More Time

- **OpenStreetMap data**: count banks, pharmacies, restaurants per hexagon — direct affluence signal
- **NASA nighttime lights**: satellite-measured light intensity correlates strongly with economic activity
- **Neighbor features**: average cost of living of adjacent hexagons (spatial autocorrelation)
- **Ensemble**: blend LightGBM with a simpler model on geographic features only, to stabilize predictions in zones with no mobility data

---

## Decision Log

| Decision | Why |
|----------|-----|
| Test resolutions 4–12, not just assume 9 | The example used a tutorial value — had to verify against actual data |
| CV experiment for aggregation resolution | Coarser aggregation reduces sparsity; finer preserves locality — only the data can tell you which matters more here |
| 5-fold CV instead of train/val split | 510 rows is not enough for a held-out set to be reliable |
| Fill missing mobility with 0 | No records in a zone means no activity — zero is the correct value, not an imputed estimate |
| LightGBM over other models | Handles missing values natively, fast to iterate, gives feature importance for review |
| RMSE as optimization target | Confirmed: `std(train.cost_of_living) = 0.192` matches the constant-mean baseline cluster on the public leaderboard |
