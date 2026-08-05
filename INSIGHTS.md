# F1 Tyre Degradation Analysis - Key Insights

**Author:** Namidu Harshana
**Date:** August 2026
**Project:** F1 Multi-Race Tyre Degradation & Lap Time Prediction

---

## 📊 Dataset Summary

- **Source:** FastF1 library (official F1 timing data)
- **Races:** 8 races from the 2024 season (Bahrain, Saudi Arabia, Australia, Japan, China, Miami, Emilia Romagna, Monaco)
- **Drivers:** 21 drivers
- **Raw laps:** 8,553
- **Cleaned laps used for modelling:** 7,409 (after removing safety car/yellow flag laps and pit stop outliers)

## 🎯 Project Goal

Predict a Formula 1 car's lap time based on tyre age, tyre compound, track, and speed data — with a focus on understanding how tyre degradation affects lap time.

## 🤖 Model Results

| Model                    | RMSE (seconds) | R² Score  |
| ------------------------ | -------------- | --------- |
| Linear Regression        | 1.40           | 0.970     |
| **Random Forest (best)** | **1.19**       | **0.978** |

The Random Forest model explains **97.8%** of the variation in lap time, with an average prediction error of **1.19 seconds**.

## 🔍 Key Findings

### 1. Track identity is the strongest predictor, not tyre wear

`SpeedST` (speed trap on the longest straight) was by far the most important feature (59% of the model's decision-making), followed by `SpeedFL` and race-specific features (e.g. `RaceName_Bahrain`). This makes sense — a car's top speed on a given straight is closely tied to that track's layout and the car's overall pace on that day.

### 2. TyreLife alone has a weak, even negative, raw correlation with lap time

The raw correlation between `TyreLife` and `LapTime` was **-0.38** — the opposite of what you'd expect from tyre wear alone. This is explained by the **fuel effect**: as a race progresses, the car burns fuel and becomes lighter, which makes it faster. This fuel-effect speed gain outweighs the tyre-wear speed loss in the raw data, masking the tyre degradation signal. This is a genuine, well-known effect in F1 — models that want to isolate pure tyre degradation would need fuel-load data, which isn't available in this dataset.

### 3. Green-flag filtering matters

621 laps (7% of raw data) were run under safety car, virtual safety car, or red flag conditions. These laps have artificially slow (or occasionally fast) times unrelated to tyre wear, and were excluded from analysis to avoid contaminating the degradation pattern.

### 4. Compound differences are visible but track-dependent

SOFT tyres had a higher average lap time (95.6s) than HARD (89.6s) or MEDIUM (89.5s) in the raw combined data — the reverse of what's normally expected (softs are usually fastest). This is a sample composition effect: which compound is used depends heavily on the specific race/track, so compound comparisons across different tracks can be misleading without controlling for track.

## ⚠️ Limitations

- **No fuel load data** — this confounds the raw tyre-degradation signal, as explained above.
- **No weather data used** — track temperature significantly affects tyre degradation in real F1 strategy but wasn't included as a model feature.
- **Driver skill not modelled** — the model deliberately excludes `Driver` as a feature, since the goal was tyre/track effects, not driver comparison.
- **One anomalous outlier** — one prediction had a ~30 second residual, likely an incident (e.g. spin, damage) not caught by the automated TrackStatus filter.

## 🚀 Possible Future Improvements

- Add weather data (track temperature, air temperature) as features
- Add fuel-corrected lap time as an alternative target variable, to isolate pure tyre degradation
- Extend to a full season (24 races) for more robust track-level patterns
- Build a "cliff point" detector — the lap number where tyre performance drops sharply rather than gradually
