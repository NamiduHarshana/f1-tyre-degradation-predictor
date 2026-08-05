"""
Step 1: Data Cleaning for Multi-Race Dataset
---------------------------------------------------------
Purpose: Clean the raw multi-race F1 lap data.
- Convert timedelta columns to seconds
- Keep only "green flag" (normal racing) laps - safety car/VSC/red flag
  laps would have unusually slow/fast lap times
- Fill missing Speed values using each race's own median (different tracks
  have very different speeds)
- Flag pit stop / anomaly laps (within each Driver + Race + Stint group,
  if a lap is much slower than normal)

Step 1: Data Cleaning for Multi-Race Dataset
---------------------------------------------------------
Purpose: Raw multi-race F1 lap data eka clean karanawa.
- Timedelta columns eka seconds ekata convert karanawa
- Only "green flag" (normal racing) laps thiyaganawa - safety car/VSC/red flag
  laps ekata gattoth lap times ekak unusual widihata slow/fast wei
- Speed columns eke missing values eka race ekakinma (RaceName) median eken
  fill karanawa (track ekak ekakata speed ganan wenas nisa)
- Pit stop / anomaly laps flag karanawa (Driver + Race + Stint group ekaka
  athule, normal ekata wada wadi slow unoth)
"""

import pandas as pd

# -----------------------------
# 1. Load raw multi-race data.
#
# 1. Raw multi-race data eka load karanawa.
# -----------------------------
df = pd.read_csv("data/multi_race_raw.csv")
print(f"Raw data shape: {df.shape}")

# -----------------------------
# 2. Convert timedelta columns to seconds.
#
# 2. Timedelta columns eka seconds ekata convert karanawa.
# -----------------------------
timedelta_cols = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
for col in timedelta_cols:
    df[col] = pd.to_timedelta(df[col]).dt.total_seconds()

# -----------------------------
# 3. Keep only "green flag" laps (TrackStatus == '1').
#    Other statuses = yellow flag, safety car, VSC, red flag - lap times
#    during these are not representative of normal tyre-degradation pace.
#
# 3. "Green flag" laps witharak thiyaganawa (TrackStatus == '1').
#    Anith statuses walata (yellow/safety car/VSC/red flag) lap times eka
#    normal tyre-degradation pace eka represent karanne na.
# -----------------------------
before = len(df)
df["TrackStatus"] = df["TrackStatus"].astype(str)
df = df[df["TrackStatus"] == "1"].copy()
print(f"Dropped {before - len(df)} non-green-flag laps (safety car/yellow/red flag)")

# -----------------------------
# 4. Drop columns not useful for our ML model.
#
# 4. Model ekata one nathi columns drop karanawa.
# -----------------------------
columns_to_drop = [
    "Unnamed: 0", "Time", "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime",
    "LapStartTime", "LapStartDate", "PitOutTime", "PitInTime",
    "DeletedReason", "Deleted", "FastF1Generated", "DriverNumber",
]
df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
print(f"Shape after dropping unnecessary columns: {df.shape}")

# -----------------------------
# 5. Drop rows with missing sector/lap times (out laps, incomplete data).
#
# 5. Sector/lap times missing wela thiyena rows (out laps, incomplete data)
#    drop karanawa.
# -----------------------------
before = len(df)
df = df.dropna(subset=["Sector1Time", "Sector2Time", "Sector3Time", "LapTime"])
print(f"Dropped {before - len(df)} row(s) with missing sector/lap times.")

# -----------------------------
# 6. Fill missing Speed values using the median FOR THAT RACE (different
#    tracks have very different speeds, so a global median would mislead).
#
# 6. Speed missing values eka ee race ekakinma median eken fill karanawa
#    (track wenas wenas kotasata speed godak wenas nisa, global median
#    eka gattoth wenn puluwan misleading).
# -----------------------------
speed_cols = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]
for col in speed_cols:
    n_missing = df[col].isnull().sum()
    if n_missing > 0:
        df[col] = df.groupby("RaceName")[col].transform(lambda s: s.fillna(s.median()))
        print(f"Filled {n_missing} missing value(s) in '{col}' with per-race median")

# Drop any rows that still have missing speed (e.g. entire race missing it)
# Speed data eka thawath missing thiyena rows drop karanawa
before = len(df)
df = df.dropna(subset=speed_cols)
if before - len(df) > 0:
    print(f"Dropped {before - len(df)} row(s) with unfixable missing speed data.")

# -----------------------------
# 7. Flag outlier laps (pit stops / incidents) PER (Driver, Race, Stint)
#    group. Lap times vary hugely by track (Monaco ~75s vs Spa ~105s), so
#    a single global threshold wouldn't make sense - we flag within groups.
#
# 7. Outlier laps (pit stops / incidents) eka (Driver, Race, Stint) group
#    ekaka athule flag karanawa. Lap times track ekakinma godak wenas nisa
#    (Monaco ~75s, Spa ~105s), global threshold ekak use karanna one na -
#    group ekaka athule witharai flag karanne.
# -----------------------------
group_cols = ["Driver", "RaceName", "Stint"]
group_stats = df.groupby(group_cols)["LapTime"].transform(lambda s: (s - s.mean()) / s.std())
df["IsOutlierLap"] = group_stats.abs() > 2  # more than 2 std devs from stint's own average
df["IsOutlierLap"] = df["IsOutlierLap"].fillna(False)  # single-lap groups -> not outlier

print(f"\nFlagged {df['IsOutlierLap'].sum()} outlier laps out of {len(df)} total laps")

# -----------------------------
# 8. Save cleaned data.
#
# 8. Clean karapu data eka save karanawa.
# -----------------------------
df.to_csv("data/multi_race_cleaned.csv", index=False)
print(f"\n✅ Cleaned data saved to data/multi_race_cleaned.csv")
print(f"Final shape: {df.shape}")
print(f"\nFinal columns: {list(df.columns)}")
print(f"\nRemaining nulls: {df.isnull().sum().sum()}")