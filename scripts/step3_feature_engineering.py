"""
Step 3: Feature Engineering for Multi-Race Dataset
-----------------------------------------------------------------
Purpose: Prepare the data before giving it to the model.
- Convert categorical columns (Compound, RaceName) to numeric
- Driver is not used as a feature - one-hot encoding 21 drivers would add
  too many columns, and we're focused on tyre degradation, not driver skill
- Do the train/test split

Step 3: Feature Engineering for Multi-Race Dataset
-----------------------------------------------------------------
Purpose: Model ekakata denna kalin data eka prepare karanawa.
- Compound + RaceName (categorical columns) numeric ekakata convert karanawa
- Driver eka feature ekak widihata use karanne na (drivers 21ka one-hot
  karoth columns godak vැඩි wei, saha api focus karanne tyre degradation
  ekka - driver skill ekka nowei)
- Train/Test split karanawa
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load cleaned multi-race data.
#
# 1. Clean karapu multi-race data eka load karanawa.
# -----------------------------
df = pd.read_csv("data/multi_race_cleaned.csv")
print(f"Loaded data: {df.shape}")

# -----------------------------
# 2. Remove outlier (pit stop / incident) laps.
#
# 2. Outlier (pit stop / incident) laps eka remove karanawa.
# -----------------------------
df = df[~df["IsOutlierLap"]].copy()
print(f"Shape after removing outlier laps: {df.shape}")

# -----------------------------
# 3. Encode categorical columns: Compound and RaceName.
#    RaceName is included because different tracks have very different
#    baseline lap times and degradation characteristics (e.g. Monaco vs Spa).
#
# 3. Categorical columns eka encode karanawa: Compound saha RaceName.
#    RaceName eka include karanne, track ekakinma baseline lap time saha
#    degradation characteristics godak wenas nisa (Monaco vs Spa vage).
# -----------------------------
df = pd.get_dummies(df, columns=["Compound", "RaceName"], drop_first=True)
print(f"\nShape after encoding: {df.shape}")

# -----------------------------
# 4. Select features (X) and target (y).
#    Sector1/2/3Time are excluded - LapTime = Sector1+Sector2+Sector3
#    exactly, so including them would be data leakage.
#    Driver is excluded - we're modelling tyre degradation, not driver skill.
#
# 4. Features (X) saha target (y) select karanawa.
#    Sector1/2/3Time eka exclude karanawa - LapTime eka Sector1+2+3 ekatama
#    ekasama nisa, include karoth eka data leakage ekak wei.
#    Driver eka exclude karanawa - api model karanne tyre degradation eka,
#    driver skill eka nowei.
# -----------------------------
base_features = ["TyreLife", "Stint", "LapNumber", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]
compound_cols = [c for c in df.columns if c.startswith("Compound_")]
race_cols = [c for c in df.columns if c.startswith("RaceName_")]

feature_cols = base_features + compound_cols + race_cols

X = df[feature_cols]
y = df["LapTime"]

print(f"\nTotal features used: {len(feature_cols)}")
print(f"Feature columns: {feature_cols}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# -----------------------------
# 5. Train/Test split.
#
# 5. Train/Test split karanawa.
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")

# -----------------------------
# 6. Save prepared data.
#
# 6. Prepare karapu data eka save karanawa.
# -----------------------------
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\n✅ Saved: data/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("\nSample of X_train:")
print(X_train.head(3))