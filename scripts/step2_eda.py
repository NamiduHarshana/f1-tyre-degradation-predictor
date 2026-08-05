"""
Step 2: Exploratory Data Analysis (EDA)
-----------------------------------------
Purpose: Visualize the tyre degradation pattern - draw 3 charts and run a
correlation analysis.

Step 2: Exploratory Data Analysis (EDA)
-----------------------------------------
Purpose: Tire degradation pattern eka visualize karanawa - charts 3ka draw
karanawa, correlation analysis ekak karanawa.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load cleaned data.
#
# 1. Clean karapu data eka load karanawa.
# -----------------------------
df = pd.read_csv("data/multi_race_cleaned.csv")
print(f"Loaded cleaned data: {df.shape}")

# Exclude the outlier (pit stop) laps from visualizations so the trend
# line isn't skewed by abnormal data points.
# Outlier (pit stop) laps eka visualizations walin ain karanawa, trend
# line eka abnormal data points walin skew wenna epa nisa.
df_no_outlier = df[~df["IsOutlierLap"]].copy()
print(f"Rows after excluding outlier lap: {df_no_outlier.shape}")

# -----------------------------
# 2. Chart 1: TyreLife vs LapTime (colored by Compound).
#    This is the CORE chart - shows the tyre degradation pattern.
#
# 2. Chart 1: TyreLife vs LapTime (Compound ekakinma color karala).
#    Meka CORE chart eka - tyre degradation pattern eka pennanawa.
# -----------------------------
plt.figure(figsize=(10, 6))
for compound in df_no_outlier["Compound"].unique():
    subset = df_no_outlier[df_no_outlier["Compound"] == compound]
    plt.scatter(subset["TyreLife"], subset["LapTime"], label=compound, alpha=0.7, s=60)
    # Add a trend line for each compound (simple linear fit: y = m*x + b)
    # Compound ekakama trend line ekak add karanawa (simple linear fit)
    z = subset["TyreLife"].astype(float)
    y = subset["LapTime"].astype(float)
    if len(z) > 1:
        m, b = np.polyfit(z, y, 1)
        plt.plot(z, m * z + b, linestyle="--", linewidth=1.5)

plt.xlabel("Tyre Life (laps on this tyre)")
plt.ylabel("Lap Time (seconds)")
plt.title("Tyre Degradation: Lap Time vs Tyre Life")
plt.legend(title="Compound")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/chart1_tyre_degradation.png", dpi=150)
print("\nSaved: outputs/chart1_tyre_degradation.png")
plt.close()

# -----------------------------
# 3. Chart 2: Compound comparison (box plot).
#
# 3. Chart 2: Compound comparison (box plot ekak).
# -----------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_no_outlier, x="Compound", y="LapTime")
plt.xlabel("Tyre Compound")
plt.ylabel("Lap Time (seconds)")
plt.title("Lap Time Distribution by Tyre Compound")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("outputs/chart2_compound_comparison.png", dpi=150)
print("Saved: outputs/chart2_compound_comparison.png")
plt.close()

# -----------------------------
# 4. Chart 3: Correlation heatmap (numeric features vs LapTime).
#
# 4. Chart 3: Correlation heatmap (numeric features saha LapTime).
# -----------------------------
numeric_cols = ["LapTime", "TyreLife", "Stint", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "LapNumber"]
corr = df_no_outlier[numeric_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - Features vs Lap Time")
plt.tight_layout()
plt.savefig("outputs/chart3_correlation_heatmap.png", dpi=150)
print("Saved: outputs/chart3_correlation_heatmap.png")
plt.close()

# -----------------------------
# 5. Print key numeric insights.
#
# 5. Key numeric insights ekam print karanawa.
# -----------------------------
print("\n--- KEY INSIGHTS ---")
for compound in df_no_outlier["Compound"].unique():
    subset = df_no_outlier[df_no_outlier["Compound"] == compound]
    fastest = subset["LapTime"].min()
    slowest = subset["LapTime"].max()
    avg = subset["LapTime"].mean()
    print(f"{compound}: fastest={fastest:.3f}s, slowest={slowest:.3f}s, avg={avg:.3f}s, laps={len(subset)}")

print("\nCorrelation of each feature with LapTime:")
print(corr["LapTime"].sort_values(ascending=False))