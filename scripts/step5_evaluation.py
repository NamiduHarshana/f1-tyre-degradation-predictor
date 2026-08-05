"""
Step 5: Model Evaluation
---------------------------
Purpose: Visualize the model's predictions - show how well the model
performs, using charts.

Step 5: Model Evaluation
---------------------------
Purpose: Model eke predictions eka visualize karanawa - kohomada model eka
performance karanne kiyala charts eken pennanawa.
"""

import pandas as pd
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# 1. Load predictions saved in Step 4.
#
# 1. Step 4 eken save karapu predictions eka load karanawa.
# -----------------------------
results = pd.read_csv("outputs/predictions_comparison.csv")
print(f"Loaded predictions: {results.shape}")
print(results.head())

# -----------------------------
# 2. Chart 1: Predicted vs Actual (Random Forest - our best model).
#    Perfect predictions would fall exactly on the diagonal line.
#
# 2. Chart 1: Predicted vs Actual (Random Forest - best model eka).
#    Perfect predictions unoth, dots ekama diagonal line ekaka pennei.
# -----------------------------
plt.figure(figsize=(8, 8))
plt.scatter(results["Actual_LapTime"], results["RandomForest_Predicted"],
            alpha=0.7, s=80, color="steelblue", label="Predictions")

# Draw the "perfect prediction" diagonal line for reference
# "Perfect prediction" diagonal line eka reference ekakata gahanawa
min_val = min(results["Actual_LapTime"].min(), results["RandomForest_Predicted"].min())
max_val = max(results["Actual_LapTime"].max(), results["RandomForest_Predicted"].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")

plt.xlabel("Actual Lap Time (seconds)")
plt.ylabel("Predicted Lap Time (seconds)")
plt.title("Predicted vs Actual Lap Time (Random Forest)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/chart4_predicted_vs_actual.png", dpi=150)
print("\nSaved: outputs/chart4_predicted_vs_actual.png")
plt.close()

# -----------------------------
# 3. Chart 2: Residual Plot (errors for each prediction).
#    Shows whether the model has a systematic bias (should scatter around 0).
#
# 3. Chart 2: Residual Plot (prediction ekaka error eka).
#    Model eke systematic bias ekak thiyenawada kiyala pennanawa (0
#    ekata langa scatter wenna one).
# -----------------------------
residuals = results["Actual_LapTime"] - results["RandomForest_Predicted"]

plt.figure(figsize=(8, 6))
plt.scatter(results["RandomForest_Predicted"], residuals, alpha=0.7, s=80, color="darkorange")
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Lap Time (seconds)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot - Prediction Errors")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/chart5_residuals.png", dpi=150)
print("Saved: outputs/chart5_residuals.png")
plt.close()

# -----------------------------
# 4. Chart 3: Feature Importance (from the saved best model).
#
# 4. Chart 3: Feature Importance (save karapu best model eken).
# -----------------------------
X_train = pd.read_csv("data/X_train.csv")
model = joblib.load("outputs/best_model.pkl")

# Feature importance only exists for tree-based models (Random Forest),
# not for Linear Regression. We check which type of model was saved.
# Feature importance eka thiyenne tree-based models walata witharai
# (Random Forest), Linear Regression ekata na. Save karapu model eke
# type eka check karanawa.
if hasattr(model, "feature_importances_"):
    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
    plt.figure(figsize=(8, 6))
    importance.plot(kind="barh", color="seagreen")
    plt.xlabel("Importance")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("outputs/chart6_feature_importance.png", dpi=150)
    print("Saved: outputs/chart6_feature_importance.png")
    plt.close()
else:
    # For Linear Regression, use coefficient magnitude instead
    # Linear Regression ekata coefficient magnitude eka use karanawa
    coeffs = pd.Series(model.coef_, index=X_train.columns).sort_values()
    plt.figure(figsize=(8, 6))
    coeffs.plot(kind="barh", color="seagreen")
    plt.xlabel("Coefficient value")
    plt.title("Feature Coefficients (Linear Regression)")
    plt.tight_layout()
    plt.savefig("outputs/chart6_feature_importance.png", dpi=150)
    print("Saved: outputs/chart6_feature_importance.png (coefficients, since best model is Linear Regression)")
    plt.close()

# -----------------------------
# 5. Print summary metrics again for reference.
#
# 5. Summary metrics eka ayemath print karanawa (reference ekakata).
# -----------------------------
from sklearn.metrics import mean_squared_error, r2_score
rmse = mean_squared_error(results["Actual_LapTime"], results["RandomForest_Predicted"]) ** 0.5
r2 = r2_score(results["Actual_LapTime"], results["RandomForest_Predicted"])
print(f"\n--- Final Model Summary ---")
print(f"RMSE: {rmse:.4f} seconds")
print(f"R² Score: {r2:.4f}")
print(f"Mean residual: {residuals.mean():.4f} (should be close to 0 if no systematic bias)")