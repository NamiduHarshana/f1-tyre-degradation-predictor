"""
Step 4: Model Building
------------------------
Purpose: Train an ML model that predicts LapTime.
- Try Linear Regression (a simple baseline model)
- Try Random Forest Regressor (a more advanced model)
- Compare the two, and see which one is better

Step 4: Model Building
------------------------
Purpose: LapTime predict karana ML model ekak train karanawa.
- Linear Regression eka (simple, baseline model ekak) try karanawa
- Random Forest Regressor eka (more advanced model ekak) try karanawa
- Deka compare karanawa, konekakda hondada balanawa
"""

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load train/test data (saved in Step 3).
#
# 1. Train/test data eka load karanawa (Step 3 eken save karapu eka).
# -----------------------------
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

# -----------------------------
# 2. Model 1: Linear Regression (simple baseline).
#    Idea: y = m1*x1 + m2*x2 + ... + c  (straight-line relationship)
#
# 2. Model 1: Linear Regression (simple baseline eka).
#    Idea eka: y = m1*x1 + m2*x2 + ... + c  (straight-line relationship)
# -----------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

lr_rmse = mean_squared_error(y_test, lr_predictions) ** 0.5
lr_r2 = r2_score(y_test, lr_predictions)

print("\n--- Linear Regression Results ---")
print(f"RMSE: {lr_rmse:.4f} seconds")
print(f"R² Score: {lr_r2:.4f}")

# -----------------------------
# 3. Model 2: Random Forest Regressor (more advanced).
#    Idea: train many decision trees, then average their predictions.
#
# 3. Model 2: Random Forest Regressor (more advanced eka).
#    Idea eka: decision trees godak train karanawa, ewange average eka
#    gannawa.
# -----------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_rmse = mean_squared_error(y_test, rf_predictions) ** 0.5
rf_r2 = r2_score(y_test, rf_predictions)

print("\n--- Random Forest Results ---")
print(f"RMSE: {rf_rmse:.4f} seconds")
print(f"R² Score: {rf_r2:.4f}")

# -----------------------------
# 4. Compare models, pick the better one.
#
# 4. Models deka compare karanawa, hondama eka gannawa.
# -----------------------------
print("\n--- Comparison ---")
print(f"Linear Regression: RMSE={lr_rmse:.4f}, R²={lr_r2:.4f}")
print(f"Random Forest:     RMSE={rf_rmse:.4f}, R²={rf_r2:.4f}")

if rf_rmse < lr_rmse:
    best_model = rf_model
    best_name = "Random Forest"
else:
    best_model = lr_model
    best_name = "Linear Regression"

print(f"\n✅ Best model: {best_name}")

# -----------------------------
# 5. Feature importance (only meaningful for Random Forest).
#
# 5. Feature importance eka (Random Forest ekata witharai meaningful eka).
# -----------------------------
print("\n--- Feature Importance (Random Forest) ---")
importance = pd.Series(rf_model.feature_importances_, index=X_train.columns)
print(importance.sort_values(ascending=False))

# -----------------------------
# 6. Save the best model to disk (so we can reuse it later, e.g. in a
#    web app).
#
# 6. Best model eka disk ekata save karanawa (passe reuse karanna
#    puluwan, udhaharana widihata web app ekaka).
# -----------------------------
joblib.dump(best_model, "outputs/best_model.pkl")
print(f"\n✅ Saved best model ({best_name}) to outputs/best_model.pkl")

# -----------------------------
# 7. Save predictions vs actual for later charting (Step 5).
#
# 7. Predictions vs actual eka save karanawa (Step 5 charting ekata).
# -----------------------------
results_df = pd.DataFrame({
    "Actual_LapTime": y_test,
    "LinearRegression_Predicted": lr_predictions,
    "RandomForest_Predicted": rf_predictions,
})
results_df.to_csv("outputs/predictions_comparison.csv", index=False)
print("Saved: outputs/predictions_comparison.csv")
print("\nSample predictions:")
print(results_df.head())