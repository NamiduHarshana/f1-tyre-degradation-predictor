"""
Step 7: Web Interface (Flask App)
------------------------------------
Purpose: Serve the trained model through a web dashboard. Since a normal
user wouldn't know a car's exact speed-trap readings, we auto-fill those
using each track's average speed (computed from the training data) - the
user only picks Race, Compound, Tyre Age, Stint, and Lap Number.

Step 7: Web Interface (Flask App)
------------------------------------
Purpose: Train karapu model eka web dashboard ekakin serve karanawa.
Normal user ekakuta car eke exact speed-trap readings danne na nisa, ee
values eka track ekakinma average speed eken (training data eken calculate
karapu eka) auto-fill karanawa - user eka select karanne Race, Compound,
Tyre Age, Stint, Lap Number witharai.
"""

from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "best_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "multi_race_cleaned.csv")

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
df = df[~df["IsOutlierLap"]]
race_speed_defaults = df.groupby("RaceName")[["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]].mean()

RACES = sorted(df["RaceName"].unique().tolist())
COMPOUNDS = ["HARD", "MEDIUM", "SOFT"]

FEATURE_COLUMNS = [
    "TyreLife", "Stint", "LapNumber", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "Compound_MEDIUM", "Compound_SOFT",
    "RaceName_Bahrain", "RaceName_China", "RaceName_Emilia Romagna",
    "RaceName_Japan", "RaceName_Miami", "RaceName_Monaco", "RaceName_Saudi Arabia",
]

# -----------------------------
# Group feature importance: individual RaceName_* and Compound_* dummy
# columns get summed into single "RaceName" and "Compound" entries, since
# showing 7 separate race percentages isn't meaningful to a viewer.
#
# Feature importance eka group karanawa: RaceName_* saha Compound_* dummy
# columns eka thani thaniyama pennanna wada, ekathu karala "RaceName" saha
# "Compound" widihata pennanawa - viewer ekakuta 7 races percentages thani
# thaniyama pennanna meaningful na nisa.
# -----------------------------
if hasattr(model, "feature_importances_"):
    raw_importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)

    grouped = {}
    for name, val in raw_importance.items():
        if name.startswith("RaceName_"):
            grouped["RaceName"] = grouped.get("RaceName", 0) + val
        elif name.startswith("Compound_"):
            grouped["Compound"] = grouped.get("Compound", 0) + val
        else:
            grouped[name] = val

    grouped_series = pd.Series(grouped).sort_values(ascending=False)
    max_val = grouped_series.max()
    FEATURE_IMPORTANCE = [
        {"name": name, "pct_of_max": round((val / max_val) * 100, 1), "pct": round(val * 100, 1)}
        for name, val in grouped_series.items()
    ]
else:
    FEATURE_IMPORTANCE = []

TOTAL_LAPS = len(df)
N_RACES = len(RACES)
N_FEATURES = len(FEATURE_IMPORTANCE)
MODEL_NAME = type(model).__name__


def build_feature_row(race, compound, tyre_life, stint, lap_number):
    speeds = race_speed_defaults.loc[race]
    row = {col: 0 for col in FEATURE_COLUMNS}
    row["TyreLife"] = tyre_life
    row["Stint"] = stint
    row["LapNumber"] = lap_number
    row["SpeedI1"] = speeds["SpeedI1"]
    row["SpeedI2"] = speeds["SpeedI2"]
    row["SpeedFL"] = speeds["SpeedFL"]
    row["SpeedST"] = speeds["SpeedST"]
    if compound == "MEDIUM":
        row["Compound_MEDIUM"] = 1
    elif compound == "SOFT":
        row["Compound_SOFT"] = 1
    race_col = f"RaceName_{race}"
    if race_col in row:
        row[race_col] = 1
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def common_context():
    return dict(
        races=RACES, compounds=COMPOUNDS, feature_importance=FEATURE_IMPORTANCE,
        total_laps=TOTAL_LAPS, n_races=N_RACES, n_features=N_FEATURES, model_name=MODEL_NAME,
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None, **common_context())


@app.route("/predict", methods=["POST"])
def predict():
    race = request.form["race"]
    compound = request.form["compound"]
    tyre_life = float(request.form["tyre_life"])
    stint = float(request.form["stint"])
    lap_number = float(request.form["lap_number"])

    X = build_feature_row(race, compound, tyre_life, stint, lap_number)
    predicted_time = model.predict(X)[0]

    return render_template(
        "index.html",
        prediction=round(predicted_time, 3),
        selected_race=race,
        selected_compound=compound,
        selected_tyre_life=tyre_life,
        selected_stint=stint,
        selected_lap_number=lap_number,
        **common_context(),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)