# 🏎️ F1 Tyre Degradation Predictor

Predicts Formula 1 lap times based on tyre wear, compound, and track conditions using a Random Forest model trained on real FastF1 telemetry data.

**Author:** Namidu Harshana

---

## What It Does

Given a track, tyre compound, tyre age, stint number, and lap number, the model predicts the expected lap time in seconds. The project also surfaces which factors the model relies on most (feature importance), and includes a full write-up of the analysis, methodology, and limitations in [INSIGHTS.md](INSIGHTS.md).

## Results

| Metric        | Value                                   |
| ------------- | --------------------------------------- |
| Model         | Random Forest Regressor                 |
| R² Score      | 0.978                                   |
| RMSE          | 1.19 seconds                            |
| Training data | 7,409 laps across 8 races (2024 season) |

## Tech Stack

- **Data:** [FastF1](https://github.com/theOehrly/Fast-F1) API (official F1 timing data)
- **ML:** pandas, scikit-learn
- **Web app:** Flask
- **Visualization:** matplotlib, seaborn

## Project Structure

f1-tire-project/
├── data/ Raw and cleaned datasets, train/test splits
├── scripts/ Step-by-step pipeline (data collection → model training)
│ ├── step0_download_data.py
│ ├── step1_clean_data.py
│ ├── step2_eda.py
│ ├── step3_feature_engineering.py
│ ├── step4_model_building.py
│ └── step5_evaluation.py
├── outputs/ Trained model, charts, prediction results
├── webapp/ Flask web app (interactive predictor)
│ ├── app.py
│ ├── templates/
│ └── static/
├── INSIGHTS.md Full analysis write-up and findings
└── requirements.txt

## How to Run Locally

1. Clone the repo and set up a virtual environment:

```bash
   git clone <this-repo-url>
   cd f1-tire-project
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
```

2. Run the data pipeline (optional — cleaned data is already included):

```bash
   python3 scripts/step0_download_data.py
   python3 scripts/step1_clean_data.py
   python3 scripts/step3_feature_engineering.py
   python3 scripts/step4_model_building.py
   python3 scripts/step5_evaluation.py
```

3. Run the web app:

```bash
   cd webapp
   python3 app.py
```

Then open `http://127.0.0.1:5001` in your browser.

## Key Finding

Raw tyre age alone had almost no correlation with lap time in this dataset — because fuel burn-off (the car getting lighter as the race progresses) outweighs the tyre-wear effect in the raw numbers. Full explanation in [INSIGHTS.md](INSIGHTS.md).

## Limitations

See [INSIGHTS.md](INSIGHTS.md) for the full discussion of limitations (no fuel-load data, no weather data, driver skill excluded by design).
