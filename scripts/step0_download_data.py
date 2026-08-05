"""
Step 0: Download Multi-Race F1 Data using FastF1
---------------------------------------------------
Purpose: Download lap data for several 2024 season races using the FastF1
library (not just one driver - all drivers), so the dataset becomes much
larger and richer.

Step 0: Download Multi-Race F1 Data using FastF1
---------------------------------------------------
Purpose: FastF1 library eken 2024 season eke races kihipayක download karanawa
(ekma driver ekak witharak nowei - drivers okkoma), dataset eka loku karanna.
"""

import fastf1
import pandas as pd
import os

# -----------------------------
# 1. Setup cache (FastF1 stores downloaded data locally so it doesn't have
#    to re-download the same session again).
#
# 1. Cache eka setup karanawa (FastF1 eka download karapu data eka
#    locally save karanawa, ayemath download karanna one na)
# -----------------------------
os.makedirs("fastf1_cache", exist_ok=True)
fastf1.Cache.enable_cache("fastf1_cache")

# -----------------------------
# 2. Races we want to download (2024 season, a mix of tracks).
#    Format: (year, race_name)
#
# 2. Download karanna one races (2024 season, track wenas wenas ekak)
#    Format eka: (year, race_name)
# -----------------------------
races_to_download = [
    (2024, "Bahrain"),
    (2024, "Saudi Arabia"),
    (2024, "Australia"),
    (2024, "Japan"),
    (2024, "China"),
    (2024, "Miami"),
    (2024, "Emilia Romagna"),
    (2024, "Monaco"),
]

all_laps = []

# -----------------------------
# 3. Download each race's lap data.
#
# 3. Race ekak ekakama lap data eka download karanawa.
# -----------------------------
for year, race_name in races_to_download:
    print(f"\nDownloading {year} {race_name} GP...")
    try:
        session = fastf1.get_session(year, race_name, "R")  # R = Race
        session.load()  # downloads the data (first time can take ~30-60 sec)

        laps = session.laps.copy()
        laps["RaceName"] = race_name
        laps["Year"] = year

        all_laps.append(laps)
        print(f"  -> Got {len(laps)} laps from {race_name}")

    except Exception as e:
        print(f"  -> FAILED to download {race_name}: {e}")
        print(f"  -> Skipping this race, continuing with the rest...")
        continue

# -----------------------------
# 4. Combine all races into one dataframe.
#
# 4. Races okkoma ekata combine karanawa (ekma dataframe ekakata).
# -----------------------------
if len(all_laps) == 0:
    print("\n❌ No races downloaded successfully. Check your internet connection.")
else:
    combined = pd.concat(all_laps, ignore_index=True)
    print(f"\n✅ Combined dataset: {combined.shape[0]} laps total, from {len(all_laps)} races")

    # -----------------------------
    # 5. Save the raw combined data.
    #
    # 5. Combine karapu raw data eka save karanawa.
    # -----------------------------
    combined.to_csv("data/multi_race_raw.csv", index=False)
    print("✅ Saved to data/multi_race_raw.csv")
    print(f"\nColumns: {list(combined.columns)}")
    print(f"\nDrivers included: {combined['Driver'].unique()}")
    print(f"Races included: {combined['RaceName'].unique()}")