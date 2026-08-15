import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -----------------------------
# CREATE PROTOTYPE DATASET
# -----------------------------

np.random.seed(42)

N = 5000

df = pd.DataFrame({
    "rainfall_mm": np.random.uniform(0, 300, N),
    "temperature_c": np.random.uniform(15, 45, N),
    "humidity_pct": np.random.uniform(30, 100, N),
    "river_level_m": np.random.uniform(0, 10, N),
    "elevation_m": np.random.uniform(20, 800, N),
    "soil_moisture_pct": np.random.uniform(5, 100, N),
    "wind_speed_kmh": np.random.uniform(0, 180, N),
    "pressure_hpa": np.random.uniform(970, 1030, N),
    "slope_deg": np.random.uniform(0, 60, N),
    "population_density": np.random.uniform(20, 15000, N),
    "vulnerability_index": np.random.uniform(0, 1, N),
    "road_access_index": np.random.uniform(0, 1, N),
    "previous_event": np.random.randint(0, 2, N)
})

# -----------------------------
# FLOOD RISK
# -----------------------------

flood_risk = (
    0.30 * (df["rainfall_mm"] / 3)
    + 0.22 * (df["river_level_m"] * 10)
    + 0.16 * df["soil_moisture_pct"]
    + 0.10 * df["humidity_pct"]
    + 0.10 * (1 - df["elevation_m"] / 800) * 100
    + 0.12 * df["previous_event"] * 100
)

# -----------------------------
# CYCLONE RISK
# -----------------------------

cyclone_risk = (
    0.38 * (df["wind_speed_kmh"] / 1.8)
    + 0.22 * ((1030 - df["pressure_hpa"]) / 0.6)
    + 0.15 * (df["rainfall_mm"] / 3)
    + 0.10 * df["humidity_pct"]
    + 0.15 * df["previous_event"] * 100
)

# -----------------------------
# LANDSLIDE RISK
# -----------------------------

landslide_risk = (
    0.32 * (df["slope_deg"] / 0.6)
    + 0.25 * (df["rainfall_mm"] / 3)
    + 0.20 * df["soil_moisture_pct"]
    + 0.10 * (1 - df["elevation_m"] / 800) * 100
    + 0.13 * df["previous_event"] * 100
)

# Keep values between 0 and 100
flood_risk = np.clip(flood_risk, 0, 100)
cyclone_risk = np.clip(cyclone_risk, 0, 100)
landslide_risk = np.clip(landslide_risk, 0, 100)

features = [
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "river_level_m",
    "elevation_m",
    "soil_moisture_pct",
    "wind_speed_kmh",
    "pressure_hpa",
    "slope_deg",
    "population_density",
    "vulnerability_index",
    "road_access_index",
    "previous_event"
]

targets = {
    "flood": flood_risk,
    "cyclone": cyclone_risk,
    "landslide": landslide_risk
}

# -----------------------------
# TRAIN THREE MODELS
# -----------------------------

for disaster, target in targets.items():

    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        target,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    print(
        f"{disaster.upper()} "
        f"| MAE: {mae:.2f} "
        f"| R2: {r2:.3f}"
    )

    joblib.dump(
        model,
        f"models/{disaster}_model.pkl"
    )

joblib.dump(features, "models/features.pkl")

df["flood_risk"] = flood_risk
df["cyclone_risk"] = cyclone_risk
df["landslide_risk"] = landslide_risk

df.to_csv(
    "data/prototype_training_data.csv",
    index=False
)

print("\n✅ All 3 disaster models trained successfully!")
print("📁 Models saved inside models/")