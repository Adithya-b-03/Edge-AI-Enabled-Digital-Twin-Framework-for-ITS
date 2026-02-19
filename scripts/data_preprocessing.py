import pandas as pd
import numpy as np

df = pd.read_csv("output/traffic_dataset.csv")

print("Initial shape:", df.shape)

# ---------------------
# DATA CLEANING
# ---------------------

# Remove completely empty rows
df.dropna(inplace=True)

# Remove unrealistic negative speeds
df = df[df["speed"] >= 0]

# Remove rows where co2 and fuel both zero AND speed zero
df = df[~((df["speed"] == 0) & (df["co2"] == 0) & (df["fuel"] == 0))]

print("After cleaning:", df.shape)

# Aggregate per time step
agg = df.groupby("time").agg({
    "vehicle_id": "count",
    "speed": ["mean", "std"],
    "acceleration": ["mean", "std"],
    "co2": "mean",
    "fuel": "mean"
}).reset_index()

# Flatten column names
agg.columns = [
    "time",
    "vehicle_count",
    "avg_speed", "speed_std",
    "avg_acceleration", "acc_std",
    "avg_co2",
    "avg_fuel"
]

# Sudden braking count
df["sudden_brake"] = df["acceleration"] < -3

brake_count = df.groupby("time")["sudden_brake"].sum().reset_index()
agg = agg.merge(brake_count, on="time")

agg.rename(columns={"sudden_brake": "sudden_brake_count"}, inplace=True)

# Save processed data
agg.to_csv("output/processed_data.csv", index=False)

print("Processed dataset saved.")
