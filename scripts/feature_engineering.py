import pandas as pd
import numpy as np

# -----------------------------
# 1️⃣ Load processed dataset
# -----------------------------
df = pd.read_csv("output/processed_data.csv")

print("Loaded processed dataset")
print("Shape:", df.shape)
print(df.head())

# -----------------------------
# 2️⃣ Congestion Score (CS)
# -----------------------------
# Idea:
# More vehicles + lower speed = more congestion

df["CS"] = df["vehicle_count"] / (df["avg_speed"] + 1)

# -----------------------------
# 3️⃣ Risk Score (RS)
# -----------------------------
# Idea:
# Higher acceleration variance + sudden braking events
# indicate instability and accident risk

df["RS"] = df["acc_std"] + df["sudden_brake_count"]

# -----------------------------
# 4️⃣ Emission Score (ES)
# -----------------------------
# Idea:
# Higher CO2 + Fuel consumption means
# inefficient traffic condition

df["ES"] = df["avg_co2"] + df["avg_fuel"]

# -----------------------------
# 5️⃣ Normalization
# -----------------------------
# Why normalize?
# Because CS, RS, ES may have different scales.
# We convert them into 0–1 range.

for col in ["CS", "RS", "ES"]:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

print("\nAfter normalization:")
print(df[["CS", "RS", "ES"]].describe())

# -----------------------------
# 6️⃣ Final Traffic Intelligence Score (TIS)
# -----------------------------
# Weighted combination
# You can justify weights in your report

df["TIS"] = (
    0.4 * df["CS"] +   # congestion importance
    0.3 * df["RS"] +   # safety importance
    0.3 * df["ES"]     # environmental importance
)

# -----------------------------
# 7️⃣ Save final intelligent dataset
# -----------------------------
df.to_csv("output/intelligent_dataset.csv", index=False)

print("\nIntelligence dataset ready.")
print("Final shape:", df.shape)
