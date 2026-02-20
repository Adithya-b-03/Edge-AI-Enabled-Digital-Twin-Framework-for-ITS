import traci
import numpy as np
import pandas as pd

# ==============================
# START SUMO (NO GUI - FAST)
# ==============================
sumoCmd = ["sumo", "-c", "config/map.sumo.cfg"]
traci.start(sumoCmd)

tls_list = traci.trafficlight.getIDList()
if len(tls_list) == 0:
    print("No traffic lights found!")
    traci.close()
    exit()

TLS_ID = tls_list[0]
print("Running Fixed-Time Controller")
print("Using TLS:", TLS_ID)

# ==============================
# METRIC STORAGE
# ==============================
waiting_times = []
co2_list = []
fuel_list = []
queue_lengths = []

MAX_STEPS = 600  # 10 minute simulation

# ==============================
# MAIN LOOP
# ==============================
while traci.simulation.getTime() < MAX_STEPS:
    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()

    step_wait = sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
    step_co2 = sum(traci.vehicle.getCO2Emission(v) for v in vehicles)
    step_fuel = sum(traci.vehicle.getFuelConsumption(v) for v in vehicles)

    waiting_times.append(step_wait)
    co2_list.append(step_co2)
    fuel_list.append(step_fuel)

    lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
    queue_lengths.append(queue)

traci.close()

# ==============================
# SAVE RESULTS
# ==============================
results = pd.DataFrame({
    "Metric": [
        "Waiting Time",
        "Queue Length",
        "CO2",
        "Fuel"
    ],
    "Value": [
        np.mean(waiting_times),
        np.mean(queue_lengths),
        np.mean(co2_list),
        np.mean(fuel_list)
    ]
})

results.to_csv("output/fixed_results.csv", index=False)

print("\n===== FIXED TIME RESULTS =====")
print(results)
print("================================")