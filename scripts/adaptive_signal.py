import traci
import numpy as np
import pandas as pd
import joblib

# ==========================================
# LOAD ENSEMBLE MODELS
# ==========================================
rf = joblib.load("models/rf.pkl")
gb = joblib.load("models/gb.pkl")

# ==========================================
# START SUMO
# ==========================================
sumoCmd = ["sumo-gui", "-c", "config/map.sumo.cfg"]
traci.start(sumoCmd)

tls_list = traci.trafficlight.getIDList()
if len(tls_list) == 0:
    print("No traffic lights found!")
    traci.close()
    exit()

TLS_ID = tls_list[0]

print("Advanced Intelligent Controller Started")
print("Using TLS:", TLS_ID)

# ==========================================
# METRIC STORAGE
# ==========================================
waiting_times = []
co2_list = []
fuel_list = []
queue_lengths = []

# ==========================================
# ENSEMBLE PREDICTION FUNCTION
# ==========================================
def predict_tis(features):
    X_live = pd.DataFrame([features], columns=[
        "vehicle_count",
        "avg_speed",
        "speed_std",
        "avg_acceleration",
        "acc_std",
        "avg_co2",
        "avg_fuel",
        "sudden_brake_count"
    ])

    tis_rf = rf.predict(X_live)[0]
    tis_gb = gb.predict(X_live)[0]

    return (tis_rf + tis_gb) / 2


# ==========================================
# MAIN CONTROL LOOP
# ==========================================
while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()

    # ======================================
    # 🚑 PRIORITY CHECK
    # ======================================
    priority_detected = False

    for vid in traci.vehicle.getIDList():
        vtype = traci.vehicle.getTypeID(vid)
        if "emergency" in vtype.lower():
            priority_detected = True
            break

    if priority_detected:
        print("🚑 Emergency detected! Extending green.")
        traci.trafficlight.setPhaseDuration(TLS_ID, 40)

    else:
        # ======================================
        # 🚦 LANE-LEVEL TIS COMPUTATION
        # ======================================
        controlled_lanes = list(set(
            traci.trafficlight.getControlledLanes(TLS_ID)
        ))

        lane_tis = {}

        for lane in controlled_lanes:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane)

            if len(veh_ids) == 0:
                lane_tis[lane] = 0
                continue

            speeds = [traci.vehicle.getSpeed(v) for v in veh_ids]
            accs = [traci.vehicle.getAcceleration(v) for v in veh_ids]
            co2 = [traci.vehicle.getCO2Emission(v) for v in veh_ids]
            fuel = [traci.vehicle.getFuelConsumption(v) for v in veh_ids]

            features = [
                len(veh_ids),
                np.mean(speeds),
                np.std(speeds),
                np.mean(accs),
                np.std(accs),
                np.mean(co2),
                np.mean(fuel),
                sum(a < -3 for a in accs)
            ]

            lane_tis[lane] = predict_tis(features)

        # ======================================
        # 📊 PHASE UTILITY COMPUTATION
        # ======================================
        logics = traci.trafficlight.getAllProgramLogics(TLS_ID)
        phases = logics[0].phases

        phase_utilities = []

        controlled_lanes = traci.trafficlight.getControlledLanes(TLS_ID)

        for phase_index, phase in enumerate(phases):
            state = phase.state
            utility = 0

            for lane_index, signal in enumerate(state):
                if signal.lower() == 'g':
                    lane = controlled_lanes[lane_index]
                    utility += lane_tis.get(lane, 0)

            phase_utilities.append(utility)

        best_phase = int(np.argmax(phase_utilities))
        best_utility = phase_utilities[best_phase]

        traci.trafficlight.setPhase(TLS_ID, best_phase)

        green_time = 15 + int(30 * best_utility)
        traci.trafficlight.setPhaseDuration(TLS_ID, green_time)

        print("Selected Phase:", best_phase,
              "| Utility:", round(best_utility, 3),
              "| Green:", green_time)

    # ======================================
    # 📊 METRIC COLLECTION (EVERY STEP)
    # ======================================
    vehicles = traci.vehicle.getIDList()

    step_wait = 0
    step_co2 = 0
    step_fuel = 0

    for v in vehicles:
        step_wait += traci.vehicle.getWaitingTime(v)
        step_co2 += traci.vehicle.getCO2Emission(v)
        step_fuel += traci.vehicle.getFuelConsumption(v)

    waiting_times.append(step_wait)
    co2_list.append(step_co2)
    fuel_list.append(step_fuel)

    lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
    queue_lengths.append(queue)


# ==========================================
# END SIMULATION
# ==========================================
traci.close()

print("\n===== INTELLIGENT CONTROLLER RESULTS =====")
print("Average Waiting Time:", np.mean(waiting_times))
print("Average Queue Length:", np.mean(queue_lengths))
print("Average CO2:", np.mean(co2_list))
print("Average Fuel:", np.mean(fuel_list))

print("Simulation ended.")