import traci
import pandas as pd
import os

sumoCmd = ["sumo","-c", "../config/map.sumo.cfg"]
traci.start(sumoCmd)

data = []

for step in range(3600):
    traci.simulationStep()
    
    vehicle_ids = traci.vehicle.getIDList()
    
    for vid in vehicle_ids:
        speed = traci.vehicle.getSpeed(vid)
        accel = traci.vehicle.getAcceleration(vid)
        wait = traci.vehicle.getWaitingTime(vid)
        co2 = traci.vehicle.getCO2Emission(vid)
        fuel = traci.vehicle.getFuelConsumption(vid)
        
        data.append([step, vid, speed, accel, wait, co2, fuel])

traci.close()

df = pd.DataFrame(data, columns=[
    "time", "vehicle_id", "speed",
    "acceleration", "waiting_time",
    "co2", "fuel"
])

df.to_csv("../output/traffic_dataset.csv", index=False)
