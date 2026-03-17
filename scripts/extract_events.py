import pandas as pd
import numpy as np
import pickle

print("Loading pivoted dataset...")

# Load filtered labs
df = pd.read_csv("data/processed/labs_500_filtered.csv")

df = df[["subject_id", "charttime", "itemid", "valuenum"]]
df = df[df["valuenum"].notna()]
df["charttime"] = pd.to_datetime(df["charttime"])

# Map lab IDs
lab_map = {
    50931: "glucose",
    50912: "creatinine",
    50852: "hba1c"
}

df["lab_name"] = df["itemid"].map(lab_map)

# Pivot to wide format
df_pivot = df.pivot_table(
    index=["subject_id", "charttime"],
    columns="lab_name",
    values="valuenum"
).reset_index()

df_pivot = df_pivot.sort_values(["subject_id", "charttime"])

# -------------------------------------------------
# ✅ Continuous time in DAYS (float, not integer)
# -------------------------------------------------

df_pivot["time_days"] = df_pivot.groupby("subject_id")["charttime"].transform(
    lambda x: (x - x.min()).dt.total_seconds() / 86400.0
)

events = {}

print("Extracting events...")

for patient_id, group in df_pivot.groupby("subject_id"):
    
    group = group.sort_values("time_days")
    group = group.dropna(subset=["glucose", "creatinine", "hba1c"])
    
    if len(group) < 2:
        continue
    
    patient_events = []
    prev_row = group.iloc[0]
    
    for i in range(1, len(group)):
        
        curr_row = group.iloc[i]
        t = float(curr_row["time_days"])
        
        # Glucose worsening
        if curr_row["glucose"] - prev_row["glucose"] > 20:
            patient_events.append((t, 0))
        
        # HbA1c worsening
        if curr_row["hba1c"] - prev_row["hba1c"] > 0.5:
            patient_events.append((t, 1))
        
        # Creatinine worsening
        if curr_row["creatinine"] - prev_row["creatinine"] > 0.2:
            patient_events.append((t, 2))
        
        prev_row = curr_row
    
    if len(patient_events) > 0:
        events[patient_id] = patient_events

print("Total patients with events:", len(events))

# Save event sequences
with open("data/processed/events.pkl", "wb") as f:
    pickle.dump(events, f)

print("Event extraction complete.")