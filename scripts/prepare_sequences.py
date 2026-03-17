import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# -----------------------------
# 1️⃣ Load Filtered Labs
# -----------------------------

print("Loading filtered labs...")

df = pd.read_csv("data/processed/labs_500_filtered.csv")

# Keep only required columns
df = df[["subject_id", "charttime", "itemid", "valuenum"]]

# Remove non-numeric lab rows
df = df[df["valuenum"].notna()]

# Convert charttime to datetime
df["charttime"] = pd.to_datetime(df["charttime"])

print("Initial rows:", len(df))


# -----------------------------
# 2️⃣ Map itemid → Lab Name
# -----------------------------

lab_map = {
    50931: "glucose",
    50912: "creatinine",
    50852: "hba1c"
}

df["lab_name"] = df["itemid"].map(lab_map)


# -----------------------------
# 3️⃣ Pivot to Wide Format
# -----------------------------

print("Pivoting data...")

df_pivot = df.pivot_table(
    index=["subject_id", "charttime"],
    columns="lab_name",
    values="valuenum"
).reset_index()

df_pivot = df_pivot.sort_values(["subject_id", "charttime"])

# Compute time in days since first test
df_pivot["time_days"] = df_pivot.groupby("subject_id")["charttime"] \
    .transform(lambda x: (x - x.min()).dt.days)

print("After pivot:", df_pivot.shape)


# -----------------------------
# 4️⃣ Clean Missing Values
# -----------------------------

# Drop rows where both glucose & creatinine are missing
df_pivot = df_pivot.dropna(subset=["glucose", "creatinine"], how="all")

# Forward fill per patient
df_pivot[["glucose", "creatinine", "hba1c"]] = \
    df_pivot.groupby("subject_id")[["glucose", "creatinine", "hba1c"]].ffill()

# Drop remaining rows with missing core labs
df_pivot = df_pivot.dropna(subset=["glucose", "creatinine"])

print("After cleaning:", df_pivot.shape)


# -----------------------------
# 5️⃣ Keep Patients with ≥3 Visits
# -----------------------------

visit_counts = df_pivot.groupby("subject_id").size()

valid_patients = visit_counts[visit_counts >= 3].index

df_pivot = df_pivot[df_pivot["subject_id"].isin(valid_patients)]

print("Patients after filtering:", df_pivot["subject_id"].nunique())


# -----------------------------
# 6️⃣ Normalize Lab Values
# -----------------------------

scaler = StandardScaler()

df_pivot[["glucose", "creatinine", "hba1c"]] = \
    scaler.fit_transform(
        df_pivot[["glucose", "creatinine", "hba1c"]].fillna(0)
    )

print("Normalization complete.")

# -----------------------------
# Normalize delta time globally
# -----------------------------

all_delta_t = []

for patient_id, group in df_pivot.groupby("subject_id"):
    group = group.sort_values("time_days")
    times = group["time_days"].values
    
    delta = np.diff(times, prepend=times[0])
    all_delta_t.extend(delta)

all_delta_t = np.array(all_delta_t).reshape(-1, 1)

delta_scaler = StandardScaler()
delta_scaler.fit(all_delta_t)

print("Delta time normalization ready.")

# -----------------------------
# 7️⃣ Build Sequences with delta_t
# -----------------------------

print("Building sequences with delta_t...")

sequence_length = 5

X = []
y = []

for patient_id, group in df_pivot.groupby("subject_id"):
    
    group = group.sort_values("time_days")
    
    values = group[["glucose", "creatinine", "hba1c"]].values
    times = group["time_days"].values
    
    for i in range(len(values) - sequence_length):
        
        seq_values = values[i:i+sequence_length]
        seq_times = times[i:i+sequence_length]
        
        # compute delta_t for each time step
        delta_t = np.diff(seq_times, prepend=seq_times[0])
        delta_t = delta_t.reshape(-1, 1)
        delta_t = delta_scaler.transform(delta_t)
        
        # concatenate labs + delta_t
        seq_input = np.hstack((seq_values, delta_t))
        
        X.append(seq_input)
        y.append(values[i+sequence_length])

X = np.array(X)
y = np.array(y)

print("New X shape:", X.shape)
print("y shape:", y.shape)

np.save("data/processed/X_sequences.npy", X)
np.save("data/processed/y_targets.npy", y)

print("Saved updated sequences.")

# -----------------------------
# 8️⃣ Save Sequences
# -----------------------------

os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/X_sequences.npy", X)
np.save("data/processed/y_targets.npy", y)

print("Saved sequences successfully.")
print("Preprocessing complete.")