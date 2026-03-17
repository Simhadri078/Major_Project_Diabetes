import pandas as pd
from tqdm import tqdm

# Load 500 diabetic patients
patients = pd.read_csv("data/processed/diabetic_500.csv")
patient_ids = set(patients["subject_id"])

# Selected lab itemids (replace with your real ones)
selected_itemids = [50931, 50912, 50852]  # Example

input_file = "data/raw/labevents.csv.gz"
output_file = "data/processed/labs_500_filtered.csv"

chunksize = 100000  # adjust if needed

first_write = True

for chunk in tqdm(pd.read_csv(input_file, chunksize=chunksize)):
    
    # Filter by patient and lab item
    filtered = chunk[
        chunk["subject_id"].isin(patient_ids) &
        chunk["itemid"].isin(selected_itemids)
    ]
    
    if not filtered.empty:
        filtered.to_csv(
            output_file,
            mode="a",
            header=first_write,
            index=False
        )
        first_write = False

print("Finished filtering labs safely.")