import pandas as pd

input_file = "data/raw/diagnoses_icd.csv.gz"
output_file = "data/processed/diabetic_500.csv"

chunksize = 200000
diabetic_ids = set()

for chunk in pd.read_csv(input_file, chunksize=chunksize):
    
    # Keep only ICD-9 codes
    chunk = chunk[chunk["icd_version"] == 9]
    
    # Convert icd_code to string
    chunk["icd_code"] = chunk["icd_code"].astype(str)
    
    # Diabetes ICD-9 codes start with 250
    diabetes = chunk[chunk["icd_code"].str.startswith("250")]
    
    diabetic_ids.update(diabetes["subject_id"].unique())
    
    if len(diabetic_ids) >= 500:
        break

if len(diabetic_ids) == 0:
    print("No diabetic patients found.")
else:
    diabetic_ids = list(diabetic_ids)[:500]
    pd.DataFrame({"subject_id": diabetic_ids}).to_csv(output_file, index=False)
    print("Saved", len(diabetic_ids), "diabetic patients.")