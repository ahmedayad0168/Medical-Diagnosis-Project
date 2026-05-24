import pandas as pd
import json

def flatten_scraped_data(raw_json_path, output_csv="data/processed/mayo_clinic_all_diseases.csv"):
    with open(raw_json_path, "r") as f:
        records = json.load(f)

    rows = []
    for rec in records:
        rows.append({
            "Disease": rec["disease_name"],
            "Overview": rec["sections"].get("overview", ""),
            "Symptoms": rec["sections"].get("symptoms", ""),
            "Causes": rec["sections"].get("causes", ""),
            "Risk Factors": rec["sections"].get("risk factors", "")
        })
    df = pd.DataFrame(rows)
    df = df[df["Disease"] != "Unknown"]
    df.to_csv(output_csv, index=False)
    return df