import pandas as pd
from fuzzywuzzy import process

class DiseaseLookup:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.diseases = self.data['Disease'].dropna().unique().tolist()

    def _fuzzy_match(self, query, threshold=75):
        match, score = process.extractOne(query, self.diseases)
        if score >= threshold:
            return match
        return None

    def get_info(self, disease_query, info_type="smart_lookup"):
        disease = self._fuzzy_match(disease_query)
        if not disease:
            return "Disease not found."

        row = self.data[self.data['Disease'] == disease].iloc[0]
        if info_type == "get_causes":
            return f"### Causes of {disease}\n\n{row['Causes']}"
        elif info_type == "get_overview":
            return f"### Overview of {disease}\n\n{row['Overview']}"
        elif info_type == "get_risk_factors":
            return f"### Risk Factors for {disease}\n\n{row['Risk Factors']}"
        else:   # smart_lookup – return all non-empty sections
            result = f"### {disease}\n"
            for col in ['Overview', 'Symptoms', 'Causes', 'Risk Factors']:
                val = row.get(col, "")
                if pd.notna(val) and val.strip():
                    result += f"\n**{col}**:\n{val}\n"
            return result