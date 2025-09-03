import json
from patient_info_builder import PatientInfoBuilder

builder = PatientInfoBuilder()

# build just 3 patients to test
results = builder.build_all_patients(limit=3)

with open("test_patients.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

