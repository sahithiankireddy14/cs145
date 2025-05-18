import pandas as pd
import os
pd.set_option("display.max_columns", None) 

# path to hosp folder
data_base = "/Users/psehgal/Documents/physionet.org/files/mimiciv/3.1/hosp"

# read relavant csv's
df = pd.read_csv(os.path.join(data_base, "patients.csv.gz"))
diagnosis_codes = pd.read_csv(os.path.join(data_base, "diagnoses_icd.csv.gz"))
diagnosis_names = pd.read_csv(os.path.join(data_base, "d_icd_diagnoses.csv.gz"))
admissions = pd.read_csv(os.path.join(data_base, "admissions.csv.gz"))
# print(df[0:6])

count = 0

for idx, patient in df.iterrows():
    patient_admissions_dict = {}
    patient_info_dict = {}
    count += 1
    if pd.isna(patient["subject_id"]):
        continue

    patient_id = patient["subject_id"]
    gender = patient["gender"] if not pd.isna(patient["gender"]) else "Unknown"
    anchor_year = int(patient["anchor_year"]) if not pd.isna(patient["anchor_year"]) else "Unknown"
    age = int(patient["anchor_age"]) if not pd.isna(patient["anchor_age"]) else "Unknown"
    if pd.isna(patient['dod']) or anchor_year == "Unknown" or age == "Unknown":
        dod_age = None
    else:
        dod_age = int(patient["dod"].split("-")[0]) - anchor_year + age
    # if count == 6:
    patient_info_dict["gender"] = gender
    # TODO: Remove age should be in admissions table
    patient_info_dict["patient age"] = age
    patient_info_dict["age of death"] = dod_age
    # print(patient_info_dict)
    # print(admissions.head())
    # TODO: improve efficience of this 
    patient_admissions = admissions[admissions['subject_id'] == patient_id]
    if not patient_admissions.empty:
        patient_info_dict["race"] = patient_admissions.iloc[0]["race"]
        for idx, admission in patient_admissions.iterrows():
            patient_admissions_dict[admission["hadm_id"]] = {}
            # TODO: add relavent information to patient_admissions_dict {hadm_id1: {}, hadm_id2: {}}

    else:
        patient_info_dict["race"] = "Unknown"
    # print(patient_admissions)
    print(patient_info_dict)
    print(patient_admissions_dict)


    break



