import pandas as pd
import os
pd.set_option("display.max_columns", None) 

# path to hosp folder
data_base = "/Users/psehgal/Documents/physionet.org/files/mimiciv/3.1/hosp"

# read relavant csv's
df = pd.read_csv(os.path.join(data_base, "patients.csv.gz"))
diagnosis_codes = pd.read_csv(os.path.join(data_base, "diagnoses_icd.csv.gz"))
diagnosis_names = pd.read_csv(os.path.join(data_base, "d_icd_diagnoses.csv.gz"))
d_hcpcs = pd.read_csv(os.path.join(data_base, "d_hcpcs.csv.gz"))
hcpcsevents = pd.read_csv(os.path.join(data_base, "hcpcsevents.csv.gz"))
admissions = pd.read_csv(os.path.join(data_base, "admissions.csv.gz"))
# print(df[0:6])
LARGE_NUMBER = 1000000000000

def get_diagnoses(hadm_id):
    admission_codes = diagnosis_codes[diagnosis_codes["hadm_id"] == hadm_id]
    if not admission_codes.empty:
        admission_codes = admission_codes["icd_code"].dropna().tolist()
        admission_diagnoses = []
        for admission_code in admission_codes:
            diagnosis_name = diagnosis_names[diagnosis_names["icd_code"] == admission_code]
            if not diagnosis_name.empty:
                admission_diagnoses.append(diagnosis_name["long_title"].iloc[0])
    return admission_diagnoses

def get_hcpcsevents(hadm_id):
    admission_codes = hcpcsevents[hcpcsevents["hadm_id"] == hadm_id]
    if not admission_codes.empty:
        admission_procedures = []
        for idx, row in admission_codes.iterrows():
            admission_procedure = {}
            if not pd.isna(row["short_description"]):
                admission_procedure["actual short description"] = row["short_description"]
            if not pd.isna(row["hcpcs_cd"]):
                procedure_code = row["hcpcs_cd"]
                procedure_desc = d_hcpcs[d_hcpcs["code"] == procedure_code]
                if not procedure_desc.empty:
                    long_desc = procedure_desc["long_description"].iloc[0]
                    official_short_desc = procedure_desc["short_description"].iloc[0]
                    if not pd.isna(official_short_desc):
                        admission_procedure["official short description of encoded procedure"] = official_short_desc
                    if not pd.isna(long_desc):
                        admission_procedure["official long description of encoded procedure"] = long_desc
            if not pd.isna(row["seq_num"]):
                admission_procedure["sequence number"] = int(row["seq_num"])
            elif pd.isna(row["seq_num"]) and admission_procedure:
                admission_procedure["sequence number"] = LARGE_NUMBER
            if admission_procedure:
                admission_procedures.append(admission_procedure)
            # TODO: WHAT TO DO WITH CHART DATE
            # TODO: assuming every hadm_id is unique across all patients
        sorted_data = sorted(admission_procedures, key=lambda x: x["sequence number"])
        return sorted_data

count = 0

for idx, patient in df.iterrows():
    patient_admissions_dict = {}
    patient_info_dict = {}
    count += 1
    if pd.isna(patient["subject_id"]):
        continue

    patient_id = patient["subject_id"]
    # if patient_id == 10000108:
    gender = patient["gender"] if not pd.isna(patient["gender"]) else "Unknown"
    anchor_year = int(patient["anchor_year"]) if not pd.isna(patient["anchor_year"]) else "Unknown"
    age = int(patient["anchor_age"]) if not pd.isna(patient["anchor_age"]) else "Unknown"
    if pd.isna(patient['dod']) or anchor_year == "Unknown" or age == "Unknown":
        dod_age = None
    else:
        dod_age = int(patient["dod"].split("-")[0]) - anchor_year + age
    # if count == 6:
    if gender != "Unknown":
        patient_info_dict["gender"] = gender 
    # TODO: Remove age should be in admissions table
    if age != "Unknown":
        patient_info_dict["patient age"] = age 
    if dod_age != "Unknown":
        patient_info_dict["age of death"] = dod_age 
    # TODO: improve efficiency of this 
    patient_admissions = admissions[admissions['subject_id'] == patient_id]
    if not patient_admissions.empty:
        patient_info_dict["race"] = patient_admissions.iloc[0]["race"]
        # TODO: add relavent information to patient_admissions_dict {hadm_id1: {}, hadm_id2: {}}
        for idx, admission in patient_admissions.iterrows():
            hadm_id = admission["hadm_id"]
            patient_admissions_dict[hadm_id] = {}
            admission_diagnoses = get_diagnoses(hadm_id)
            sorted_procedures = get_hcpcsevents(hadm_id)
            if admission_diagnoses:
                patient_admissions_dict[hadm_id]["diagnoses"] = admission_diagnoses
            if sorted_procedures:
                patient_admissions_dict[hadm_id]["hcpcs events"] = sorted_procedures
        # print(hcpcsevents)
        # print("deez nuts: ", d_hcpcs)
        print(patient_info_dict)
        print(patient_admissions_dict)
        break



