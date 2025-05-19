import pandas as pd
import os
pd.set_option("display.max_columns", None) 

# path to hosp folder
#data_base = "/Users/psehgal/Documents/physionet.org/files/mimiciv/3.1/hosp"
data_base = "/Users/sahithi/Desktop/Research/physionet.org/files/mimiciv/3.1/hosp"


# read relavant csv's
df = pd.read_csv(os.path.join(data_base, "patients.csv.gz"))
diagnosis_codes = pd.read_csv(os.path.join(data_base, "diagnoses_icd.csv.gz"))
diagnosis_names = pd.read_csv(os.path.join(data_base, "d_icd_diagnoses.csv.gz"))
hcpcs_codes = pd.read_csv(os.path.join(data_base, "d_hcpcs.csv.gz"))
hcpcsevents = pd.read_csv(os.path.join(data_base, "hcpcsevents.csv.gz"))
admissions = pd.read_csv(os.path.join(data_base, "admissions.csv.gz"))


lab_codes = pd.read_csv(os.path.join(data_base, "d_labitems.csv.gz"))
procedure_codes =  pd.read_csv(os.path.join(data_base, "d_icd_procedures.csv.gz"))
all_patient_procedures =  pd.read_csv(os.path.join(data_base, "procedures_icd.csv.gz"))
all_patient_labs = pd.read_csv(os.path.join(data_base, "labevents.csv.gz"))


    
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
        # Sahithi Question: I don't see a icd code attribute, are we adding this as new attribute? 
        icd_codes = admission_codes["icd_code"]
        # real world events 



def symptoms(pid):
    # TODO : extract symptoms for clinical notes 
    pass


def get_procedures(hadm_id):
     admission_procedures = all_patient_procedures[all_patient_procedures['hadm_id'] == hadm_id]
     merged_df = admission_procedures.merge(procedure_codes, on='icd_code', how='left')
     return merged_df['long_title'].tolist()
              
         
def get_labs(pid, admit_time, discharge_time):
    patient_events = all_patient_labs[all_patient_labs['subject_id'] == pid]
    patient_events_in_range = patient_events[
        (pd.to_datetime(patient_events['charttime']) >= pd.to_datetime(admit_time)) &
        (pd.to_datetime(patient_events['charttime']) <= pd.to_datetime(discharge_time))
     ]
    
    if not patient_events_in_range.empty:
        merged_df = patient_events_in_range.merge(lab_codes, on='itemid', how='left')
        return merged_df['label'].tolist()
    return []
                   


def main():
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
        # TODO: improve efficiency of this 
        patient_admissions = admissions[admissions['subject_id'] == patient_id]
        if not patient_admissions.empty:
            patient_info_dict["race"] = patient_admissions.iloc[0]["race"]
            # TODO: add relavent information to patient_admissions_dict {hadm_id1: {}, hadm_id2: {}}
            for idx, admission in patient_admissions.iterrows():
                hadm_id = admission["hadm_id"]
                patient_admissions_dict[hadm_id] = {}
                admission_diagnoses = get_diagnoses(hadm_id)
                # Sahithi Comment --> I think we can just set to empty list if nothing present?  
                if admission_diagnoses:
                    patient_admissions_dict[hadm_id]["diagnoses"] = admission_diagnoses
                else:
                    pass
                    # TODO: need to handle this?


                 # no hadm_id in labs events chart, so going based off admit time and admission time
                patient_admissions_dict[hadm_id]['lab events'] = get_labs(patient_id, admission['admittime'], admission['dischtime'])
                patient_admissions_dict[hadm_id]['procedures'] = get_procedures(hadm_id)

            # diagnosis_codes
        else:
            patient_info_dict["race"] = "Unknown"
        # print(hcpcsevents)
        print(patient_info_dict)
        print(patient_admissions_dict)
        break



main()


