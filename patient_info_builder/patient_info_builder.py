import pandas as pd
import os
from image_utils import * 
# from clinical_text_extractor import *
# install pyarrow
                                        
class PatientInfoBuilder:
    def chunk_file(self, file_path):
        chunksize = 100000  
        reader = pd.read_csv(file_path, chunksize=chunksize)
        
        chunks = []
        while True:
            try:
                chunk = next(reader)
                chunks.append(chunk)
            except StopIteration:
                break
        
        return chunks
        


    def __init__(self):
        pd.set_option("display.max_columns", None) 
        # path to hosp folder
        data_base = "/Users/psehgal/Documents/physionet.org/files/mimiciv/3.1/hosp"
        # self.clinical_data_base = "/Users/sahithi/Desktop/Research/physionet.org/files/mimic-iv-note/2.2"
        # data_base = "/Users/sahithi/Desktop/Research/physionet.org/files/mimiciv/3.1/hosp"

    # read relavant csv's
        self.df = pd.read_csv(os.path.join(data_base, "patients.csv.gz"))
        self.diagnosis_codes = pd.read_csv(os.path.join(data_base, "diagnoses_icd.csv.gz"))
        self.diagnosis_names = pd.read_csv(os.path.join(data_base, "d_icd_diagnoses.csv.gz"))
        self.hcpcs_codes = pd.read_csv(os.path.join(data_base, "d_hcpcs.csv.gz"))
        self.hcpcsevents = pd.read_csv(os.path.join(data_base, "hcpcsevents.csv.gz"))
        self.admissions = pd.read_csv(os.path.join(data_base, "admissions.csv.gz"))
        self.drgcodes = pd.read_csv(os.path.join(data_base, "drgcodes.csv.gz"))
        self.poe_chunks = self.chunk_file(os.path.join(data_base, "poe.csv.gz"))
        self.pharmacy_chunks = self.chunk_file(os.path.join(data_base, "pharmacy.csv.gz"))
        # TODO: admissions data
         
        # self.discharge_notes  = pd.read_csv(os.path.join(self.clinical_data_base, "discharge.csv.gz"))
        # self.emar_chunks = self.chunk_file(os.path.join(self.data_base, "emar.csv.gz"))
        
        # self.lab_codes = pd.read_csv(os.path.join(self.data_base, "d_labitems.csv.gz"))
        # self.procedure_codes =  pd.read_csv(os.path.join(self.data_base, "d_icd_procedures.csv.gz"))
        # self.all_patient_procedures =  pd.read_csv(os.path.join(self.data_base, "procedures_icd.csv.gz"))
        #self.all_patient_labs = pd.read_csv(os.path.join(self.data_base, "labevents.csv.gz"))
        # self.all_patient_labs_chunks = self.chunk_file(os.path.join(self.data_base, "labevents.csv.gz"))
        # self.all_patient_prescriptions = pd.read_csv(os.path.join(self.data_base, "prescriptions.csv.gz"))

    def get_diagnoses(self, hadm_id):
        admission_codes = self.diagnosis_codes[self.diagnosis_codes["hadm_id"] == hadm_id]
        if not admission_codes.empty:
            admission_codes = admission_codes["icd_code"].dropna().tolist()
            admission_diagnoses = []
            for admission_code in admission_codes:
                diagnosis_name = self.diagnosis_names[self.diagnosis_names["icd_code"] == admission_code]
                if not diagnosis_name.empty:
                    admission_diagnoses.append(diagnosis_name["long_title"].iloc[0])
        return admission_diagnoses

    def get_drg(self, hadm_id):
        admission_codes = self.drgcodes[self.drgcodes["hadm_id"] == hadm_id]
        if not admission_codes.empty:
            admission_drgs = []
            for _, row in admission_codes.iterrows():
                admission_drg = {}
                for key in ["drg_type", "description", "drg_severity"]:
                    if not pd.isna(row[key]):
                        admission_drg[key.split("drg_")[-1]] = row[key]
                if admission_drg:
                    admission_drgs.append(admission_drg)
            # TODO: change return if need to output empty list
            return admission_drgs
        
    def get_hcpcsevents(self, hadm_id):
        LARGE_NUMBER = 1000000000000
        admission_events = self.hcpcsevents[self.hcpcsevents["hadm_id"] == hadm_id]
        if not admission_events.empty:
            admission_procedures = []
            for idx, row in admission_events.iterrows():
                admission_procedure = {}
                if not pd.isna(row["short_description"]):
                    admission_procedure["actual_short_description"] = row["short_description"]
                if not pd.isna(row["hcpcs_cd"]):
                    procedure_code = row["hcpcs_cd"]
                    procedure_desc = self.hcpcs_codes[self.hcpcs_codes["code"] == procedure_code]
                    if not procedure_desc.empty:
                        long_desc = procedure_desc["long_description"].iloc[0]
                        official_short_desc = procedure_desc["short_description"].iloc[0]
                        if not pd.isna(official_short_desc):
                            admission_procedure["official_short_description"] = official_short_desc
                        if not pd.isna(long_desc):
                            admission_procedure["official_long_description"] = long_desc
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

    def symptoms(self, hadm_id):
        discharge_note = self.discharge_notes[self.discharge_notes['hadm_id'] == hadm_id]['text']
        symptoms = extract_symptoms(discharge_note)
        return symptoms

    def images(self, hadm_id):
        discharge_note = self.discharge_notes[self.discharge_notes['hadm_id'] == hadm_id]['text']
        pass 

    def get_prescriptions(self, hadm_id):
        # all drugs given to patient per admision are returned
        # the drugs are sorted by startime (so first drug in list was adminstered first etc.)
        admission_prescriptions = self.all_patient_prescriptions[self.all_patient_prescriptions['hadm_id'] == hadm_id]
        if not admission_prescriptions.empty:
            merged_df = admission_prescriptions.merge(self.all_patient_prescriptions, on='hadm_id', how='left')
            merged_df = merged_df.sort_values(by='starttime')
            return merged_df['drug'].tolist()
        return []
        
              
    def get_procedures(self, hadm_id):
        admission_procedures = self.all_patient_procedures[self.all_patient_procedures['hadm_id'] == hadm_id]
        if not admission_procedures.empty:
            merged_df = admission_procedures.merge(self.procedure_codes, on='icd_code', how='left')
            return merged_df['long_title'].tolist()
        return []
              
         
    def get_labs(self, pid, admit_time, discharge_time):

        patient_events = self.get_filtered_chunks("'subject_id'", pid, self.all_patient_labs_chunks)
        #patient_events = self.all_patient_labs[self.all_patient_labs['subject_id'] == pid]
        patient_events_in_range = patient_events[
            (pd.to_datetime(patient_events['charttime']) >= pd.to_datetime(admit_time)) &
            (pd.to_datetime(patient_events['charttime']) <= pd.to_datetime(discharge_time))
        ]
        
        if not patient_events_in_range.empty:
            merged_df = patient_events_in_range.merge(self.lab_codes, on='itemid', how='left')
            return merged_df['label'].tolist()
        return []
                   

    def generate_patient_admission_table(self, patient_admissions, patient_id):
        patient_admissions_dict = {}
        # TODO: add relavent information to patient_admissions_dict {hadm_id1: {}, hadm_id2: {}}
        for _, admission in patient_admissions.iterrows():
            hadm_id = admission["hadm_id"]
            patient_admissions_dict[hadm_id] = {}
            admission_diagnoses = self.get_diagnoses(hadm_id)
            hcpcs_events = self.get_hcpcsevents(hadm_id)
            drg = self.get_drg(hadm_id)
            poe = self.get_poe(hadm_id)
            pharmacy = self.get_pharmacy(hadm_id)
            patient_admissions_dict[hadm_id]["diagnoses"] = admission_diagnoses
            patient_admissions_dict[hadm_id]["hcpcs_events"] = hcpcs_events
            patient_admissions_dict[hadm_id]["diagnosis_related_group"] = drg
            patient_admissions_dict[hadm_id]["physician_order_entry"] = poe
            patient_admissions_dict[hadm_id]["pharmacy"] = pharmacy

            # no hadm_id in labs events chart, so going based off admit time and admission time
            # patient_admissions_dict[hadm_id]['lab events'] = self.get_labs(patient_id, admission['admittime'], admission['dischtime'])
            # patient_admissions_dict[hadm_id]['procedures'] = self.get_procedures(hadm_id)
            # patient_admissions_dict[hadm_id]['prescriptions'] = self.get_prescriptions(hadm_id)
            # patient_admissions_dict[hadm_id]['symptoms'] = self.symptoms(hadm_id)
        return patient_admissions_dict
    

    def get_filtered_chunks(self, keyword, hadm_id, chunks):
        filtered_chunks = []
        for chunk in chunks:
            matches = chunk[chunk[keyword] == hadm_id]
            if not matches.empty:
                filtered_chunks.append(matches)
        if not filtered_chunks:
            return []
        return filtered_chunks
    
    def get_poe(self, hadm_id):
        # TODO: do poe details
        # TODO: can map by change, discontinued, etc, types 
        # TODO: make sure to map discontinued and discontinues 
        filtered_chunks = self.get_filtered_chunks("hadm_id", hadm_id, self.poe_chunks)
        if filtered_chunks:
            admission_poes = pd.concat(filtered_chunks, ignore_index=True)
            admission_poes = admission_poes.sort_values("poe_seq")
            poe_events = []
            for _, row in admission_poes.iterrows():
                final_event = {}
                event = {}
                if pd.isna(row["poe_id"]):
                    continue
                poe_id = row["poe_id"]
                if pd.notna(row["order_type"]):
                    event["order_type"] = row["order_type"]
                if pd.notna(row["order_status"]):
                    event["order_status"] = row["order_status"]
                if pd.notna(row["transaction_type"]):
                    event["transaction_type"] = row["transaction_type"]
                if pd.notna(row["ordertime"]):
                    event["order_time"] = str(row["ordertime"])
                if pd.notna(row["poe_seq"]):
                    event["sequence_number"] = int(row["poe_seq"])
                if pd.notna(row["order_provider_id"]):
                    event["ordered_by"] = f"Provider_{row['order_provider_id']}"
                if pd.notna(row["discontinue_of_poe_id"]):
                    event["discontinue_of"] = int(row["discontinue_of_poe_id"].split("-")[1])
                if pd.notna(row["discontinued_by_poe_id"]):
                    event["discontinued_by"] = int(row["discontinued_by_poe_id"].split("-")[1])
                final_event[poe_id] = event
                if event:
                    poe_events.append(final_event)
                
            return poe_events

            
    def get_pharmacy(self, hadm_id):
        filtered_chunks = self.get_filtered_chunks("hadm_id", hadm_id, self.pharmacy_chunks)
        if filtered_chunks:
            pharmacy = pd.concat(filtered_chunks, ignore_index=True)
            pharmacy_events = []
            for _, row in pharmacy.iterrows():
                poe_id = None
                if pd.notna(row["poe_id"]):
                    poe_id = row["poe_id"]
                excluded = ["subject_id", "hadm_id", "pharmacy_id", "poe_id"]
                row = row.drop(labels=excluded)
                start = row["starttime"]
                end = row["stoptime"]
                length = -1
                if not pd.isna(start) and not pd.isna(end):
                    start = pd.to_datetime(start)
                    end = pd.to_datetime(end)
                    length = end - start
                    length = length.total_seconds() / 60
                if length <= 0:
                    length = None
                row["duration"] = length
                # TODO: do processing with one of these
                row = row.drop("stoptime")
                pharm_dict = row.to_dict()
                if poe_id:
                    pharm_dict = {poe_id: pharm_dict}
                pharmacy_events.append(pharm_dict)
            return pharmacy_events



    # def get_emar(self):
    #     # TODO: do emar details
    #     filtered_chunks = []
    #     for chunk in self.emar_chunks:
    #         matches = chunk[chunk["hadm_id"] == hadm_id]
    #         if not matches.empty:
    #             filtered_chunks.append(matches)
    #     admission_emars = pd.concat(filtered_chunks, ignore_index=True) 
    #     if not admission_emars.empty:
    #         emar_outputs = []
    #         for _, row in admission_emars.iterrows():
    #             emar_output = {}
    #             if not pd.isna(row["emar_id"]):
    #                 emar_id = row["emar_id"]
    #             if not pd.isna(row["emar_seq"]):
    #                 emar_output["sequence number"] = row["emar_seq"]
    #                 poe_output = self.get_poe(poe_id)
    #                 if poe_output:
    #                     emar_output["Physicial Order Entry"] = poe_output
    #                 if


    def generate_patient_info_table(self, patient):
        patient_info_dict = {}
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
            patient_info_dict["patient_age"] = age 
        if dod_age != "Unknown":
            patient_info_dict["age_of_death"] = dod_age 
        return patient_info_dict

    def patient_loop(self, patient):
        patient_id = patient["subject_id"]
        patient_info_dict = self.generate_patient_info_table(patient)
        patient_admissions = self.admissions[self.admissions['subject_id'] == patient_id]
        if not patient_admissions.empty:
            patient_info_dict["race"] = patient_admissions.iloc[0]["race"]
            patient_admissions_dict = self.generate_patient_admission_table(patient_admissions, patient_id)
        else:
            patient_info_dict["race"] = "Unknown"
            patient_info_dict["marital status"] = "Unknown"
            patient_admissions_dict = {}
            
        # print(self.pharmacy_chunks[0][0:35])
        # print(patient_info_dict)
        # print(patient_admissions_dict)
        return patient_info_dict, patient_admissions_dict
    




