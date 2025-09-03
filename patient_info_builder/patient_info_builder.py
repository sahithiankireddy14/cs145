import pandas as pd
import os
import json
from tqdm import tqdm
from clinical_text_extractor import extract_symptoms


class PatientInfoBuilder:


    def __init__(self):
        pd.set_option("display.max_columns", None)
        data_base = "/Users/sahithi/Desktop/Research/physionet.org/files/mimiciv/3.1/hosp"
        self.clinical_data_base = "/Users/sahithi/Desktop/Research/physionet.org/files/mimic-iv-note/2.2/note"

        # --- Core tables ---
        self.df = pd.read_csv(os.path.join(data_base, "patients.csv.gz"))
        self.diagnosis_codes = pd.read_csv(os.path.join(data_base, "diagnoses_icd.csv.gz"))
        self.diagnosis_names = pd.read_csv(os.path.join(data_base, "d_icd_diagnoses.csv.gz"))
        self.hcpcs_codes = pd.read_csv(os.path.join(data_base, "d_hcpcs.csv.gz"))
        self.hcpcsevents = pd.read_csv(os.path.join(data_base, "hcpcsevents.csv.gz"))
        self.admissions = pd.read_csv(os.path.join(data_base, "admissions.csv.gz"))
        self.drgcodes = pd.read_csv(os.path.join(data_base, "drgcodes.csv.gz"))

        # --- Chunked big tables ---
        self.poe_chunks = self.chunk_file(os.path.join(data_base, "poe.csv.gz"))
        self.pharmacy_chunks = self.chunk_file(os.path.join(data_base, "pharmacy.csv.gz"))
        # issue with lab events file not loading
        #self.labevents_chunks = self.chunk_file(os.path.join(data_base, "labevents.csv.gz"))
        self.emar_chunks = self.chunk_file(os.path.join(data_base, "emar.csv.gz"))

        # --- Clinical notes ---
        self.discharge_notes = pd.read_csv(os.path.join(self.clinical_data_base, "discharge.csv.gz"))

        # --- Dictionaries for lookup ---
        self.lab_codes = pd.read_csv(os.path.join(data_base, "d_labitems.csv.gz"))
        self.procedure_codes = pd.read_csv(os.path.join(data_base, "d_icd_procedures.csv.gz"))
        self.all_patient_procedures = pd.read_csv(os.path.join(data_base, "procedures_icd.csv.gz"))
        self.all_patient_prescriptions = pd.read_csv(os.path.join(data_base, "prescriptions.csv.gz"))


    def chunk_file(self, file_path):
        chunksize = 100000
        reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=False)
        chunks = []
        while True:
            try:
                chunk = next(reader)
                chunks.append(chunk)
            except StopIteration:
                break
        return chunks
    
    def get_filtered_chunks(self, keyword, value, chunks):
        filtered_chunks = []
        for chunk in chunks:
            matches = chunk[chunk[keyword] == value]
            if not matches.empty:
                filtered_chunks.append(matches)
        return filtered_chunks
  
    def get_diagnoses(self, hadm_id):
        admission_codes = self.diagnosis_codes[self.diagnosis_codes["hadm_id"] == hadm_id]
        admission_diagnoses = []
        if not admission_codes.empty:
            codes = admission_codes["icd_code"].dropna().tolist()
            for code in codes:
                diagnosis_name = self.diagnosis_names[self.diagnosis_names["icd_code"] == code]
                if not diagnosis_name.empty:
                    admission_diagnoses.append(diagnosis_name["long_title"].iloc[0])
        return admission_diagnoses

    def get_drg(self, hadm_id):
        admission_codes = self.drgcodes[self.drgcodes["hadm_id"] == hadm_id]
        admission_drgs = []
        if not admission_codes.empty:
            for _, row in admission_codes.iterrows():
                admission_drg = {}
                for key in ["drg_type", "description", "drg_severity"]:
                    if not pd.isna(row[key]):
                        admission_drg[key.split("drg_")[-1]] = row[key]
                if admission_drg:
                    admission_drgs.append(admission_drg)
        return admission_drgs

    def get_hcpcsevents(self, hadm_id):
        LARGE_NUMBER = 1000000000000
        admission_events = self.hcpcsevents[self.hcpcsevents["hadm_id"] == hadm_id]
        admission_procedures = []
        if not admission_events.empty:
            for _, row in admission_events.iterrows():
                admission_procedure = {}
                if not pd.isna(row["short_description"]):
                    admission_procedure["actual_short_description"] = row["short_description"]
                if not pd.isna(row["hcpcs_cd"]):
                    procedure_code = row["hcpcs_cd"]
                    procedure_desc = self.hcpcs_codes[self.hcpcs_codes["code"] == procedure_code]
                    if not procedure_desc.empty:
                        if not pd.isna(procedure_desc["short_description"].iloc[0]):
                            admission_procedure["official_short_description"] = procedure_desc[
                                "short_description"
                            ].iloc[0]
                        if not pd.isna(procedure_desc["long_description"].iloc[0]):
                            admission_procedure["official_long_description"] = procedure_desc[
                                "long_description"
                            ].iloc[0]
                if not pd.isna(row["seq_num"]):
                    admission_procedure["sequence_number"] = int(row["seq_num"])
                elif pd.isna(row["seq_num"]) and admission_procedure:
                    admission_procedure["sequence_number"] = LARGE_NUMBER
                if admission_procedure:
                    admission_procedures.append(admission_procedure)
            admission_procedures = sorted(admission_procedures, key=lambda x: x["sequence_number"])
        return admission_procedures

   
    def symptoms(self, hadm_id):
        discharge_note = self.discharge_notes[self.discharge_notes["hadm_id"] == hadm_id]
        if discharge_note.empty:
            return []
        text = discharge_note["text"].iloc[0]
        return extract_symptoms(text)


    def get_prescriptions(self, hadm_id):
        admission_prescriptions = self.all_patient_prescriptions[
            self.all_patient_prescriptions["hadm_id"] == hadm_id
        ]
        if not admission_prescriptions.empty:
            admission_prescriptions = admission_prescriptions.sort_values(by="starttime")
            return admission_prescriptions["drug"].dropna().tolist()
        return []

  

    def get_procedures(self, hadm_id):
        admission_procedures = self.all_patient_procedures[
            self.all_patient_procedures["hadm_id"] == hadm_id
        ]
        if not admission_procedures.empty:
            merged_df = admission_procedures.merge(self.procedure_codes, on="icd_code", how="left")
            return merged_df["long_title"].dropna().tolist()
        return []


    # Lab events: file not loading issues?
    def get_labs(self, pid, admit_time, discharge_time):
        filtered_chunks = self.get_filtered_chunks("subject_id", pid, self.labevents_chunks)
        if not filtered_chunks:
            return []
        patient_events = pd.concat(filtered_chunks, ignore_index=True)
        patient_events_in_range = patient_events[
            (pd.to_datetime(patient_events["charttime"]) >= pd.to_datetime(admit_time))
            & (pd.to_datetime(patient_events["charttime"]) <= pd.to_datetime(discharge_time))
        ]
        if not patient_events_in_range.empty:
            merged_df = patient_events_in_range.merge(self.lab_codes, on="itemid", how="left")
            return merged_df["label"].dropna().tolist()
        return []


    def get_emar(self, hadm_id):
        filtered_chunks = self.get_filtered_chunks("hadm_id", hadm_id, self.emar_chunks)
        if not filtered_chunks:
            return []
        admission_emars = pd.concat(filtered_chunks, ignore_index=True)
        emar_outputs = []
        for _, row in admission_emars.iterrows():
            emar_output = {}
            if not pd.isna(row["emar_id"]):
                emar_output["emar_id"] = row["emar_id"]
            if not pd.isna(row["emar_seq"]):
                emar_output["sequence_number"] = int(row["emar_seq"])
            if "medication" in row and not pd.isna(row["medication"]):
                emar_output["medication"] = row["medication"]
            if "charttime" in row and not pd.isna(row["charttime"]):
                emar_output["time"] = str(row["charttime"])
            if emar_output:
                emar_outputs.append(emar_output)
        return emar_outputs
    


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

  
    def get_providers(self, hadm_id):
        
            providers = set()

            # From POE
            poe_events = self.get_poe(hadm_id)
            for event_dict in poe_events:
                for _, event in event_dict.items():
                    if "ordered_by" in event:
                        providers.add(event["ordered_by"])

            # From EMAR
            emar_events = self.get_emar(hadm_id)
            for event in emar_events:
                if "provider_id" in event:
                    providers.add(f"Provider_{event['provider_id']}")

            return sorted(providers)



    # Admission table
    def generate_patient_admission_table(self, patient_admissions, patient_id):
        patient_admissions_dict = {}
        for _, admission in patient_admissions.iterrows():
            hadm_id = admission["hadm_id"]
            patient_admissions_dict[hadm_id] = {}
            patient_admissions_dict[hadm_id]["diagnoses"] = self.get_diagnoses(hadm_id)
            patient_admissions_dict[hadm_id]["hcpcs_events"] = self.get_hcpcsevents(hadm_id)
            patient_admissions_dict[hadm_id]["diagnosis_related_group"] = self.get_drg(hadm_id)
            patient_admissions_dict[hadm_id]["physician_order_entry"] = self.get_poe(hadm_id)
            patient_admissions_dict[hadm_id]["pharmacy"] = self.get_pharmacy(hadm_id)

            patient_admissions_dict[hadm_id]["procedures"] = self.get_procedures(hadm_id)
            patient_admissions_dict[hadm_id]["prescriptions"] = self.get_prescriptions(hadm_id)
            patient_admissions_dict[hadm_id]["symptoms"] = self.symptoms(hadm_id)
            patient_admissions_dict[hadm_id]["emar"] = self.get_emar(hadm_id)
            patient_admissions_dict[hadm_id]["providers_involved"] = self.get_providers(hadm_id)
            # patient_admissions_dict[hadm_id]["lab_events"] = self.get_labs(
            #     patient_id, admission["admittime"], admission["dischtime"]
            # )
        return patient_admissions_dict
    

    # Patient Info Table
    def generate_patient_info_table(self, patient):
        patient_info_dict = {}
        gender = patient["gender"] if not pd.isna(patient["gender"]) else "Unknown"
        anchor_year = int(patient["anchor_year"]) if not pd.isna(patient["anchor_year"]) else "Unknown"
        age = int(patient["anchor_age"]) if not pd.isna(patient["anchor_age"]) else "Unknown"
        if pd.isna(patient["dod"]) or anchor_year == "Unknown" or age == "Unknown":
            dod_age = None
        else:
            dod_age = int(patient["dod"].split("-")[0]) - anchor_year + age
        if gender != "Unknown":
            patient_info_dict["gender"] = gender
        if age != "Unknown":
            patient_info_dict["patient_age"] = age
        if dod_age:
            patient_info_dict["age_of_death"] = dod_age
        return patient_info_dict

    # Generate all info per patient (patient demographical + admissions data)
    def patient_loop(self, patient):
        patient_id = patient["subject_id"]
        patient_info_dict = self.generate_patient_info_table(patient)
        patient_admissions = self.admissions[self.admissions["subject_id"] == patient_id]
        if not patient_admissions.empty:
            patient_info_dict["race"] = patient_admissions.iloc[0]["race"]
            patient_admissions_dict = self.generate_patient_admission_table(patient_admissions, patient_id)
        else:
            patient_info_dict["race"] = "Unknown"
            patient_info_dict["marital status"] = "Unknown"
            patient_admissions_dict = {}
        return patient_info_dict, patient_admissions_dict


    # Runner
    def build_all_patients(self, output_dir=None, limit=None):
        """
        Loop over all patients, build info+admission dicts, and optionally save to JSON.
        """
        results = {}
        patients = self.df
        if limit:
            patients = patients.head(limit)

        for _, patient in tqdm(patients.iterrows(), total=len(patients)):
            pid = patient["subject_id"]
            info, admissions = self.patient_loop(patient)
            results[pid] = {"info": info, "admissions": admissions}

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f"{pid}.json"), "w") as f:
                    json.dump(results[pid], f, indent=2, default=str)

        if not output_dir:
            return results
