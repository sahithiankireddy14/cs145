import openai
import os
import pickle 
import json
import re
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr


openai.api_key = os.getenv("OPENAI_API_KEY")

TEST = False
test_triples_patient_1 = [
    "99384712 diagnosed_with Congestive Heart Failure",
    "99384712 diagnosed_with Chronic Kidney Disease",
    "99384712 has_symptom Fatigue",
    "99384712 has_symptom Swelling in Legs",
    "99384712 has_symptom Shortness of Breath",
    "99384712 treated_with Metoprolol",
    "99384712 treated_with Sertraline",
    "99384712 treated_with Furosemide",
    "99384712 treated_with Atorvastatin",
    "99384712 treated_with Hemodialysis",
    "99384712 has_symptom Chest Pain",
    "99384712 treated_with Albuterol Inhaler",
    "99384712 has_symptom Difficulty Concentrating"
]

test_triples_patient_2 = [
    "88246109 diagnosed_with Lupus",
    "88246109 diagnosed_with Generalized Anxiety Disorder",
    "88246109 has_symptom Fatigue",
    "88246109 has_symptom Shortness of Breath",
    "88246109 has_symptom Difficulty Concentrating",
    "88246109 treated_with Metoprolol",
    "88246109 treated_with Sertraline",
    "88246109 treated_with Prednisone",
    "88246109 treated_with Atorvastatin",
    "88246109 treated_with Albuterol Inhaler",
    "88246109 has_symptom Joint Pain",
    "88246109 has_symptom Swelling in Legs",
    "88246109 treated_with Furosemide"
]


def query_llm(prompt: str, model="gpt-4o-2024-08-06") -> str:
    model="gpt-4o"
    client = openai.OpenAI()  
   

    response = client.chat.completions.create(
        model=model,
       
        messages=[
            {"role": "system", "content": "You are an expert in clinical health analysis"},
            {"role": "user", "content": prompt}
            
        ],
        
    )
    return response.choices[0].message.content.strip()



def generate_benchmark():


    prompt = """
    Generate Benchmark Dataset for Patient Similarity

    You are generating a JSON benchmark dataset to evaluate patient similarity algorithms using clinical knowledge graph triples. Each patient is described using a list of subject–predicate–object triples that represent their clinical features, such as diagnoses, symptoms, and treatments.

    Triple Format:
    - Subject: Patient ID (a unique numerical identifier)
    - Predicate: One of the following:
        - diagnosed_with — a diagnosis or medical condition
        - has_symptom — a symptom experienced
        - treated_with — a medication or procedure administered
    - Object: A valid medical concept (diagnosis, symptom, or treatment)

    ✏️ Task:
    1. Generate structured clinical knowledge graph triples data for exactly **20 unique patients**.
    - Each patient should have 10-15 triples.
    - Ensure diversity in clinical profiles across patients, including:
        - Common chronic conditions (e.g., diabetes, hypertension, asthma)
        - Acute conditions (e.g., pneumonia, injury)
        - Psychological conditions (e.g., depression, anxiety)
        - Varied symptom presentations and treatments

    2. Then compute **pairwise similarity** between all unique pairs of patients (total of 190 pairs).
    - Use a ground truth similarity score between **0 and 1**:
        - `1.0` = nearly identical profiles
        - `0.0` = completely dissimilar
        - Intermediate values (e.g., 0.25, 0.5, 0.75) represent partial overlaps based on:
        - Shared diagnoses
        - Overlapping symptoms
        - Common treatments
        - Related clinical presentations or comorbidities

    3. Ensure:
    - A diverse spread of similarity scores across the full range [0.0–1.0]
    - Consistent terminology and formatting (no typos or fabricated medical terms)

    Output Format:

    ```json
    {
    "patients": {
        "10000001": [
        (10000001, diagnosed_with, Hypertension),
        (10000001, has_symptom, Headache),
        (10000001, treated_with, Lisinopril),
        ...
        ],
        ...
    },
    "pairwise_similarity": [
        {
        "patient_1": "10000001",
        "patient_2": "10000002",
        "ground_truth_similarity": 0.75
        },
        {
        "patient_1": "10000001",
        "patient_2": "10000003",
        "ground_truth_similarity": 0.25
        },
        ...
    ]
    }

    Guidelines:
    - Use only realistic and commonly used clinical terms.
    - Format each triple clearly and consistently.
    - Capture a wide variety of realistic clinical scenarios, including some with:
    - Similar diagnoses but different symptoms
    - Shared medications but different underlying diseases
    - Psychological vs physical conditions

    Output as JSON as shown above. Only use format accepted by JSON.
    """

    response = query_llm(prompt=prompt)
    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        result = match.group(1)
        with open("benchmark.json", "w") as f:
            data = json.loads(result)
            json.dump(data, f)



def create_patient_triples_string(knowldege_graph_triples):
        patient_number = None
        patient_list = []
        master_triple_string = ""
        patient_count = 0

        for triple in knowldege_graph_triples:
            s, p, o = triple[0], triple[1], triple[2]
            # if (isinstance(s, str) and "10000980" in s and "-" not in s) or s == 10000980:
            #     print(triple)
            #     print(patient_number)
            #     print(type(patient_number))
            #     print(s)
            #     print(type(s))

            if (p == "has_admission" or p =="has_gender" or p == "diagnosed_with") and (patient_number != s or (isinstance(s, str) and (str(patient_number) != s.split("_")[-1]))):
                # Dump previous patient's data
                if len(patient_list) > 0:
                    master_triple_string += "\n".join(patient_list) + "\n"
                    patient_list = []

                # Start new patient section
                # TODO: This may have empty sections 
                master_triple_string += f"\nPatient {s}\n"
                patient_count += 1

                if isinstance(s, str) and "_" in s:
                    patient_number = int(s.split("_")[-1])
                else:
                    patient_number = int(s)


            patient_list.append(f"{s} {p} {o}")
            # if patient_count >= 20:
            #  return master_triple_string

        # Dump any remaining triples
        if patient_list:
            master_triple_string += "\n".join(patient_list) + "\n"

        #print("Master triple: ", master_triple_string)
        print("patient count: ", patient_count)
        
        return master_triple_string

def strip_json_comments(text):
    return re.sub(r"//.*", "", text)

def reformat(formatted_relation_triples):
    prompt = f"""
        You are given a flat list of subject-predicate-object triples for multiple patients. Your task is to reorganize this list into a structured, human-readable summary grouped by:
            - Patient ID
            - Admissions
                - DRGs (with type, description, severity)
                - Medications
                - Procedures
                - Orders (with route and status)

            Use the following format:
            Patient <patient_id>  
            Admission <admission_id>  
                DRGs:  
                - <drg_id>: <type> - <description> (Severity <severity>)  
                Medications:  
                - <medication name>  
                Orders:  
                - <order_id>: <route>, <medication_status>  

            Here is the list of triples:
            {formatted_relation_triples}

            Now return the fully grouped and formatted output as specified. If the patient has no admission, only display demographic details (e.g., gender, age, race). Include
            all the patients' information and include no other information, explanation, and commentary. 
    """

    # prompt = prompt.format(formatted_relation_triples = formatted_relation_triples)
    return query_llm(prompt=prompt)


def patient_similarity_v2(patients_info, patient_id):
    patient_info = patients_info[patient_id]
    synth1 = "2"+patient_id[1:]
    synth1_info = patients_info[synth1]

    synth2 = "3"+patient_id[1:]
    synth2_info = patients_info[synth2]

    synth3 = "4"+patient_id[1:]
    synth3_info = patients_info[synth3]
    

    base_prompt = f"""

        I am modeling clinical interactions between patients, providers, drugs, prescriptions and more. 
        To do so, I am using a knowledge graph. This knowledge graph will then be used for various downstream 
        applications. 

     """

    similarity_prompt = f""" 
        Given the above knowledge graph relation triples, perform a comprehensive similarity analysis between:
        1. Patient {patient_id}: {patient_info} and Patient {synth1}: {synth1_info} 
        
        2. Patient {patient_id}: {patient_info} and Patient {synth2}: {synth2_info}
        
        3. Patient {patient_id}: {patient_info} and Patient {synth3}: {synth3_info}
        
        in the knowledge graph.
        Your objective is to evaluate how similar each patient pair is across key clinical dimensions, including prescriptions, medical procedures, symptoms, and other relevant medical factors.

        Use the relation types in the knowledge graph triples to determine which category each item belongs to.
        For each category, assign a similarity score between 0 and 1 by analyzing overlapping items. If a category is missing for either patient or if there are no overlapping items, assign a similarity score of 0 for that category.

        Then, compute an overall similarity score by adding all scores and then dividing by the number of individual category scores, which is 4 here. Additionally, list the key contributors that explain the observed similarities. 
        If a cateogry has no key contributors, then the cateogry similarity score should accordingly be 0.
        
        Perform this pairwise similairty analysis between the patient and three other patients listed above. Output all of these pairs. All results should be included in the json format below.

        First, explain your reasoning and then transfer results to following json format. Ensure to use the same cateogory fields in the final JSON. Show all the entries for all patients. 
        
        Do NOT include any comments or explanations inside the JSON block (e.g., no lines starting with `//` or containing annotations. Follow the exact formatting allowed by JSON:

            [
            {{
                "patient_1": "{patient_id}",
                "patient_2": "Patient ID",
                "sub_similarities": {{
                    "prescription_similarity": 0.75,
                    "procedure_similarity": 0.7, 
                    "symptom_similarity": 0.7
                }},
                "overall_similarity": 0.82,
                "key_contributors": ["high blood pressure", "shared insulin prescription"]
            }},
            ...
            ]

        """
    response = query_llm(base_prompt + "\n" + similarity_prompt)
    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        result = match.group(1)
        with open("patient_similarity_output.json", "a") as f:
            result = strip_json_comments(result)
            data = json.loads(result)
            # if patient_id == "10000719":
            #     print("Dataaaaaaa")
            #     print(data)
            json.dump(data, f)
            return data
    return response

def patient_similarity(formatted_relation_triples, patient=None, num_patients=None):

    base_prompt = f"""

        I am modeling clinical interactions between patients, providers, drugs, prescriptions and more. 
        To do so, I am using a knowledge graph. This knowledge graph will then be used for various downstream 
        applications. Here are all the relation triples part of the knowledge graph

        {formatted_relation_triples}

     """
    base_prompt = base_prompt.format(formatted_relation_triples = formatted_relation_triples)

    if not patient:
        similarity_prompt = """ 

            Given the above knowledge graph relation triples, perform a comprehensive similarity analysis between patients.
            Your objective is to evaluate how similar each patient pair is across key clinical dimensions, including prescriptions, medical procedures, symptoms, and other relevant medical factors.

            Use the relation types in the knowledge graph triples to determine which category each item belongs to.
            For each category, assign a similarity score between 0 and 1 by analyzing overlapping items. If a category is missing for either patient or if there are no overlapping items, assign a similarity score of 0 for that category.

            Then, compute an overall similarity score by adding all scores and then dividing by the number of individual category scores, which is 4 here. Additionally, list the key contributors that explain the observed similarities. 
            If a cateogry has no key contributors, then the cateogry similarity score should accordingly be 0.
            
            Perform this analysis for every possible pair of patients, not just a single pair. Give me every single one of these pairs. For example, if there are three patients. Patient 1, Patient
            2, and Patient 3, I would do the analysis for Patient 1 and Patient 2, Patient 1 and Patient 3 and Patient 2 and Patient 3.
            This way, I would have covered all the unique pairs of patients. Do they same for all the patients provided. 

            First, explain your reasoning and then transfer results to following json format. Ensure to use the same cateogory fields in the final JSON and follow formatting allowed by JSON
        
                [
                {
                    "patient_1": "Patient ID",
                    "patient_2": "Patient ID",
                    "sub_similarities": {
                        "prescription_similarity": 0.75,
                        "procedure_similarity": 0.7, 
                        "symptom_similarity": 0.7
                    },
                    "overall_similarity": 0.82,
                    "key_contributors": ["high blood pressure", "shared insulin prescription"]
                },
                ...
                ]
         """
        
    else:
        similarity_prompt = f""" 

        Given the above knowledge graph relation triples, perform a comprehensive similarity analysis between {patient} and all of the other patients in the knowledge graph.
        Your objective is to evaluate how similar each patient pair is across key clinical dimensions, including prescriptions, medical procedures, symptoms, and other relevant medical factors.

        Use the relation types in the knowledge graph triples to determine which category each item belongs to.
        For each category, assign a similarity score between 0 and 1 by analyzing overlapping items. If a category is missing for either patient or if there are no overlapping items, assign a similarity score of 0 for that category.

        Then, compute an overall similarity score by adding all scores and then dividing by the number of individual category scores, which is 4 here. Additionally, list the key contributors that explain the observed similarities. 
        If a cateogry has no key contributors, then the cateogry similarity score should accordingly be 0.
        
        Perform this pairwise similairty analysis between {patient} and all other patients found in the knowledge graph relation triples. Output all of these pairs. Since there are a total of {str(num_patients - 1)} patients in addition
        to this patient, the similarity analysis should be done {str(num_patients - 1)} times and all results should be included in the json format below.

        First, explain your reasoning and then transfer results to following json format. Ensure to use the same cateogory fields in the final JSON. Show all the entries for all patients. 
        
        Do NOT include any comments or explanations inside the JSON block (e.g., no lines starting with `//` or containing annotations. Follow the exact formatting allowed by JSON:
      
            [
            {{
                "patient_1": "{patient}",
                "patient_2": "Patient ID",
                "sub_similarities": {{
                    "prescription_similarity": 0.75,
                    "procedure_similarity": 0.7, 
                    "symptom_similarity": 0.7
                }},
                "overall_similarity": 0.82,
                "key_contributors": ["high blood pressure", "shared insulin prescription"]
            }},
            ...
            ]


        """
        
   
    response = query_llm(base_prompt + "\n" + similarity_prompt)
    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        result = match.group(1)
        with open("patient_similarity_output.json", "w") as f:
            result = strip_json_comments(result)
            data = json.loads(result)
            json.dump(data, f)
            return data
    return response
    


# TODO: another metric
def get_top_k_patients():
    pass
   

def generate_injection(patient, patient_diagnosis, patient_details, synth_num):
    base_prompt = f"""

        I am modeling clinical interactions between patients, providers, drugs, prescriptions and more. 
        To do so, I am using a knowledge graph. This knowledge graph will then be used for various downstream 
        applications. 

     """
        
    prompt = f"""
    Synthesize data for Patient 8{str(synth_num)} that has that has the same diagnosis as {patient} diagnosis: 
    {patient_diagnosis} and is similar to this patient {patient_details} in terms of prescription_similarity, procedure_similarity, and symptom_similarity and has the
    same diagnosis at the end. 

    You are generating a JSON benchmark dataset to evaluate patient similarity algorithms using clinical knowledge graph triples. 
    Each patient is described using a list of subject–predicate–object triples that represent their clinical features, such as diagnoses, symptoms, and treatments.

    Triple Format:
    - Subject: Patient ID (a unique numerical identifier)
    - Predicate: One of the following:
        - diagnosed_with — a diagnosis or medical condition
        - has_symptom — a symptom experienced
        - treated_with — a medication or procedure administered
    - Object: A valid medical concept (diagnosis, symptom, or treatment)

    ✏️ Task:
    1. Generate a structured clinical knowledge graph triple data for one patient.
    - The patient should have 10-15 triples.
    - Clinical profiles can include:
        - Common chronic conditions (e.g., diabetes, hypertension, asthma)
        - Acute conditions (e.g., pneumonia, injury)
        - Psychological conditions (e.g., depression, anxiety)
        - Varied symptom presentations and treatments

    Here are some examples {formatted_relation_triples} for other patients. Structure it similarily. 
    
    Output JSON format only

    Guidelines:
    - Use only realistic and commonly used clinical terms.
    - Format each triple clearly and consistently.
    - Psychological vs physical conditions

    Output as JSON as shown above. Only use format accepted by JSON.
    """

    response = query_llm(prompt=prompt)
    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        result = match.group(1)
        with open("benchmark.json", "w") as f:
            data = json.loads(result)
            json.dump(data, f)



# def inject_similar_diagnosis(patient_data):
#     prompt = """Give me data for a patient that is similar to this patient in terms of 
#     prescription_similarity, procedure_similarity, and symptom_similarity and has the
#     same diagnosis at the end. 

#     Put it in the following format exactly. Don't include explanations or any extra information:

#     You are given a flat list of subject-predicate-object triples for multiple patients. Your task is to reorganize this list into a structured, human-readable summary grouped by:
#             - Patient ID
#             - Admissions
#                 - DRGs (with type, description, severity)
#                 - Medications
#                 - Procedures
#                 - Orders (with route and status)

#             Use the following format:
#             Patient <patient_id>  
#             Admission <admission_id>  
#                 DRGs:  
#                 - <drg_id>: <type> - <description> (Severity <severity>)  
#                 Medications:  
#                 - <medication name>  
#                 Orders:  
#                 - <order_id>: <route>, <medication_status>  
#     """

def diagnosis_gt_similarity_v2(patients_diagnosis, patient_id):
    patient_diagnosis = patients_diagnosis[patient_id]
    synth1 = "2"+patient_id[1:]
    synth1_info = patients_diagnosis[synth1]

    synth2 = "3"+patient_id[1:]
    synth2_info = patients_diagnosis[synth2]

    synth3 = "4"+patient_id[1:]
    synth3_info = patients_diagnosis[synth3]
    


    diagnosis_prompt = f""" 
        Given the above knowledge graph relation triples, perform a comprehensive similarity analysis between:
        1. Patient {patient_id}: {patient_diagnosis} and Patient {synth1}: {synth1_info} 
        
        2. Patient {patient_id}: {patient_diagnosis} and Patient {synth2}: {synth2_info}
        
        3. Patient {patient_id}: {patient_diagnosis} and Patient {synth3}: {synth3_info}

        Given the above knowledge graph relation triples, perform a comprehensive similarity analysis between {patient_id} and all of the other patients listed above.
        Your objective is to evaluate how similar each patient pair is across key clinical dimensions, including factors like exact diagnosis matches, related conditions, or number of overlapping categories. 

        Then, assign an overall similarity score between 0 and 1. If information is missing, assign a similarity score of 0 for that category.

        First, explain your reasoning and then transfer results to following json format. Ensure to use the same cateogory fields in the final JSON. Show all the entries for all patients. 
        
        Do NOT include any comments or explanations inside the JSON block (e.g., no lines starting with `//` or containing annotations. Make sure to include the 
        actual other patient's id in the patient_2 section. Follow the exact formatting allowed by JSON:

            [
            {{
                "patient_1": "{patient_id}",
                "patient_2": "Patient ID",
                "overall_similarity": 0.82, 
            }},
            ...
            ]

        """
    
    response = query_llm(diagnosis_prompt)

    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        result = match.group(1)
        with open("groundtruth_sim.json", "a") as f:
            result = strip_json_comments(result)
            data = json.loads(result)
            json.dump(data, f)
            return data
    return response



def diagnosis_gt_similarity(gt_diag, patient=None, num_patients=None):
    
    if not patient:
        diagnosis_similarity = """ 

            Given the list of diagnoses for each patient: {gt_diag}, compute a similarity score based on how similar their combination of diagnoses is for each pair of patients. Think through the process step by step—consider factors like exact diagnosis matches, related conditions, or number of overlapping categories. Then, return a final similarity score between 0 and 1.
            Ensure to do this for each pairwise combination of patients. Output all of these pairs. 
            
            For example, if there are three patients. Patient 1, Patient
            2, and Patient 3, I would do the analysis for Patient 1 and Patient 2, Patient 1 and Patient 3 and Patient 2 and Patient 3.
            This way, I would have covered all the unique pairs of patients. Do they same for all the patients provided. 
            
            After reasoning step by step, present the results in the following JSON format. Ensure you use the exact same field names in the final JSON output and include no explanations or description.  Follow formatting allowed by JSON: 

                [
                {
                    "patient_1": "Patient ID",
                    "patient_2": "Patient ID",
                    "overall_similarity": 0.82, 
                },
                ...
                ]


            """
        
    else:
        # diagnosis_similarity = f""" 

        #     Given the list of diagnoses for each patient, compute a similarity score based on how similar {patient} is to all of the other patients in the knowledge graph. Think through the process step by step—consider factors like exact diagnosis matches, related conditions, or number of overlapping categories. Then, return a final similarity score between 0 and 1.
        #     Output all of these pairs.

        #     Perform this pairwise similairty analysis between {patient} and all other patients found in the knowledge graph relation triples. Output all of these pairs. Since there are a total of {str(num_patients - 1)} patients in addition
        #     to this patient, this similarity analysis should be done {str(num_patients - 1)} times and all results should be included in the json format below.
            
        #     After reasoning step by step, present the results in the following JSON format. Ensure you use the exact same field names in the final JSON output and include no explanations or description.  Show all the entries for all patients. 
            
        #     Do NOT include any comments or explanations inside the JSON block (e.g., no lines starting with `//` or containing annotations. Follow the exact formatting allowed by JSON: 

        #         [
        #         {{
        #             "patient_1": "{patient}", 
        #             "patient_2": "Patient ID",
        #             "overall_similarity": 0.82,
                    
        #         }},
        #         ...
        #         ]


        #     """
        

            diagnosis_similarity = f""" 

            Given the above knowledge graph relation triples, perform a comprehensive similarity analysis between {patient} and all of the other patients in the knowledge graph.
            Your objective is to evaluate how similar each patient pair is across key clinical dimensions, including factors like exact diagnosis matches, related conditions, or number of overlapping categories. 

            Then, assign an overall similarity score between 0 and 1. If information is missing, assign a similarity score of 0 for that category.

            Perform this pairwise similairty analysis between {patient} and all other patients found in the knowledge graph relation triples. Output all of these pairs. Since there are a total of {str(num_patients - 1)} patients in addition
            to this patient, the similarity analysis should be done {str(num_patients - 1)} times and all results should be included in the json format below.

            First, explain your reasoning and then transfer results to following json format. Ensure to use the same cateogory fields in the final JSON. Show all the entries for all patients. 
            
            Do NOT include any comments or explanations inside the JSON block (e.g., no lines starting with `//` or containing annotations. Follow the exact formatting allowed by JSON:
        
                [
                {{
                    "patient_1": "{patient}",
                    "patient_2": "Patient ID",
                    "sub_similarities": {{
                        "prescription_similarity": 0.75,
                        "procedure_similarity": 0.7, 
                        "symptom_similarity": 0.7
                    }},
                    "overall_similarity": 0.82,
                    "key_contributors": ["high blood pressure", "shared insulin prescription"]
                }},
                ...
                ]


            """
    
    base_prompt = base_prompt.format(gt_diag = gt_diag)
    response = query_llm(base_prompt + "\n" + diagnosis_similarity)

    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        result = match.group(1)
        with open("groundtruth_sim.json", "w") as f:
            result = strip_json_comments(result)
            data = json.loads(result)
            json.dump(data, f)
            print("Diagnosis dataaa: ", data)
            return data
    return response
    
    
def get_correlation(pred_list, gt_list):
    print("Pred list: ", pred_list)
    print("G list: ", gt_list)
    print("Getting Correlation")

    corr, p_value = spearmanr(pred_list, gt_list)
    print(f"Spearman correlation: {corr:.4f}, p-value: {p_value:.4g}")
    print(f"P-value: {p_value:.4g}")

    corr, p_value = pearsonr(pred_list, gt_list)
    print(f"Pearson correlation: {corr:.4f}")
    print(f"P-value: {p_value:.4g}")





def compare_similarities(sim_reports, gt_reports):
    gt_lookup = {}
    print(gt_reports)
    for gt in gt_reports:
        print((gt["patient_1"], gt["patient_2"]))
        if "patient_1" in gt and "patient_2" in gt:
            gt_lookup[(gt["patient_1"], gt["patient_2"])] = gt["overall_similarity"]

    print(gt_lookup)

    formatted_results = []
    full_count = 0
    partial_count = 0
    no_match_count = 0
    pred_sim_list = []
    gt_sim_list = []
    for sim in sim_reports:
        p1 = sim["patient_1"]
        p2 = sim["patient_2"]
        pred_sim = sim["overall_similarity"]
        gt_sim = gt_lookup.get((p1, p2), gt_lookup.get((p2, p1), None))
        print("Patient 1: ", p1)
        print("Patient 2: ", p2)
        print("P sim: ", pred_sim)
        print("G sim: ", gt_sim)
        if gt_sim is not None:
            pred_sim_list.append(pred_sim)
            gt_sim_list.append(gt_sim)
        if gt_sim is None:
            match_score = "N/A"
        else:
            diff = abs(pred_sim - gt_sim)
            if diff <= 0.1:
                match_score = "Full"
                full_count += 1
            elif diff <= 0.4:
                match_score ="Partial"
                partial_count += 1
            else:
                match_score = "No Match"
                no_match_count += 1
        formatted_string = (
            f"Patient [{p1}] vs [{p2}]\n"
            f"Predicted similarity: {pred_sim}\n"
            f"Ground truth similarity: {gt_sim}\n"
            f"Match: {match_score}\n"
        )
        formatted_results.append(formatted_string)
    

    with open("patient_similarity_output.txt", 'w') as f:
        f.write("\n".join(formatted_results))
        f.write(f"Full matches: {full_count}\nPartial matches: {partial_count}\nNo matches: {no_match_count}\n")
    get_correlation(pred_sim_list, gt_sim_list)
    return "\n".join(formatted_results)


def evaluate(formatted_relation_triples):
     # ASK LLM to seperate
    #  prompt = """
    #     You are given a structured summary of patient records. Separate this into two parts:
    #     Diagnosis Information: Include only lines related to DRG codes, types, descriptions, severities, and diagnosis groups.
    #     Remaining Information: Include everything else (e.g., medications, orders, procedures, demographics).
    #     Clearly label each section with a header:

    #     --- DIAGNOSIS INFORMATION ---
    #     <grouped diagnosis section>

    #     --- REMAINING INFORMATION ---
    #     <everything else>

    #     Here is the structured input:
    #     {formatted_relation_triples}

    # """
    #  response = query_llm(prompt=prompt.format(formatted_relation_triples = formatted_relation_triples))


     diagnosis_keywords = re.compile(
        r"\b(DRG\d+|DRG_\d+|has_diagnosis_related_group|diagnosis|severity|description|type|is_a)\b",
        re.IGNORECASE
    )
     other_info = []
     diagnosis_info = []
     lines = formatted_relation_triples.strip().split("\n")
     patient_diagnosis = {}
     patient_info = {}
     patient_num = 0
     for line in lines:
         match = re.search(r"Patient (\d+)", line)
         if match:
             diagnosis_info.append(line)
             patient_num += 1
             patient_id = match.group(1)
         else:
             pass
         if diagnosis_keywords.search(line):
             diagnosis_info.append(line)
             if patient_id in patient_diagnosis:
                 patient_diagnosis[patient_id].append(line)
             else:
                 patient_diagnosis[patient_id] = [line]
                
         else:
             other_info.append(line)
             if patient_id in patient_info:
                 patient_info[patient_id].append(line)
             else:
                 patient_info[patient_id] = [line]

     other_info = "\n".join(other_info)
     diagnosis_info = "\n".join(diagnosis_info)
    #  print("Regular info: ", formatted_stripped_data)
    #  print("Diagnosis info: ", diagnosis_info)
     
     # patient similarity without diagnosis 
     print("Patient numbers: ", patient_num)
    #  print(patient_info)
    #  print(patient_diagnosis)
     patient_sim_all = []
     gt_sim_all = []
     test_list = ['10000032', '10000084', '10000117', '10000560', '10000690', '10000719']
     for patient_id in test_list:
        #  patient_similarity(other_info, patient_id, patient_num)
         patient_sim = patient_similarity_v2(patient_info, patient_id)
         patient_sim_all.extend(patient_sim)
         gt_sim = diagnosis_gt_similarity_v2(patient_diagnosis, patient_id)
         gt_sim_all.extend(gt_sim)


     final_result = compare_similarities(patient_sim_all, gt_sim_all)

     return final_result
     

def run_benchmark():
    generate_benchmark()
    with open('benchmark.json', 'r') as f:
        data = json.load(f)  # load() expects a file object
    triples = []
    for pid_data in data['patients'].values():
        triples.extend(list(pid_data))
    
   
   


# the structured patient triples string for 20 patients (so we don't need to keep regenerating)
with open('triples_string_full.txt', 'r') as file:
    formatted_relation_triples = file.read()  

final = evaluate(formatted_relation_triples)
