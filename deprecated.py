# old pipelines (ignore - deprecated)
import json
import re

def patient_similarity_v2(patients_info, patient_id, with_graph=False):
    patient_info = patients_info[patient_id]
    synth1 = "2"+patient_id[1:]
    synth1_info = patients_info[synth1]

    synth2 = "3"+patient_id[1:]
    synth2_info = patients_info[synth2]

    synth3 = "4"+patient_id[1:]
    synth3_info = patients_info[synth3]
    
    synth4 = "5"+patient_id[1:]
    synth4_info = patients_info[synth4]

    synth5 = "6"+patient_id[1:]
    synth5_info = patients_info[synth4]

    base_prompt = f"""

        I am modeling clinical interactions between patients, providers, drugs, prescriptions and more. 
        To do so, I am using a knowledge graph. This knowledge graph will then be used for various downstream 
        applications. 

     """

    if not with_graph:
        similarity_prompt = f""" 
            Given the above knowledge graph relation triples, perform a comprehensive similarity analysis between:
            1. Patient {patient_id}: {patient_info} and Patient {synth1}: {synth1_info} 
            
            2. Patient {patient_id}: {patient_info} and Patient {synth2}: {synth2_info}
            
            3. Patient {patient_id}: {patient_info} and Patient {synth3}: {synth3_info}

            4. Patient {patient_id}: {patient_info} and Patient {synth4}: {synth4_info}

            5. Patient {patient_id}: {patient_info} and Patient {synth5}: {synth5_info}

            in the knowledge graph.
            
            Your objective is to evaluate how similar each patient pair is across key clinical dimensions, including prescriptions, medical procedures, symptoms, and other relevant medical factors.

            Use the relation types in the knowledge graph triples to determine which category each item belongs to.
            For each category, assign a similarity score between 0 and 1 by analyzing overlapping items. If a category is missing for either patient or if there are no overlapping items, assign a similarity score of 0 for that category.

            Then, compute an overall similarity score by adding all scores and then dividing by the number of individual category scores, which is 4 here. Additionally, list the key contributors that explain the observed similarities. 
            If a cateogry has no key contributors, then the cateogry similarity score should accordingly be 0.
            
            Perform this pairwise similairty analysis between the patient and five other patients listed above. Output all of these pairs. All results should be included in the json format below.

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
    else:
        similarity_prompt = f""" 
            Given the above knowledge graph relation triples, perform a comprehensive similarity analysis between:
            1. Patient {patient_id}: {patient_info} and Patient {synth1}: {synth1_info} 
            
            2. Patient {patient_id}: {patient_info} and Patient {synth2}: {synth2_info}
            
            3. Patient {patient_id}: {patient_info} and Patient {synth3}: {synth3_info}

            4. Patient {patient_id}: {patient_info} and Patient {synth4}: {synth4_info}

            5. Patient {patient_id}: {patient_info} and Patient {synth5}: {synth5_info}

            in the knowledge graph.
            
            Your objective is to evaluate how similar each patient pair is across key clinical dimensions, including prescriptions, medical procedures, symptoms, and other relevant medical factors.

            Use the relation types in the knowledge graph triples to determine which category each item belongs to.
            For each category, assign a similarity score between 0 and 1 by analyzing overlapping items. If a category is missing for either patient or if there are no overlapping items, assign a similarity score of 0 for that category.

            Then, compute an overall similarity score by adding all scores and then dividing by the number of individual category scores, which is 4 here. Additionally, list the key contributors that explain the observed similarities. 
            If a cateogry has no key contributors, then the cateogry similarity score should accordingly be 0.
            
            Perform this pairwise similairty analysis between the patient and five other patients listed above. Output all of these pairs. All results should be included in the json format below.

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
            return data
    return response

def generate_injection(patient, patient_diagnosis, patient_details, synth_num):
    base_prompt = f"""

        I am modeling clinical interactions between patients, providers, drugs, prescriptions and more. 
        To do so, I am using a knowledge graph. This knowledge graph will then be used for various downstream 
        applications. 

     """
        
    prompt = f"""
    Synthesize data for Patient {str(synth_num)} that has that has the same diagnosis as {patient} diagnosis: 
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