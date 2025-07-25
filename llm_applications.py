import openai
import os
import pickle 
import json
import re
from pydantic import BaseModel
from typing import List, Dict


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
    client = openai.OpenAI()  
   

    response = client.chat.completions.create(
        model=model,
       
        messages=[
            {"role": "system", "content": "You are an expert in clinical health analysis"},
            {"role": "user", "content": prompt}
            
        ],
        
    )
    return response.choices[0].message.content.strip()



def create_patient_triples_string(knowldege_graph_triples):
    master_triple_string = ""
    if TEST:
        master_triple_string+=f"\n Patient 1  Information: \n"
        master_triple_string += "\n".join(test_triples_patient_1)

        master_triple_string+=f"\n Patient 2 Information: \n"
        master_triple_string += "\n".join(test_triples_patient_2)

    else: 
        patient_number = -1
        patient_count = 0
        for triple in knowldege_graph_triples:
            print(triple)
            s, p, o = triple
            if triple[1] == "has_admission" and patient_number != triple[0]:
                patient_number = triple[0]
                patient_count += 1
                print(patient_count)
                master_triple_string += f"\n Patient {patient_count} Information: \n"
                master_triple_string += "\n".join(patient_list)
                patient_list = []
            patient_list.append(f"{s} {p} {o}")

            # if patient_count > 100:
            #     print("Breaking at 100")
            #     break

    return master_triple_string

 
def patient_similarity(formatted_relation_triples, patient=None):

    base_prompt = """

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
            
            Perform this analysis for every possible pair of patients, not just a single pair.

            First, explain your reasoning and then transfer results to following json format. Ensure to use the same cateogory fields in the final json.
        
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

        Given the above knowledge graph relation triples, perform a comprehensive similarity analysis between {patient} and all of the other patients in the knoledge graph.
        Your objective is to evaluate how similar each patient pair is across key clinical dimensions, including prescriptions, medical procedures, symptoms, and other relevant medical factors.

        Use the relation types in the knowledge graph triples to determine which category each item belongs to.
        For each category, assign a similarity score between 0 and 1 by analyzing overlapping items. If a category is missing for either patient or if there are no overlapping items, assign a similarity score of 0 for that category.

        Then, compute an overall similarity score by adding all scores and then dividing by the number of individual category scores, which is 4 here. Additionally, list the key contributors that explain the observed similarities. 
        If a cateogry has no key contributors, then the cateogry similarity score should accordingly be 0.
        
        Perform this analysis for every possible pair of patients, not just a single pair.

        First, explain your reasoning and then transfer results to following json format. Ensure to use the same cateogory fields in the final json.
      
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
        
   
    response = query_llm(base_prompt + "\n" + similarity_prompt)
    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        result = match.group(1)
        print(result)
        with open("patient_similarity_output.json", "w") as f:
            data = json.loads(result)
            json.dump(data, f)
            return data
    return response
    


# TODO: another metric
def get_top_k_patients():
    pass
   

def diagnosis_gt_similarity(gt_diag):

    base_prompt = """

       Here is diagnosis information related to patients. 

        {gt_diag}

     """
    

    diagnosis_similarity = """ 

        Given the list of diagnoses for each patient, compute a similarity score based on how similar their combination of diagnoses is for each pair of patients. Think through the process step by step—consider factors like exact diagnosis matches, related conditions, or number of overlapping categories. Then, return a final similarity score between 0 and 1.
        Ensure to do this for each pairwise combination of patients. 
        
        After reasoning step by step, present the results in the following JSON format. Ensure you use the exact same field names in the final JSON output:

            [
            {
                "patient_1": "Patient ID",
                "patient_2": "Patient ID",
                "overall_similarity": 0.82,
                
            },
            ...
            ]


        """
    
    base_prompt = base_prompt.format(gt_diag = gt_diag)
    response = query_llm(base_prompt + "\n" + diagnosis_similarity)

    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        result = match.group(1)
        with open("groundtruth_sim.json", "w") as f:
            data = json.loads(result)
            json.dump(data, f)
            return data
    return response
    
    



def compare_similarities(sim_reports, gt_reports):
    gt_lookup = {
        (gt["patient_1"], gt["patient_2"]): gt["overall_similarity"]
        for gt in gt_reports
    }

    formatted_results = []
    for sim in sim_reports:
        p1 = sim["patient_1"]
        p2 = sim["patient_2"]
        pred_sim = sim["overall_similarity"]
        gt_sim = gt_lookup.get((p1, p2), None)

        if gt_sim is None:
            match_score = "N/A"
        else:
            diff = abs(pred_sim - gt_sim)
            if diff <= 0.1:
                match_score = "Full"
            elif diff <= 0.4:
                match_score ="Partial"
            else:
                match_score = "No Match"

        formatted_string = (
            f"Patient [{p1}] vs [{p2}]\n"
            f"Predicted similarity: {pred_sim}\n"
            f"Ground truth similarity: {gt_sim}\n"
            f"Match: {match_score}\n"
        )
        formatted_results.append(formatted_string)

    return "\n".join(formatted_results)


def evaluate(formatted_relation_triples):
     # remove diagnosis info
     formatted_stripped_data = []
     diagnosis_info = []
     lines = formatted_relation_triples.strip().split("\n")
     for line in lines:
         if "diagnosed_with" in line.lower():
             diagnosis_info.append(line)
         else:
             formatted_stripped_data.append(line)
     formatted_stripped_data = "/n".join(formatted_stripped_data)
     diagnosis_info = "\n".join(diagnosis_info)
     
     # patient similarity without diagnosis 
    #  patient_sim = patient_similarity(formatted_stripped_data)
     patient_sim = patient_similarity(formatted_stripped_data, patient)
     # ground truth similarity with only diagnosis 
     gt_sim = diagnosis_gt_similarity(diagnosis_info)
    
     final_result = compare_similarities(patient_sim, gt_sim)



     return final_result
     

# knowldege_graph_data= pickle.load(open("/Users/sahithi/Desktop/Research/cs145/patient_similarity_results.pkl", "rb"))
knowldege_graph_data= pickle.load(open("/Users/psehgal/Documents/cs145/patient_similarity_results.pkl", "rb"))
formatted_relation_triples = create_patient_triples_string([knowldege_graph_data])
#print(patient_similarity(formatted_relation_triples))
print(evaluate(formatted_relation_triples))