import openai
import os
import pickle 
import json
import re
from pydantic import BaseModel
from typing import List, Dict


openai.api_key = os.getenv("OPENAI_API_KEY")

TEST = True
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
    
        for i, triple in enumerate(knowldege_graph_triples):
            master_triple_string += f"\n Patient {i + 1} Information: \n"
            master_triple_string += "\n".join([f"{s} {p} {o}" for s, p, o in triple])

    
    
    return master_triple_string

 
def patient_similarity(formatted_relation_triples):

    base_prompt = """

        I am modeling clinical interactions between patients, providers, drugs, prescriptions and more. 
        To do so, I am using a knowledge graph. This knowledge graph will then be used for various downstream 
        applications. Here are all the relation triples part of the knowledge graph


        {formatted_relation_triples}

     """
    base_prompt = base_prompt.format(triples = formatted_relation_triples)
    similarity_prompt = """ 
    


        Given the above knowledge graph relation triples, perform a comprehensive similarity analysis between patients.
        Your objective is to evaluate how similar each patient pair is across key clinical dimensions, including diagnoses, prescriptions, medical procedures, symptoms, and other relevant medical factors.

        Use the relation types in the knowledge graph triples to determine which category each item belongs to.
        For each category, assign a similarity score between 0 and 1 by analyzing overlapping items. If a category is missing for either patient or if there are no overlapping items, assign a similarity score of 0 for that category.

        Then, compute an overall similarity score by averaging the individual category scores. Additionally, list the key contributors that explain the observed similarities. 
        If a cateogry has no key contributors, then the cateogry similarity score should accordingly be 0.
        
        Perform this analysis for every possible pair of patients, not just a single pair.

        First, explain your reasoning and then transfer results to following json format. Ensure to use the same cateogory fields in the final json.

      
            [
            {
                "patient_1": "Patient ID",
                "patient_2": "Patient ID",
                "sub_similarities": {
                "diagnosis_similarity": 0.9,
                "prescription_similarity": 0.75,
                "procedure_similarity": 0.7
                },
                "overall_similarity": 0.82,
                "key_contributors": ["diabetes diagnosis", "shared insulin prescription"]
            },
            ...
            ]

        
        """
   
    response = query_llm(base_prompt + "\n" + similarity_prompt)
    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        result = match.group(1)
        with open("patient_similarity_output.json", "w") as f:
            data = json.loads(result)
            json.dump(data, f)

        return result
    return response
    


   


def evaluate(formatted_relation_triples):
     
     # remove diaganosis info
     formatted_stripped_data = []
     lines = formatted_relation_triples.strip().split("\n")
     for line in lines:
         if "diagnosed_with" in line.lower():
             continue
         else:
             formatted_stripped_data.append(line)
     formatted_stripped_data = "/n".join(formatted_stripped_data)

     print(formatted_stripped_data)

     similarity= patient_similarity(formatted_stripped_data)

     # TODO: another call to evalute the similarity with ground truth
     




knowldege_graph_data= pickle.load(open("patient_similarity_results.pkl", "rb"))
formatted_relation_triples = create_patient_triples_string([knowldege_graph_data])
# print(patient_similarity(formatted_relation_triples))
evaluate(formatted_relation_triples)