import openai
import os
import pickle 
import json


openai.api_key = os.getenv("OPENAI_API_KEY")

TEST = True
test_triples = [
    "99384712 received Furosemide",
    "99384712 received Salbutamol Nebulizer",
    "99384712 received Potassium Chloride",
    "99384712 received Sodium Chloride 0.9% Flush",
    "99384712 received Raltegravir",
    "99384712 received Acetaminophen",
    "99384712 received Influenza Vaccine Quadrivalent",
    "99384712 received Metformin",
    "99384712 received Lisinopril",
    "99384712 received Emtricitabine-Tenofovir (Truvada)",
    "99384712 received Atorvastatin",
    "99384712 received Heparin",
    "99384712 received Nicotine Patch"
]
        
def query_llm(prompt: str, model="gpt-4") -> str:
    client = openai.OpenAI()  # Uses the API key in env or config
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in clinical health analysis"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


# array of arrays, where each array has all triples for one patient
def create_patient_triples_string(knowldege_graph_triples):
    master_triple_string = ""
    for i, triple in enumerate(knowldege_graph_triples):
        master_triple_string += f"\n Patient {i + 1} Information: \n"
        master_triple_string += "\n".join([f"{s} {p} {o}" for s, p, o in triple])

    if TEST:
        master_triple_string+=f"\n Patient {len(knowldege_graph_triples) + 1} Information: \n"
        master_triple_string += "\n".join(test_triples)
    
    return master_triple_string

 
def patient_similarity(knowldege_graph_triples):

    triples = create_patient_triples_string([knowldege_graph_triples])
    print(triples)
    base_prompt = """

        I am modeling clinical interactions between patients, providers, drugs, prescriptions and more. 
        To do so, I am using a knowledge graph. This knowledge graph will then be used for various downstream 
        applications. Here are all the relation triples part of the knowledge graph


        {triples}

     """
    base_prompt = base_prompt.format(triples = triples)
    similarity_prompt = """ 
            
            Given the above knowledge graph relation triples, perform a comprehensive similarity analysis of the patients. 
            Your goal is to identify how patients are similar based on key medical attributes such as diagnoses, 
            prescriptions, medical procedures, symptoms, and other relevant clinical factors. Then attribute a final 
            simiarity score for each set of patients. Return only in JSON format. 

            Example JSON output format:
                

            ```json
            [
            {
                "patient_1": "Patient_A",
                "patient_2": "Patient_B",
                "overall_similarity": 0.82,
                "sub_similarities": {
                "diagnosis_similarity": 0.9,
                "prescription_similarity": 0.75,
                "procedure_similarity": 0.7
                },
                "key_contributors": ["diabetes diagnosis", "shared insulin prescription"]
            },
            ...
            ]
                    
        """


    print(base_prompt + "\n" + similarity_prompt)
    response = query_llm(base_prompt + "\n" + similarity_prompt)
    with open("patient_similarity_output.json", "w") as f:
        json.dump(response, f)





knowldege_graph_triples = pickle.load(open("patient_similarity_results.pkl", "rb"))
print(knowldege_graph_triples)
patient_similarity(knowldege_graph_triples)