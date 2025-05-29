import logging
import torch
import openai
import os
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import pickle 


openai.api_key = os.getenv("OPENAI_API_KEY")



        
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
    return master_triple_string

 
def patient_similarity(knowldege_graph_triples):

    triples = create_patient_triples_string([knowldege_graph_triples, knowldege_graph_triples])
    print(triples)
    base_prompt = """

        I am modeling clinical interactions between patients, providers, drugs, prescriptions and more. 
        To do so, I am using a knowledge graph. This knowledge graph will then be used for various downstream 
        applications. Here are all the relation triples part of the knowledge graph


        {triples}

     """
    base_prompt = base_prompt.format(triples = knowldege_graph_triples)
    prompt = """ 
            
            Given the above knowledge graph triples, perform a comprehensive similarity analysis of the patients. 
            Your goal is to identify how patients are similar based on key medical attributes such as diagnoses, 
            prescriptions, medical procedures, symptoms, and other relevant clinical factors. Then attribute a final 
            simiarity score for each set of patients. 

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
                "symptom_similarity": 0.8,
                "procedure_similarity": 0.7
                },
                "key_contributors": ["diabetes diagnosis", "shared insulin prescription"]
            },
            ...
            ]
                    
        """


    #print(base_prompt + "\n" + prompt)
    #query_llm(base_prompt + "\n" + prompt)





knowldege_graph_triples = pickle.load(open("patient_similarity_results.pkl", "rb"))

    
patient_similarity(knowldege_graph_triples)