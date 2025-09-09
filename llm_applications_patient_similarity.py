import openai
import os
import pickle 
import json
import re
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import time


openai.api_key = os.getenv("OPENAI_API_KEY")

def query_llm(prompt: str, model="gpt-4o-2024-08-06") -> str:
    # model="gpt-4o"
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
        patient_number = None
        patient_list = []
        master_triple_string = ""
        patient_count = 0

        for triple in knowldege_graph_triples:
            s, p, o = triple[0], triple[1], triple[2]

            if (p == "has_admission" or p =="has_gender" or p == "diagnosed_with") and (patient_number != s or (isinstance(s, str) and (str(patient_number) != s.split("_")[-1]))):
                # Dump previous patient's data
                if len(patient_list) > 0:
                    master_triple_string += "\n".join(patient_list) + "\n"
                    patient_list = []

                # Start new patient section
                master_triple_string += f"\nPatient {s}\n"
                patient_count += 1

                if isinstance(s, str) and "_" in s:
                    patient_number = int(s.split("_")[-1])
                else:
                    patient_number = int(s)


            patient_list.append(f"{s} {p} {o}")

        # Dump any remaining triples
        if patient_list:
            master_triple_string += "\n".join(patient_list) + "\n"

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

def _candidate_id(prefix, patient_id):
    return f"{prefix}{patient_id[1:]}"

def _pair_block(anchor, candidate, patients_info, degree_map=None, graph_map=None):
    a_txt = patients_info.get(anchor, "")
    c_txt = patients_info.get(candidate, "")

    # IMPORTANT: Graph degree information passed into LLM call
    a_deg = degree_map.get(anchor, {}) if degree_map else {}
    c_deg = degree_map.get(candidate, {}) if degree_map else {}

    # IMPORTANT: Graph Jaccard information passed into LLM call
    jac = {}
    if graph_map and anchor in graph_map:
        jac = graph_map[anchor].get(candidate, {}) or {}

    return f"""
        [PAIR]
        anchor_pid: {anchor}
        candidate_pid: {candidate}

        [anchor_degree_stats]
        {json.dumps(a_deg, ensure_ascii=False)}

        [candidate_degree_stats]
        {json.dumps(c_deg, ensure_ascii=False)}

        [graph_jaccard_scores]
        {json.dumps(jac, ensure_ascii=False)}

        [anchor_triples_text]
        {a_txt}

        [candidate_triples_text]
        {c_txt}
        """

def patient_similarity_v2(patients_info, patient_id, degree_map=None, graph_map=None, with_graph=False):
    synth_ids = [_candidate_id(p, patient_id) for p in ("2","3","4","5","6")]

    base_prompt = """
        You are analyzing clinical-patient similarity using patient profiles.
        ONLY use the provided blocks. Don't invent items that aren’t present.

        Scoring categories (0..1 each):
        - prescriptions
        - procedures
        - symptoms
        - other_factors (orders/“misc”)
        Overall = the mean of the 4 category scores.

        Rules:
        - If a category is missing or has no overlap, give that category a score of 0.
        - Use provided Jaccard scores (if present) as hints for overlap strength,
        but ground your final category scores in the actual text blocks.
        - Use degree stats to sanity-check intensity mismatches (can moderate scores).
        - Keep outputs concise and deterministic.

        Output strictly valid JSON array; no markdown fences and no comments. 
        """

    # Combine graph information across patients
    pair_sections = []
    for sid in synth_ids:
        pair_sections.append(_pair_block(patient_id, sid, patients_info, degree_map, graph_map))

    # Final LLM prompt
    similarity_prompt = f"""
        {base_prompt}

        Analyze the following pairs:
        {''.join(pair_sections)}

        Return JSON shaped exactly like:
        [
        {{
            "patient_1": "{patient_id}",
            "patient_2": "<candidate_pid>",
            "sub_similarities": {{
            "prescription_similarity": 0.0,
            "procedure_similarity": 0.0,
            "symptom_similarity": 0.0,
            "other_factors_similarity": 0.0
            }},
            "overall_similarity": 0.0,
            "key_contributors": []
        }}
        ]
        """
    # print(similarity_prompt)

    response = query_llm(similarity_prompt)

    m = re.search(r"```json\s*(.*?)\s*```", response, flags=re.DOTALL)
    txt = m.group(1) if m else response.strip()

    data = json.loads(txt)
    with open("patient_similarity_output.json", "a") as f:
        json.dump(data, f)
        f.write("\n")
    return data

def diagnosis_gt_similarity_v2(patients_diagnosis, patient_id):
    patient_diagnosis = patients_diagnosis[patient_id]
    synth1 = "2"+patient_id[1:]
    synth1_info = patients_diagnosis[synth1]

    synth2 = "3"+patient_id[1:]
    synth2_info = patients_diagnosis[synth2]

    synth3 = "4"+patient_id[1:]
    synth3_info = patients_diagnosis[synth3]

    synth4 = "5"+patient_id[1:]
    synth4_info = patients_diagnosis[synth4]

    synth5 = "6"+patient_id[1:]
    synth5_info = patients_diagnosis[synth5]
    
    diagnosis_prompt = f""" 
        Given the above knowledge graph relation triples, perform a comprehensive similarity analysis between:
        1. Patient {patient_id}: {patient_diagnosis} and Patient {synth1}: {synth1_info} 
        
        2. Patient {patient_id}: {patient_diagnosis} and Patient {synth2}: {synth2_info}
        
        3. Patient {patient_id}: {patient_diagnosis} and Patient {synth3}: {synth3_info}

        4. Patient {patient_id}: {patient_diagnosis} and Patient {synth4}: {synth4_info}

        5. Patient {patient_id}: {patient_diagnosis} and Patient {synth5}: {synth5_info}

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
    
    
def get_correlation(pred_list, gt_list):
    print("Getting Correlation")

    corr, p_value = spearmanr(pred_list, gt_list)
    print(f"Spearman correlation: {corr:.4f}, p-value: {p_value:.4g}")
    print(f"P-value: {p_value:.4g}")

    corr, p_value = pearsonr(pred_list, gt_list)
    print(f"Pearson correlation: {corr:.4f}")
    print(f"P-value: {p_value:.4g}")


def compare_similarities(sim_reports, gt_reports):
    gt_lookup = {}

    for gt in gt_reports:
        if "patient_1" in gt and "patient_2" in gt:
            gt_lookup[(gt["patient_1"], gt["patient_2"])] = gt["overall_similarity"]

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
     
     print("Patient numbers: ", patient_num)

     patient_sim_all = []
     gt_sim_all = []

     for patient_id in patient_info.keys():
         if patient_id[0] == "1":
            patient_sim = patient_similarity_v2(patient_info, patient_id)
            patient_sim_all.extend(patient_sim)
            gt_sim = diagnosis_gt_similarity_v2(patient_diagnosis, patient_id)
            gt_sim_all.extend(gt_sim)

     final_result = compare_similarities(patient_sim_all, gt_sim_all)

     return final_result
   

def main():
    knowldege_graph_data= pickle.load(open("knowledege_graph_triples_new_version.pkl", "rb"))
    formatted_relation_triples = create_patient_triples_string(knowldege_graph_data)
    def split_into_chunks(text, max_chars=5000):
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    def reformat_in_chunks(text, max_chars=18000):
        chunks = split_into_chunks(text, max_chars=max_chars)
        reformatted = []
        for i, c in enumerate(chunks):
            out = reformat(c)
            if out is None:
                out = c
            reformatted.append(out)
            time.sleep(1)
        return "\n".join(reformatted)

    formatted_relation_triples = reformat_in_chunks(formatted_relation_triples, max_chars=5000)

    with open("llm_applications_data/triples_string_full.txt", 'w') as f:
        f.write(formatted_relation_triples)

    # the structured patient triples string 
    with open('llm_applications_data/triples_string_full.txt', 'r') as file:
        formatted_relation_triples = file.read()  

    final = evaluate(formatted_relation_triples)
    print(final)