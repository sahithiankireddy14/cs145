import openai
import re
from baseline_models import TFIDFBaseline, Node2VecBaseline, BERTTextBaseline, ControlledTripletEvaluator, set_seed, PSNFusionBaseline, SapBERTConceptBaseline, ClinPath 
from llm_applications import patient_similarity_v2

set_seed(42)

class Baselines:
    def __init__(self):
        self.diagnosis_dict = {}
        self.patient_info = {}

    def run_benchmarks(self):
        with open("Benchmark/new_benchmark.txt", 'r') as file:
            patient_profiles = file.read()

        patient_diagnosis = self.process_diagnosis(patient_profiles)
        self.process_patients(patient_profiles)

        tfidf = TFIDFBaseline(self.patient_info)
        n2v   = Node2VecBaseline(self.patient_info)
        clinpath = ClinPath(self.patient_info, patient_similarity_v2)

        # ClinPath but LLM was not augmented with graph information is clinpath_no_graph
        clinpath_no_graph = ClinPath(self.patient_info, patient_similarity_v2, with_graph=False)
        bert  = BERTTextBaseline(self.patient_info)
        sap   = SapBERTConceptBaseline(self.patient_info, patient_diagnosis_lines=patient_diagnosis)

        views = {
            "tfidf": tfidf,
            "sap":   sap,
            "n2v":   n2v,
            "bert":  bert,
        }
        evaluator = ControlledTripletEvaluator(self.diagnosis_dict)

        pids = list(self.patient_info.keys())
        print("TF-IDF:",        evaluator.evaluate(tfidf, pids, k_pool=20))
        print("Node2Vec:", evaluator.evaluate(n2v, pids, k_pool=20))
        print("ClinPath: ",  evaluator.evaluate(clinpath, pids, k_pool=20))
        print("ClinPathNoGraph: ",  evaluator.evaluate(clinpath_no_graph, pids, k_pool=20))
        print("BioClinicalBERT:", evaluator.evaluate(bert,  pids, k_pool=20))
        print("SapBERT concepts:", evaluator.evaluate(sap,   pids, k_pool=20))

        # PSN section
        psn_uniform = PSNFusionBaseline(views)  # uniform weights
        print("PSN (uniform):",   evaluator.evaluate(psn_uniform, pids, k_pool=20))
        learned_weights = psn_uniform.fit_weights(self.diagnosis_dict)
        psn_learned = PSNFusionBaseline(views, view_weights=learned_weights) #learned weights
        print("PSN (learned):",   evaluator.evaluate(psn_learned, pids, k_pool=20))

    def get_processed_info(self):
        return self.patient_info, self.diagnosis_dict
    
    def query_llm(self, prompt: str, model="gpt-4o-2024-08-06") -> str:
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
        
        
    def strip_json_comments(self, text):
        return re.sub(r"//.*", "", text)   
    
    def diagnosis_gt_similarity_v2(self, patients_diagnosis, patient_id):
        synth1 = "2"+patient_id[1:]
        synth2 = "3"+patient_id[1:]
        synth3 = "4"+patient_id[1:]
        synth4 = "5"+patient_id[1:]
        synth5 = "6"+patient_id[1:]
        comparison_dict = {synth1: 1.0, synth2: 0.75, synth3: 0.5, synth4: 0.25, synth5: 0.0}
        return patient_id, comparison_dict
            

    def process_diagnosis(self, formatted_relation_triples):
        diagnosis_keywords = re.compile(
            r"\b(DRG(?:s|\d+|_\d+)?|has_diagnosis_related_group|HCFA:|\bHCFA\b(?=\s*:)|diagnosis|severity|description|type|is_a)\b",
            re.IGNORECASE
        )
        
        lines = formatted_relation_triples.strip().split("\n")
        patient_diagnosis = {}
        patient_num = 0

        for line in lines:
            match = re.search(r"Patient (\d+)", line)
            if match:
                patient_num += 1
                patient_id = match.group(1)
            else:
                pass
            
            if diagnosis_keywords.search(line):
                if patient_id in patient_diagnosis:
                    patient_diagnosis[patient_id].append(line)
                else:
                    patient_diagnosis[patient_id] = [line]
        
        for patient_id in patient_diagnosis.keys():
            if patient_id[0] == "1":
                patient, patient_sim = self.diagnosis_gt_similarity_v2(patient_diagnosis, patient_id)
                self.diagnosis_dict[patient] = patient_sim
        return patient_diagnosis      

                
    def process_patients(self, patient_profiles):
        diagnosis_keywords = re.compile(
            r"\b(DRG(?:s|\d+|_\d+)?|HCFA:|\bHCFA\b(?=\s*:)|has_diagnosis_related_group|diagnosis|severity|description|type|is_a)\b",
            re.IGNORECASE
        )

        lines = patient_profiles.strip().split("\n")
        patient_info = {}
        patient_num = 0
        for line in lines:
            match = re.search(r"Patient (\d+)", line)
            if match:
                patient_num += 1
                patient_id = match.group(1)
            else:
                pass
            
            if not diagnosis_keywords.search(line):
                if patient_id in patient_info:
                    patient_info[patient_id].append(line)
                else:
                    patient_info[patient_id] = [line]

        print("Patient numbers: ", patient_num)
        # creates patient information dictionary which has patient_1: all their info
        self.patient_info = patient_info

base = Baselines()
base.run_benchmarks()