import pickle
import json
import re
import os
from collections import defaultdict
from typing import Dict

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ProviderBehaviorAnalysis:
    def __init__(self, pickle_path: str):
        with open(pickle_path, "rb") as f:
            self.triples = pickle.load(f)

        self.providers_info = self.extract_provider_profiles()

    def extract_provider_profiles(self) -> Dict[str, Dict]:
        profiles = defaultdict(lambda: {
            "medications": set(),
            "procedures": set(),
            "labs": set(),
            "orders":set()
            
        })

        for s, p, o in self.triples:
            if p == "ordered_by":
                provider_id = o
                order_id = s
                profiles[provider_id]["orders"].add(order_id)

            elif p == "involves_provider":
                provider_id = o
                profiles[provider_id]["orders"].add(s)

            elif p == "has_medication":
                # find the provider(s) linked to this order
                for prov in profiles:
                    if s in profiles[prov]["orders"]:
                        profiles[prov]["medications"].add(o)

            elif p == "has_procedure":
                for prov in profiles:
                    if s in profiles[prov]["orders"]:
                        profiles[prov]["procedures"].add(o)

            elif p == "has_lab":
                for prov in profiles:
                    if s in profiles[prov]["orders"]:
                        profiles[prov]["labs"].add(o)

        # Convert sets to lists for JSON serialization
        return {k: {kk: list(vv) if isinstance(vv, set) else vv for kk, vv in v.items()}
                for k, v in profiles.items()}

    def query_llm(self, prompt: str, model="gpt-4o"):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in provider anomaly detection in clinical knowledge graphs."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()

    def analyze_providers(self, save_path="llm_applications_data/provider_behavior_results.json"):
        providers_json = json.dumps(self.providers_info, indent=2)
        prompt = f"""
        You are analyzing provider behavior using a clinical knowledge graph.

        Here is provider data (JSON):
        {providers_json}

        Tasl
        1. Perform pairwise comparisons of providers to evaluate similarities in their ordering patterns.
        2. Identify key factors that make each provider’s behavior unique relative to others.
        3. Flag providers as outliers if their ordering patterns deviate significantly from the broader peer group. Return this info

        Output strictly in JSON, following this schema. STRICT JSON FORMAT NO EXTRA TEXT
        ``` json
        {{
    
        "outliers": {{
            "providers": [<list of provider_ids>],
            "reasons": [<list of strings>]
        }}
        }}
        """

        response = self.query_llm(prompt)
        match = re.search(r"```json(.*?)```", response, re.DOTALL)
        result = response if not match else match.group(1)

        try:
            data = json.loads(result)
        except Exception:
            raise ValueError("LLM did not return valid JSON:\n" + response)

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        return data




def run_benchmark(triples_path):
    detector = ProviderBehaviorAnalysis(triples_path)
    print(detector.triples)
    print(detector.providers_info)
    print(detector.analyze_providers())

if __name__ == "__main__":
    run_benchmark(
        "llm_applications_data/provider_behavior_benchmark_triples.pkl"
    )