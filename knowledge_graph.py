import openai
import json
import networkx as nx
from typing import List, Dict, Tuple

openai.api_key = 'your_openai_api_key'

def query_llm(prompt: str, model="gpt-4") -> str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are an expert in mental health knowledge graph construction."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
