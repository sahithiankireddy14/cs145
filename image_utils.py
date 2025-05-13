import logging
import torch
import openai
import os
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

app = Flask(__name__)

# Setup logging
device = "cuda"
dtype = torch.float16


# Load ChexAgent model for captioning
logging.info("Loading model... This might take a while.")
processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
generation_config.max_length = 4096  
model = AutoModelForCausalLM.from_pretrained("StanfordAIMI/CheXagent-8b", torch_dtype=dtype, trust_remote_code=True)
model = model.to(device)
logging.info("Model loaded successfully!")

 
directory = ""  
images = []

for filename in os.listdir(directory):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):  
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path).convert("RGB")
        images.append(image)



# Step 1: Diagnosis image and if there's abnormality tell me where it is
prompt = "Describe the findings in this brain scan. If there abnormalities, also tell me in what region threre are abnormalities. "
inputs = processor(images=images, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt").to(device=device, dtype=dtype)
output = model.generate(**inputs, generation_config=generation_config)[0]
result = processor.tokenizer.decode(output, skip_special_tokens=True)



# Step 2: from these captions, give me list of if thre's abnoriality and if so then another list of regions 
# SECOND CALL: Use LLM to extract abnormality presence and regions
followup_prompt = f"""
You are a medical information extractor.

Given the following brain scan captions:
{"/n".join(result)}

For each caption, identify:
1. Whether an abnormality is present (Yes/No)
2. If yes, the region(s) involved

Return your output as a list of subject–predicate–object triples, in this format:
- If abnormal: (MRI, is abnormal, [region list])
- If no abnormality: (MRI, is abnormal, none)

Output only the list of triples, no extra text.
""" 



# GPT4o followup call 
def query_llm(prompt: str, model="gpt-4") -> str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are an expert in mental health knowledge graph construction."},
                  {"role": "user", "content": followup_prompt}]
    )
    return response.choices[0].message.content.strip()



triples = query_llm["choices"][0]["message"]["content"]
print(triples)




