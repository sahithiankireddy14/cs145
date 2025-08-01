import logging
import torch
import openai
import os
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests




openai.api_key = os.getenv("OPENAI_API_KEY")

def query_llm(prompt: str, system_prompt:str, model="gpt-4") -> str:

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


