import numpy 
import os 
import openai



openai.api_key = os.getenv("OPENAI_API_KEY")
def query_llm(prompt: str, model="gpt-4o") -> str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You have excellent medical knowledege are an expert in reading clinical notes."},
                    {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def extract_symptoms(clinical_note):
    prompt = f"""  
        You are a medical assistant reviewing a clinical note. Please extract and return a list of all relevant patient-reported symptoms mentioned in the note.

        Focus only on subjective experiences the patient reports (e.g., pain, discomfort, bleeding, confusion), not diagnoses, lab values, or physical exam findings.

        Ignore any symptoms the patient denies.

        Return the list in the following format, with each symptom starting with an asterisk (*) on a new line.

        Here is the Clinical Note:
        {clinical_note}

        ----------
        Sample Output Format:
        * Symptom 1  
        * Symptom 2  
        * Symptom 3
    """
  
    output = query_llm(prompt)
    symptoms = [line.strip("* ").strip() for line in output.strip().splitlines() if line.startswith("*")]
    return symptoms
  
        

