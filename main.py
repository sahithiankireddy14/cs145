import pandas as pd 
import numpy as np 
import os 
import gzip


path = "/Users/sahithi/Desktop/Research/physionet.org/files/mimiciv/3.1/hosp"

dfs = {}

for filename in os.listdir(path):
    filepath = os.path.join(path, filename)
    print(filepath)
    try:
        if filename.endswith('.csv.gz'):
            
            with gzip.open(filepath, 'rt') as f: 
                df = pd.read_csv(f)
                dfs[filename.replace('.csv.gz', '')] = df
        elif filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            dfs[filename.replace('.csv', '')] = df
    except Exception as e:
         print(f"Failed to load {filename}: {e}")

