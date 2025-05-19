import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 
import openai

graph_list = []
csv_file = "patient_data.csv"
df = pd.read_csv(csv_file)

openai.api_key = 'your_openai_api_key'

def query_llm(prompt: str, model="gpt-4") -> str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are an expert in mental health knowledge graph construction."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

for _, row in df.iterrows:
    patient = row['patient']
    for col_name, data in row.items():
        if col_name != "patient":
            label = query_llm(f"""Assuming you are creating a graph with nodes as patients connecting to nodes with data
                    about the patient. Given the following {col_name} representing the type of data in the node,
                    output a short string of what I should label the edge connecting the two nodes together
                        """)
            graph_list.append((patient, label, data))

graph_list = [
    ("patient1", "has_symptom", "depression"),
    ("patient1", "has_mri", "region1"),
    ("patient1", "has_mri", "region2")
    ("patient2", "has_symptom", "anxiety"),
]

G = nx.Graph()
for ele in graph_list:
    patient, label, data = ele
    G.add_node(patient, type="patient")
    if label != "has_mri":
        G.add_node(data, type="data")
        G.add_edge(patient, data, label=label)
    else:
        G.add_node(patient + "_mri_scan", type="mri")
        G.add_edge(patient, patient + "_mri_scan", label=label) 
        G.add_node(data, type="mri_data")
        G.add_edge(patient + "_mri_scan", data, label=label+"_data")

pos = nx.spring_layout(G) 
node_colors = [ 
    'lightblue' if G.nodes[n]['type'] == 'patient' 
    else 'lightgreen' if G.nodes[n]['type'] == 'data'
    else 'salmon' if G.nodes[n]['type'] == 'mri'
    else 'orange'
    for n in G.nodes
]

nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=1500, font_size=10)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Patient Data Graph")
plt.show()