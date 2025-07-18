import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 
import openai
import os
import json
from patient_info_builder.patient_info_builder import PatientInfoBuilder
import ast
import pickle
from graphviz import Digraph
import json
import re
import pyvis
from pyvis.network import Network
import time

from tiktoken import encoding_for_model

class KnowledgeGraph:
    def __init__(self):
        data_base = "/Users/psehgal/Documents/physionet.org/files/mimiciv/3.1/hosp"
        # data_base = "/Users/sahithi/Desktop/Research/physionet.org/files/mimiciv/3.1/hosp"
        self.df = pd.read_csv(os.path.join(data_base, "patients.csv.gz"))
        self.graph_list = []
        openai.api_key = os.getenv("OPENAI_API_KEY")
        with open("patient_info_builder/output_description.txt", "r") as file:
            content = file.read()
        print("hi")
        self.dictionary_description = content
        json_path = os.path.join(os.path.dirname(__file__), "patient_info_builder", "output_description.json")
        with open(json_path, "r") as f:
            output_description = json.load(f)
        self.json = output_description
        print("hi2")

    def safe_parse_json_triples(self, text):
        try:
            match = re.search(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)
            raw = match.group(0) if match else text.strip()
            triple_list = json.loads(raw)

            if isinstance(triple_list, list) and all(isinstance(t, list) and len(t) == 3 for t in triple_list):
                ans = [tuple(t) for t in triple_list]
                return ans
            else:
                raise ValueError("Expected list of 3-element lists.")
        except Exception as e:
            print("Failed to parse GPT output:\n", text)
            return []

    def estimate_tokens(self, text, model):
        enc = encoding_for_model(model)
        return len(enc.encode(text))
    
    def get_triple(self, data, output_desc):
        time.sleep(0.876)
        prompt = f"""
        You are an expert creating a knowledge graph for the MIMIC-IV dataset.

        Given the following data: {data}
        And the following field descriptions: {output_desc}

        Output all of the relationships in the provided data above.

        ONLY output a JSON array of triples like:
        [["node1", "relation", "node2"], ...]

        Do not include any explanation or commentary. If there are no relationships, return [].
        """
        if self.estimate_tokens(prompt, model="gpt-4") > 8000:
            print("Skipping admission: prompt is too long")
            return []
        raw_output = self.query_llm(prompt)
        return self.safe_parse_json_triples(raw_output)
 
    
    def process_admissions(self, patient, patient_admissions_dict):
        for admission, info in patient_admissions_dict.items():
            self.graph_list.append((patient, "has_admission", admission))
            for col_name, data in info.items():
                # print("col_name: ", col_name)
                # labels = self.get_triple({admission: col_name}, None)
                # self.graph_list.extend(labels)
                output_desc = None
                if col_name in self.json["admission_number"]:
                    output_desc = self.json["admission_number"][col_name]
                    # print("desc name: ", output_desc)
                    labels = self.get_triple({admission: {col_name: data}}, output_desc)
                    # generally the keys should be 
                    self.graph_list.extend(labels)
            #break
    
    def process_info(self, patient, patient_info_dict):
        for col_name, info in patient_info_dict.items():
            # print("col_name: ", col_name)
            output_desc = None
            # print(self.json)
            # print(self.json["patient_number"])
            if col_name in self.json["patient_number"]:
                output_desc = self.json["patient_number"][col_name]
                # print("desc name: ", output_desc)
                labels = self.get_triple({patient: {col_name: info}}, output_desc)
                self.graph_list.extend(labels)

    def iter_patients(self):
        print("hi3")
        pi = PatientInfoBuilder()
        print("hi4")
        for index, patient in self.df.iterrows():
            print(index)
            if pd.isna(patient["subject_id"]):
                continue
            patient_info_dict, patient_admissions_dict = pi.patient_loop(patient)
            patient_id = patient["subject_id"]
            self.process_info(patient_id, patient_info_dict)
            # print("here3")
            # print("patient admissions dict: ", patient_admissions_dict)
            self.process_admissions(patient_id, patient_admissions_dict)
            # print("Graph list: ", self.graph_list)
            if index > 100:
                break
            #break
            # TODO: base path passed in as arg, fine for now 
                
    def query_llm(self, prompt: str, model="gpt-4") -> str:
            client = openai.OpenAI()  # Uses the API key in env or config
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in mental health knowledge graph construction."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
    

    def show_graph_pyvis(self):
        graph_list = self.graph_list
        with open("knowledege_graph_triples.pkl", "wb") as f:
                    pickle.dump(graph_list, f)
        net = Network(height="800px", width="100%", directed=True, notebook=False)
        G = nx.DiGraph()
        for source, label, target in graph_list:
            G.add_edge(str(source), str(target), label=label)
        for node in G.nodes:
            net.add_node(node, label=str(node))
        for u, v, d in G.edges(data=True):
            net.add_edge(u, v, label=d["label"])
        net.show("graph.html")

    def show_graph_graphviz(self):
        filename="knowledge_graph"
        graph_list = self.graph_list
        dot = Digraph(format='png') 
        dot.attr(rankdir='LR') 

        for source, label, target in graph_list:
            dot.node(str(source))
            dot.node(str(target))
            dot.edge(str(source), str(target), label=label)

        output_path = dot.render(filename, cleanup=True)
        print(f"Graph saved to: {output_path}")

    def make_graph(self):
        G = nx.DiGraph()
        # TODO: MultiDiGraph?
        for source, label, target in self.graph_list:
            G.add_edge(str(source), str(target), label=label)
        pos = nx.spectral_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True, node_size=500, font_size=8)
        edge_labels = nx.get_edge_attributes(G, 'label')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        for (n1, n2), label in edge_labels.items():
            x1, y1 = pos[n1]
            x2, y2 = pos[n2]
            x_mid, y_mid = (x1 + x2) / 2, (y1 + y2) / 2
            plt.text(
                x_mid, y_mid, label,
                fontsize=6,
                ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6)
            )
        # nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        # nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10)
        # nx.draw_networkx_labels(G, pos, font_size=8)
        # edge_labels = nx.get_edge_attributes(G, 'label')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        plt.title("Knowledge Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        # pos = nx.spring_layout(G) 
        # node_colors = [ 
        #     'lightblue' if G.nodes[n]['type'] == 'patient' 
        #     else 'lightgreen' if G.nodes[n]['type'] == 'data'
        #     else 'salmon' if G.nodes[n]['type'] == 'mri'
        #     else 'orange'
        #     for n in G.nodes
        # ]

        # nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=1500, font_size=10)
        # edge_labels = nx.get_edge_attributes(G, 'label')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        # plt.title("Patient Data Graph")
        # plt.show()

kg = KnowledgeGraph()
kg.iter_patients()
kg.show_graph_graphviz()

