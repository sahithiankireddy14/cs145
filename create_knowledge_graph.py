import os
import json
import pickle
import networkx as nx
from graphviz import Digraph


class KnowledgeGraph:
    def __init__(self, input_data=None, input_json_path=None, output_pkl="graph_data/knowledge_graph_triples.pkl"):
        """
        Initialize KnowledgeGraph.

        Args:
            input_data (dict): dictionary from PatientInfoBuilder
            input_json_path (str): path to JSON file
            output_pkl (str): where to save the triples
        """
        self.output_pkl = output_pkl
        self.graph_list = []

        if input_data is not None:
            self.data = input_data
        elif input_json_path is not None:
            with open(input_json_path, "r") as f:
                self.data = json.load(f)
        else:
            raise ValueError("Must provide either input_data (dict) or input_json_path (str).")

    def build_graph(self):
        for patient_id, patient_info in self.data.items():
            patient_node = f"Patient_{patient_id}"

            # ---------------- Patient demographics ----------------
            if "info" in patient_info:
                for field in ["patient_age", "race", "gender", "age_of_death"]:
                    if field in patient_info["info"]:
                        self.graph_list.append((patient_node, f"has_{field}", str(patient_info["info"][field])))

            # ---------------- Admissions ----------------
            if "admissions" in patient_info:
                self.process_admissions(patient_node, patient_info["admissions"])

        # Save triples to pickle
        with open(self.output_pkl, "wb") as f:
            pickle.dump(self.graph_list, f)

        return self.graph_list

    def process_admissions(self, patient_node, admissions_dict):
        for admission_id, admission_info in admissions_dict.items():
            admission_node = f"Admission_{admission_id}"
            self.graph_list.append((patient_node, "has_admission", admission_node))

            # --- Diagnoses
            if "diagnoses" in admission_info:
                for diag in admission_info["diagnoses"]:
                    self.graph_list.append((admission_node, "hasDiagnosis", diag))

            # --- Symptoms
            if "symptoms" in admission_info:
                for sym in admission_info["symptoms"]:
                    self.graph_list.append((admission_node, "hasSymptom", sym))

            # --- Procedures
            if "procedures" in admission_info:
                for proc in admission_info["procedures"]:
                    self.graph_list.append((admission_node, "hasProcedure", proc))

            # --- Prescriptions
            if "prescriptions" in admission_info:
                for drug in admission_info["prescriptions"]:
                    self.graph_list.append((admission_node, "hasPrescription", drug))

            # --- Providers
            if "providers_involved" in admission_info:
                for prov in admission_info["providers_involved"]:
                    self.graph_list.append((admission_node, "involves_provider", prov))  # keep provider ID as-is

            # --- EMAR
            if "emar" in admission_info:
                for e in admission_info["emar"]:
                    emar_node = f"Emar_{e['emar_id']}"
                    self.graph_list.append((admission_node, "hasEmarEvent", emar_node))
                    for k, v in e.items():
                        if k != "emar_id":
                            self.graph_list.append((emar_node, f"has_{k}", str(v)))

            # --- HCPCS events
            if "hcpcs_events" in admission_info:
                for hcpcs in admission_info["hcpcs_events"]:
                    hcpcs_node = f"HCPCS_{hcpcs['sequence_number']}"
                    self.graph_list.append((admission_node, "hasHCPCS", hcpcs_node))
                    for k, v in hcpcs.items():
                        self.graph_list.append((hcpcs_node, f"has_{k}", str(v)))

            # --- Diagnosis related groups (DRG)
            if "diagnosis_related_group" in admission_info:
                for drg in admission_info["diagnosis_related_group"]:
                    drg_node = f"DRG_{drg['type']}"
                    self.graph_list.append((admission_node, "hasDRG", drg_node))
                    for k, v in drg.items():
                        self.graph_list.append((drg_node, f"has_{k}", str(v)))

            # --- Physician order entry (POE)
            if "physician_order_entry" in admission_info:
                for order_dict in admission_info["physician_order_entry"]:
                    for order_id, order in order_dict.items():
                        order_node = f"Physician_Order_{order_id}"
                        self.graph_list.append((admission_node, "hasPhysicianOrder", order_node))
                        for k, v in order.items():
                            if k == "ordered_by":
                                self.graph_list.append((order_node, "ordered_by", str(v)))
                            else:
                                self.graph_list.append((order_node, f"has_{k}", str(v)))

            # --- Pharmacy
            if "pharmacy" in admission_info:
              for pharm_dict in admission_info["pharmacy"]:
                for pharm_id, pharm in pharm_dict.items():
                    pharm_node = f"Pharmacy_{pharm_id}"
                    self.graph_list.append((admission_node, "hasPharmacyOrder", pharm_node))
                    for k, v in pharm.items():
                        self.graph_list.append((pharm_node, f"has_{k}", str(v)))

    # ---------------- Utility functions ----------------
    def export_networkx(self):
        G = nx.DiGraph()
        for h, r, t in self.graph_list:
            G.add_edge(h, t, relation=r)
        return G

    def visualize(self, out_path="graph_output/knowledge_graph"):
        dot = Digraph()
        for h, r, t in self.graph_list:
            dot.edge(h, t, label=r)
        dot.render(out_path, format="pdf", cleanup=True)


if __name__ == "__main__":
    from patient_info_builder.patient_info_builder import PatientInfoBuilder


    builder = PatientInfoBuilder()
    results = builder.build_all_patients() 

    kg = KnowledgeGraph(input_data=results)
    triples = kg.build_graph()
    print(f"Built KG with {len(triples)} triples")
    G = kg.export_networkx()
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
