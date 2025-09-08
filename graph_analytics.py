import networkx as nx
from networkx.algorithms import community
import numpy as np
import pickle


class GraphAnalytics:
    def __init__(self, triples):
        self.G = nx.DiGraph()
        for h, r, t in triples:
            self.G.add_edge(h, t, relation=r)

    # -------------------- Patient Similarity -------------------- #
    def get_patient_attributes(self, patient_node):
        attrs = {"diagnoses": set(), "symptoms": set(), "procedures": set(),
                 "prescriptions": set(), "providers": set()}

        for _, adm, data in self.G.out_edges(patient_node, data=True):
            if data.get("relation") == "has_admission":
                for _, target, d in self.G.out_edges(adm, data=True):
                    rel = d.get("relation")
                    if rel == "hasDiagnosis": attrs["diagnoses"].add(target)
                    elif rel == "hasSymptom": attrs["symptoms"].add(target)
                    elif rel == "hasProcedure": attrs["procedures"].add(target)
                    elif rel == "hasPrescription": attrs["prescriptions"].add(target)
                    elif rel == "involves_provider": attrs["providers"].add(target)
        return attrs


    def compute_patient_similarity(self, pid1, pid2, use_):
        """Compute Jaccard similarity"""
        p1 = self.get_patient_attributes(f"Patient_{pid1}")
        p2 = self.get_patient_attributes(f"Patient_{pid2}")

        def jaccard(a, b): return len(a & b) / len(a | b) if (a | b) else 0

        result = {
            "patient_1": pid1,
            "patient_2": pid2,
            "diagnosis_similarity": jaccard(p1["diagnoses"], p2["diagnoses"]),
            "symptom_similarity": jaccard(p1["symptoms"], p2["symptoms"]),
            "procedure_similarity": jaccard(p1["procedures"], p2["procedures"]),
            "prescription_similarity": jaccard(p1["prescriptions"], p2["prescriptions"]),
            "provider_overlap": jaccard(p1["providers"], p2["providers"])
        }


        return result
    

    def detect_patient_communities(self):
        patients = [n for n in self.G.nodes if str(n).startswith("Patient_")]
        diag_edges = []
        for p in patients:
            for _, adm, d in self.G.out_edges(p, data=True):
                if d.get("relation") == "has_admission":
                    for _, diag, dd in self.G.out_edges(adm, data=True):
                        if dd.get("relation") == "hasDiagnosis":
                            diag_edges.append((p, diag))

        if not diag_edges:
            return {}

        B = nx.Graph()
        B.add_edges_from(diag_edges)
        comms = community.greedy_modularity_communities(B)
        result = {}
        for i, comm in enumerate(comms):
            for node in comm:
                if str(node).startswith("Patient_"):
                    result[node] = {"community": i}
        return result

    # -------------------- Provider Analytics -------------------- #
    def get_provider_stats(self, provider_id):
        provider_node = provider_id
        stats = {"provider_id": provider_node, "orders": []}
        for order in self.G.nodes:
            for _, target, data in self.G.out_edges(order, data=True):
                if data.get("relation") == "ordered_by" and target == provider_node:
                    stats["orders"].append(order)
        return stats
    
     
    def provider_centrality(self):
        providers = [n for n in self.G.nodes if str(n).startswith("Provider_")]
        orders = [n for n in self.G.nodes if str(n).startswith("Physician_Order_")]
        subG = self.G.subgraph(providers + orders).to_undirected()

        if subG.number_of_nodes() == 0:
            return {}

        degree = nx.degree_centrality(subG)
        betweenness = nx.betweenness_centrality(subG)
        try:
            eigenvector = nx.eigenvector_centrality(subG, max_iter=500)
        except nx.PowerIterationFailedConvergence:
            eigenvector = {n: 0 for n in subG.nodes}

        result = {}
        for p in providers:
            result[p] = {
                "degree": degree.get(p, 0),
                "betweenness": betweenness.get(p, 0),
                "eigenvector": eigenvector.get(p, 0)
            }
        return result

    def detect_provider_communities(self):
        providers = [n for n in self.G.nodes if str(n).startswith("Provider_")]
        orders = [n for n in self.G.nodes if str(n).startswith("Physician_Order_")]
        subG = self.G.subgraph(providers + orders).to_undirected()

        if subG.number_of_edges() == 0:
            return {}

        comms = community.greedy_modularity_communities(subG)
        result = {}
        for i, comm in enumerate(comms):
            for node in comm:
                if str(node).startswith("Provider_"):
                    result[node] = {"community": i}
        return result

    def detect_provider_anomalies(self):
        """Flag providers with abnormal centrality scores."""
        cent = self.provider_centrality()
        if not cent: return {}

        degrees = np.array([v["degree"] for v in cent.values()])
        mean, std = degrees.mean(), degrees.std() if degrees.std() > 0 else 1

        anomalies = []
        for p, stats in cent.items():
            z = (stats["degree"] - mean) / std
            if abs(z) > 3:  # outlier threshold
                anomalies.append({
                    "provider_id": p,
                    "degree": stats["degree"],
                    "zscore_degree": float(z),
                    "note": "Anomalous ordering pattern"
                })
        return {"provider_anomalies": anomalies}

    # -------------------- Global Graph -------------------- #
    def global_summary(self):
        return {
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges(),
            "density": nx.density(self.G)
        }



with open("knowledge_graph_triples_new_version_no_llm.pkl", "rb") as f:
    triples = pickle.load(f)

# Initialize analytics
ga = GraphAnalytics(triples)


print("Global Summary:")
print(ga.global_summary())

patient_attr = ga.get_patient_attributes("Patient_10000032")
print("Patient Attr:")
print(patient_attr)

sim = ga.compute_patient_similarity("10000032", "10000032")
print("\nPatient Similarity:")
print(sim)

patient_comms = ga.detect_patient_communities()
print("\nPatient Communities:")
print(patient_comms)


prov_stats = ga.get_provider_stats("Provider_P21AVM")
print("\nProvider Stats:")
print(prov_stats)


prov_cent = ga.provider_centrality()
print("\nProvider Centrality:")
print(prov_cent)


prov_comms = ga.detect_provider_communities()
print("\nProvider Communities:")
print(prov_comms)


prov_anoms = ga.detect_provider_anomalies()
print("\nProvider Anomalies:")
print(prov_anoms)
