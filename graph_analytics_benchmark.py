import json
import networkx as nx
import re

def patient_json_to_triples(data):
    triples = []
    for patient in data["patients"]:
        pid = f"Patient_{patient['patient_id']}"

        # demographics
        triples.append((pid, "has_gender", patient["gender"]))
        triples.append((pid, "has_age", str(patient["age"])))
        triples.append((pid, "has_race", patient["race"]))

        for adm in patient["admissions"]:
            adm_node = f"Admission_{adm['admission_id']}"
            triples.append((pid, "has_admission", adm_node))

            # DRGs
            for k, v in adm["drgs"].items():
                triples.append((adm_node, f"hasDRG_{k}", v))

            # Medications
            for med in adm["medications"]:
                triples.append((adm_node, "hasMedication", med))

            # Orders
            for order in adm["orders"]:
                order_node = f"Order_{order['name']}"
                triples.append((adm_node, "hasOrder", order_node))
                if "route" in order:
                    triples.append((order_node, "route", order["route"]))
                if "notes" in order:
                     triples.append((order_node, "notes", order["notes"]))

            # Procedures
            for proc in adm["procedures"]:
                triples.append((adm_node, "hasProcedure", proc))
    return triples

# ---------------- Build Graph ---------------- #
def build_graph(triples):
    G = nx.DiGraph()
    for h, r, t in triples:
        G.add_edge(h, t, relation=r)
    return G



# --------------- Helper Functions --------------

def normalize_string(s: str) -> str:
    return (s or "").strip().lower()


def tokenize(s: str) -> set:
    return set(re.findall(r"\w+", normalize_string(s)))

# ---------------- Patient Analysis / Similarity Functions ---------------- #
def get_patient_attributes(G, pid):
    node = f"Patient_{pid}"
    attrs = {"drgs": [], "medications": [], "procedures": [], "orders": []}

    for _, adm, data in G.out_edges(node, data=True):
        if data.get("relation") == "has_admission":
            for _, target, d in G.out_edges(adm, data=True):
                rel = d.get("relation")
                print(target)
                print(tokenize(target))
                if rel.startswith("hasDRG"):
                    attrs["drgs"].append(tokenize(target))
                elif rel == "hasMedication":
                    attrs["medications"].append(tokenize(target))
                elif rel == "hasProcedure":
                    attrs["procedures"].append(tokenize(target))
                elif rel == "hasOrder":
                    attrs["orders"].append(tokenize(target))

    return attrs


def compute_jaccard(G, pid1, pid2):
    a1, a2 = get_patient_attributes(G, pid1), get_patient_attributes(G, pid2)

    def token_jaccard(list1, list2):
        set1 = set().union(*list1) if list1 else set()
        set2 = set().union(*list2) if list2 else set()
        return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0


    return {
        "drg_similarity": token_jaccard(a1["drgs"], a2["drgs"]),
        "medication_similarity": token_jaccard(a1["medications"], a2["medications"]),
        "procedure_similarity": token_jaccard(a1["procedures"], a2["procedures"]),
        "order_similarity": token_jaccard(a1["orders"], a2["orders"])
    }


def compute_patient_summary(G, pid):
    node = f"Patient_{pid}"
    if node not in G:
        return None

    admissions = []
    meds, procs, orders = set(), set(), set()

    # Traverse admissions
    for _, adm, data in G.out_edges(node, data=True):
        if data.get("relation") == "has_admission":
            admissions.append(adm)

            for _, target, d in G.out_edges(adm, data=True):
                rel = d.get("relation")
                if rel == "hasMedication":
                    meds.add(target)
                elif rel == "hasProcedure":
                    procs.add(target)
                elif rel == "hasOrder":
                    orders.add(target)

    return {
        "pid": pid,
        "num_admissions": len(admissions),
        "avg_admission_out_degree": (
            sum(G.out_degree(adm) for adm in admissions) / len(admissions)
            if admissions else 0
        ),
        "num_medications": len(meds),
        "num_procedures": len(procs),
        "num_orders": len(orders),
    }






def dump_all_patient_degree_stats(G, output_file="patient_degree_stats_v2.jsonl"):
    with open(output_file, "w") as f:
        for node in G.nodes():
            if node.startswith("Patient_"):
                pid = node.split("_", 1)[1]
                stats = compute_patient_summary(G, pid)
                if stats:
                    f.write(json.dumps(stats) + "\n")



    
def run_benchmark(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)
        pids = []
        for item in data["patients"]:
            pids.append(item["patient_id"])


    triples = patient_json_to_triples(data)
    G = build_graph(triples)

    with open(output_file, "w") as f:
        for i in range(0, len(pids), 6):
            print("in loop")
            primary = pids[i]
            comparisons = [pids[i+1], pids[i+2], pids[i+3], pids[i+4], pids[i+5]]
            print(comparisons)
            comp_dict = {}
            for secondary in comparisons:
                jaccard_scores = compute_jaccard(G, primary, secondary)
                comp_dict[secondary] = {"jaccard_scores": jaccard_scores}
            print(comp_dict)
            record = {"pid": primary, "comparisons": comp_dict}
            f.write(json.dumps(record) + "\n")

    dump_all_patient_degree_stats(G)

   


run_benchmark("benchmark_new_v2.json", "patient_similarity_graph_benchmark_v2.jsonl")
