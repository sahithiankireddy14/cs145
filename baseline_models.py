# baseline_models.py
import math
import json
import random
import re
import networkx as nx
import numpy as np, torch

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr, pearsonr

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cos(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)

    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v)/(nu * nv))

class TFIDFBaseline:
    def __init__(self, patient_info):
        self.pids = list(patient_info.keys())
        corpus = [" ".join(patient_info[pid]) for pid in self.pids] 

        self.vectorizer = TfidfVectorizer(min_df=1, max_df=0.8, ngram_range=(1, 2))
        self.X = self.vectorizer.fit_transform(corpus)

    def similarity(self, pid1, pid2):
        i, j = self.pids.index(pid1), self.pids.index(pid2)

        if i == j:
            return 0.0
        return float(cosine_similarity(self.X[i], self.X[j])[0, 0])

    def rank(self, pid, k):
        i = self.pids.index(pid)

        sims = cosine_similarity(self.X[i], self.X).ravel()

        sims[i] = -1
        k = min(k, len(self.pids) - 1)
        idx = np.argpartition(-sims, k-1)[:k]
        
        order = idx[np.argsort(-sims[idx])]
        return [(self.pids[j], float(sims[j])) for j in order]


class Node2VecBaseline:
    def __init__(self, patient_info,
                 vector_size=128, walk_len=40, num_walks=10, window=10):
        self.pids = list(patient_info.keys())

        def norm(s):
            return s.strip().lower().replace(" ", "_")

        G = nx.Graph()
        for pid, lines in patient_info.items():
            G.add_node(pid, bipartite=0)
            bag = set()
            for line in lines:
                for tok in [t for t in re_split(line)]:
                    if tok:
                        bag.add(norm(tok))
            for feat in bag:
                fn = f"feat::{feat}"
                G.add_node(fn, bipartite=1)
                G.add_edge(pid, fn)

        self.G = G
        walks = self._random_walks(G, walk_len=walk_len, num_walks=num_walks)
        model = Word2Vec(sentences=walks, vector_size=vector_size, window=window,
                 min_count=1, sg=1, workers=4, seed=42)
        self.emb = {n: model.wv[n] for n in G.nodes() if n in model.wv}

    def _random_walks(self, G, walk_len=40, num_walks=10):
        walks = []
        nodes = list(G.nodes())

        for _ in range(num_walks):
            random.shuffle(nodes)
            for n in nodes:
                walk = [n]

                while len(walk) < walk_len:
                    nbrs = list(G.neighbors(walk[-1]))
                    if not nbrs:
                        break

                    walk.append(random.choice(nbrs))
                walks.append(walk)
        return walks

    def similarity(self, pid1, pid2):
        if pid1 not in self.emb or pid2 not in self.emb:
            return 0.0
        return _cos(self.emb[pid1], self.emb[pid2])

    def rank(self, pid, k=10):
        out = []
        for other in self.pids:
            if other == pid:
                continue
            out.append((other, self.similarity(pid, other)))

        out.sort(key=lambda x: -x[1])
        return out[:k]

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def build_graph_map(graph_rows):
    graph_map = {}
    for row in graph_rows:
        anchor = row["pid"]
        comp = row.get("comparisons", {})
        graph_map[anchor] = {}
        for pid2, payload in comp.items():
            graph_map[anchor][pid2] = payload.get("jaccard_scores", {})
    return graph_map

def build_degree_map(degree_rows):
    degree_map = {row["pid"]: {k: v for k, v in row.items() if k != "pid"} for row in degree_rows}
    return degree_map

# baseline was created by author
class ClinPath:
    def __init__(self, patients_info, model_fn, anchors=None, synth_prefixes=("2","3","4","5","6"), with_graph=True,
        degree_stats_path="patient_degree_stats_v2.jsonl",
        graph_stats_path="patient_similarity_graph_benchmark_v2.jsonl"):

        # with_graph allows you to test with graph information in LLM call
        
        self.patients_info = patients_info
        self.pids = list(patients_info.keys())
        self.model_fn = model_fn
        self.synth_prefixes = synth_prefixes
        self.with_graph = with_graph

        self.patients_info = {pid: (
                "\n".join(lines).strip()
                if isinstance(lines, list)
                else str(lines).strip()
            )
            for pid, lines in patients_info.items()}

        if anchors is None:
            anchors = [pid for pid in self.pids if pid and pid[0] == "1"]
        self.anchors = anchors

        self.degree_map = None
        self.graph_map  = None
        if self.with_graph:
            try:
                # loads graph output json files
                degree_rows = load_jsonl(degree_stats_path)
                graph_rows  = load_jsonl(graph_stats_path)
                self.degree_map = build_degree_map(degree_rows)
                self.graph_map  = build_graph_map(graph_rows)

            except Exception as e:
                self.degree_map = None
                self.graph_map  = None

        # so you only have to compute this one time
        self._cache = {a: {} for a in self.anchors}
        self._precompute_all()

    def _precompute_all(self):
        for a in self.anchors:
            try:
                rows = self.model_fn(
                    self.patients_info,
                    a,
                    degree_map=self.degree_map,
                    graph_map=self.graph_map,
                    with_graph=self.with_graph,
                )
            except Exception:
                rows = []
            if isinstance(rows, list):
                for row in rows:
                    p2 = row.get("patient_2")
                    if not p2:
                        continue
                    try:
                        s = float(row.get("overall_similarity", 0.0))
                    except Exception:
                        s = 0.0
                    self._cache[a][p2] = s

            for pref in self.synth_prefixes:
                pid2 = pref + a[1:]
                self._cache[a].setdefault(pid2, self._cache[a].get(pid2, -1e9))

    def similarity(self, pid1, pid2):
        if pid1 == pid2:
            return 0.0
        
        d = self._cache.get(pid1, {})
        # missing is -1e9
        return float(d.get(pid2, -1e9))

    def rank(self, pid, k=10):
        d = self._cache.get(pid, {})
        scored = [(p2, float(s)) for p2, s in d.items() if p2 != pid]

        if len(scored) < k:
            have = set(d.keys())
            for other in self.pids:
                if other == pid or other in have:
                    continue
                scored.append((other, -1e9))
                if len(scored) >= k:
                    break

        scored.sort(key=lambda x: -x[1])
        return scored[:k]

class BERTTextBaseline:
    def __init__(self, patient_info,
                 model_name="emilyalsentzer/Bio_ClinicalBERT", max_length=512, device=None):
        
        self.pids = list(patient_info.keys())
        self.texts = [" ".join(patient_info[pid]) for pid in self.pids]

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModel.from_pretrained(model_name)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.mdl.to(self.device).eval()
        self.max_length = max_length
        self.vecs = self._embed_all(self.texts)

    def _embed_all(self, texts):
        vecs = []

        with torch.no_grad():
            for t in texts:
                enc = self.tok(t, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
                out = self.mdl(**enc).last_hidden_state  # [1, L, H]
                v = out.mean(dim=1).squeeze(0).detach().cpu().numpy()
                vecs.append(v)
        return np.stack(vecs, axis=0)

    def similarity(self, pid1, pid2):

        i, j = self.pids.index(pid1), self.pids.index(pid2)

        return _cos(self.vecs[i], self.vecs[j])

    def rank(self, pid, k=10):
        i = self.pids.index(pid)
        q = self.vecs[i]

        sims = np.array([_cos(q, v) for v in self.vecs])
        sims[i] = -1

        k = min(k, len(self.pids) - 1)
        idx = np.argpartition(-sims, k)[:k]
        order = idx[np.argsort(-sims[idx])]

        return [(self.pids[j], float(sims[j])) for j in order]
    

_SPLIT = re.compile(r"[\,;:\|\-\(\)\[\]\{\}\s]+")

def re_split(s):
    return [t for t in _SPLIT.split(s) if t]

# compute ndcg at 5
def ndcg_at_k(rels_in_pred_order, k=5):
    def dcg(scores):
        return sum((s / math.log2(i + 2)) for i, s in enumerate(scores[:k]))
    ideal = sorted(rels_in_pred_order, reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(rels_in_pred_order[:k]) / idcg) if idcg > 0 else 0.0

class ControlledTripletEvaluator:

    # evaluation function for all baselines

    def __init__(self, gt_per_query):
        self.gt_per_query = gt_per_query

    def evaluate(self, ranker, query_ids, k_pool=20):
        # counts Top-1 label == 1.0
        top1_hits_exact = [] 
         # counts Top-1 label >= 0.75
        top1_hits_ge_0p75 = []
        order_accs = []
        # NDCG@n where n = 5 in this example since there are 5 comparisons
        ndcgs_at_all = [] 
        rank_of_1pt0 = []

        all_gt, all_pred = [], []

        for q in query_ids:
            if q not in self.gt_per_query:
                continue

            labels = self.gt_per_query[q]      
            if not labels:
                continue

            wanted = set(labels.keys())
            retrieved = ranker.rank(q, k=k_pool) 
            filtered = [(pid2, s) for (pid2, s) in retrieved if pid2 in wanted]

            seen = {pid for pid, _ in filtered}
            for pid2 in wanted - seen:
                filtered.append((pid2, -1e9))

            filtered.sort(key=lambda x: -x[1])

            # Build relevance list in predicted order
            rels = [labels[pid2] for (pid2, _) in filtered]
            preds = [s for (_, s) in filtered]
            n = len(rels)


            # The section below has all of the performance metrics
            # Top-1 metrics
            top1_hits_exact.append(1.0 if n and rels[0] == 1.0 else 0.0)
            top1_hits_ge_0p75.append(1.0 if n and rels[0] >= 0.75 else 0.0)

            # Pairwise order accuracy over all 5 items
            correct = 0; total = 0
            for i in range(n):
                for j in range(i+1, n):
                    total += 1
                    if (rels[i] > rels[j] and preds[i] >= preds[j]) or \
                       (rels[i] < rels[j] and preds[i] <= preds[j]) or \
                       (rels[i] == rels[j] and abs(preds[i] - preds[j]) < 1e-12):
                        correct += 1
            order_accs.append(correct / total if total else 0.0)

            # NDCG metrics
            k_all = n
            ndcgs_at_all.append(ndcg_at_k(rels, k=k_all))

            # Mean rank of any 1.0 item (the lower, the better)
            try:
                r = next(i for i, (pid, _) in enumerate(filtered) if labels[pid] == 1.0) + 1
                rank_of_1pt0.append(float(r))

            except StopIteration:
                rank_of_1pt0.append(float('inf'))

            all_gt.extend(rels)
            all_pred.extend(preds)

        # Final output metrics
        res = {
            "Top1_is_1.0":          float(np.mean(top1_hits_exact))   if top1_hits_exact else 0.0,
            "Top1_is>=0.75":         float(np.mean(top1_hits_ge_0p75))  if top1_hits_ge_0p75 else 0.0,
            "Pairwise_Order_Acc":   float(np.mean(order_accs))        if order_accs else 0.0,
            "NDCG@all":             float(np.mean(ndcgs_at_all))      if ndcgs_at_all else 0.0,
            "Mean_Rank_of_1.0":     float(np.mean(rank_of_1pt0))      if rank_of_1pt0 else None,
        }

        # Correlation metrics

        if len(all_gt) >= 6:
            sp, sp_p = spearmanr(all_gt, all_pred)
            pr, pr_p = pearsonr(all_gt, all_pred)
            res.update({
                "Spearman": float(sp), "Spearman_p": float(sp_p),
                "Pearson":  float(pr), "Pearson_p":  float(pr_p),
                "pairs_evaluated": len(all_gt)
            })
        else:
            res["pairs_evaluated"] = len(all_gt)
        return res


class PSNFusionBaseline:
    """
    netDx-style patient similarity network (PSN) fusion:
    """
    def __init__(self, views, view_weights=None):
        assert len(views) >= 1, "Need at least one view"
        self.views = views

        if view_weights is None:
            w = 1.0 / len(views)
            self.view_weights = {k: w for k in views.keys()}

        else:
            s = sum(max(0.0, v) for v in view_weights.values())
            self.view_weights = {k: (max(0.0, view_weights.get(k, 0.0)) / (s if s>0 else 1.0))
                                 for k in views.keys()}


        any_view = next(iter(views.values()))
        self.pids = getattr(any_view, "pids", None)

    def similarity(self, pid1, pid2):
        s = 0.0
        for name, v in self.views.items():
            try:
                s += self.view_weights[name] * float(v.similarity(pid1, pid2))
            except Exception:
                s += 0.0
        return s

    def rank(self, pid, k=10):
        assert self.pids is not None, "At least one view must expose .pids"
        out = []

        for other in self.pids:
            if other == pid:
                continue
            out.append((other, self.similarity(pid, other)))

        out.sort(key=lambda x: -x[1])
        return out[:min(k, len(out))]

    # For learned weights
    def fit_weights(self, gt_per_query, candidate_ids=None,
                    l2=1e-3, nonneg=True):
        if self.pids is None:
            raise ValueError("Need .pids on at least one view to fit weights.")
        if candidate_ids is None:
            candidate_ids = self.pids

        view_names = list(self.views.keys())
        rows = []
        targets = []

        for q, labels in gt_per_query.items():
            if q not in candidate_ids:
                continue
            for pid2, y in labels.items():
                if pid2 not in candidate_ids:
                    continue

                phi = []
                for name in view_names:
                    try:
                        phi.append(float(self.views[name].similarity(q, pid2)))
                    except Exception:
                        phi.append(0.0)
                rows.append(phi)
                targets.append(float(y))

        if not rows:
            return  # nothing to fit

        X = np.asarray(rows, dtype=np.float64)        # [n, v]
        y = np.asarray(targets, dtype=np.float64)     # [n]

        #closed-form ridge
        XtX = X.T @ X
        V = XtX.shape[0]
        w = np.linalg.solve(XtX + l2 * np.eye(V), X.T @ y)

        if nonneg:
            w = np.clip(w, 0.0, None)

        # Normalize to sum = 1 for interpretability
        s = w.sum()
        if s <= 0:
            w = np.ones_like(w)/len(w)
        else:
            w = w/s

        self.view_weights = {name: float(w[i]) for i, name in enumerate(view_names)}
        return self.view_weights
    
_ICD = re.compile(r"\b([A-Z]\d{1,2}[A-Z0-9\.]*)\b", re.I)

def _extract_icd_codes_from_lines(lines):
    bag = set()

    for ln in lines:
        for c in _ICD.findall(ln):
            bag.add(c.upper())
    return sorted(bag) if bag else []

class SapBERTConceptBaseline:
    def __init__(self,patient_info, patient_diagnosis_lines=None,
                 model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token",
                 term_lookup= None, max_length=32, device=None, batch_size=64):
        
        self.pids = list(patient_info.keys())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.max_length = max_length
        self.batch_size = batch_size

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModel.from_pretrained(model_name).to(self.device).eval()

        # Build term lists per patient
        self.patient_terms = {}
        vocab_terms = []

        for pid in self.pids:
            if patient_diagnosis_lines and pid in patient_diagnosis_lines:
                codes = _extract_icd_codes_from_lines(patient_diagnosis_lines[pid])
            else:
                codes = _extract_icd_codes_from_lines(patient_info[pid])


            if term_lookup:
                terms = [term_lookup.get(c, c) for c in codes] or ["UNKNOWN_TOKEN"]
            else:
                terms = codes or ["UNKNOWN_TOKEN"]

            self.patient_terms[pid] = terms
            vocab_terms.extend(terms)

        uniq_terms = sorted(set(vocab_terms))
        term_vecs = self._embed_texts_batch(uniq_terms) 

        self.term2vec = {t: term_vecs[i] for i, t in enumerate(uniq_terms)}

        # Patient vectors - mean of their terms - L2 Norm
        reps = []
        for pid in self.pids:
            V = np.stack([self.term2vec[t] for t in self.patient_terms[pid]], axis=0)  
            v = V.mean(axis=0)
            n = np.linalg.norm(v) + 1e-12
            reps.append(v / n)

        self.mat = np.stack(reps, axis=0) 

    def _embed_texts_batch(self, texts):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                enc = self.tok(batch, return_tensors="pt", truncation=True,
                               padding=True, max_length=self.max_length).to(self.device)
                
                # mean-token pooling
                hs = self.mdl(**enc).last_hidden_state  
                v = hs.mean(dim=1)                     
                v = torch.nn.functional.normalize(v, p=2, dim=1)
                out.append(v.cpu().numpy())

        return np.vstack(out) if out else np.zeros((0, 768), dtype=np.float32)

    def similarity(self, pid1, pid2):
        i, j = self.pids.index(pid1), self.pids.index(pid2)
        if i == j: return 0.0
        return float(np.dot(self.mat[i], self.mat[j]))

    def rank(self, pid, k=10):
        i = self.pids.index(pid)
        sims = self.mat @ self.mat[i]
        sims[i] = -1

        k = min(k, len(self.pids) - 1)
        idx = np.argpartition(-sims, k)[:k]
        order = idx[np.argsort(-sims[idx])]

        return [(self.pids[j], float(sims[j])) for j in order]