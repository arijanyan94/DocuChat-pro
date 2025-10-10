import os, pickle, orjson, numpy as np, faiss
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ART = "artifacts"

class Retriever:
	def __init__(self, art_dir: str = ART, emb_model="BAAI/bge-small-en-v1.5"):
		self.art_dir = art_dir
		# Load meta rows
		self.metas: List[Dict] = []
		with open(os.path.join(art_dir, "meta_rows.jsonl"), "rb") as f:
			for line in f:
				self.metas.append(orjson.loads(line))
		# Dense
		self.emb_model = SentenceTransformer(emb_model)
		self.index = faiss.read_index(os.path.join(art_dir, "faiss.index"))
		# BM25 (re-create from tokens for portability)
		with open(os.path.join(art_dir, "bm25_tokens.pkl"), "rb") as f:
			self.bm25_tokens = pickle.load(f)
		self.bm25 = BM25Okapi(self.bm25_tokens)

	def dense_search(self, query: str, k=20) -> List[Tuple[int, float]]:
		q = self.emb_model.encode([query], normalize_embeddings=True)
		D, I = self.index.search(np.asarray(q, dtype="float32"), k)
		return [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i != -1]

	def bm25_search(self, query: str, k=20) -> List[Tuple[int, float]]:
		scores = self.bm25.get_scores(query.split())
		idx = np.argpartition(-scores, min(k, len(scores)-1))[:k]
		idx = idx[np.argsort(-scores[idx])]
		return [(int(i), float(scores[i])) for i in idx]

	@staticmethod
	def rrf_fuse(d_hits: List[Tuple[int, float]], b_hits: [List[float]], k=10, k_rrf=60):
		# Reciprocal Rank Fusion over ranks (score component doesn’t need calibration)
		ranks: Dict[int, float] = {}
		for rank, (i, _) in enumerate(d_hits, start=1):
			ranks[i] = ranks.get(i, 0.0) + 1.0 / (k_rrf + rank)
		for rank, (i, _) in enumerate(b_hits, start=1):
			ranks[i] = ranks.get(i, 0.0) + 1.0 / (k_rrf + rank)
		fused = sorted(ranks.items(), key=lambda x: -x[1])[:k]
		return fused # list of (row_idx, fused_score)

	def hybrid(self, query: str, k_dense=20, k_bm25=20, k_final=8) -> List[Dict]:
		d = self.dense_search(query, k_dense)
		b = self.bm25_search(query, k_bm25)
		fused = self.rrf_fuse(d, b, k=k_final)

		out = []
		for row_idx, fscore in fused:
			m = self.metas[row_idx]
			out.append({
				"row": row_idx,
				"score": fscore,
				"chunk_id": m["chunk_id"],
				"doc_id": m["doc_id"],
				"page": m["page"],
				"source_path": m["source_path"],
				"snippet": self._short_snippet(row_idx)
			})
		return out

	def _short_snippet(self, row_idx: int, n=240) -> str:
		with open(os.path.join(self.art_dir, "meta_count.json"), "rb") as f:
			pass

		line = None
		with open(os.path.join(self.art_dir, "chunks.jsonl"), "rb") as f:
			for i, l in enumerate(f):
				if i == row_idx:
					line = l; break
		if line is None: return ""
		rec = orjson.loads(line)
		t = rec["text"].replace("\n", " ").strip()
		return (t[:n] + "...") if len(t) > n else t



















