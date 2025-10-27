import os, pickle, orjson, numpy as np, faiss, time
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from backend.rag.rerank import Reranker

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
		self._reranker: Optional[Reranker] = None

	def _ensure_reranker(self):
		if self._reranker is None:
			# self._reranker = Reranker("BAAI/bge-reranker-base")
			self._reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

	def _get_text_by_row(self, row_idx: int) -> str:
		with open(os.path.join(self.art_dir, "chunks.jsonl"), "rb") as f:
			for i, l in enumerate(f):
				if i == row_idx:
					return orjson.loads(l)["text"]
		return ""

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

	def hybrid(self, query: str, k_dense=20, k_bm25=20, k_final=8,
				rerank: bool = False, top_m: int = 50) -> List[Dict]:
		d = self.dense_search(query, k_dense)
		b = self.bm25_search(query, k_bm25)
		fused = self.rrf_fuse(d, b, k=max(k_final, top_m))

		candidates: List[Dict] = []
		for row_idx, fscore in fused:
			m = self.metas[row_idx]
			candidates.append({
				"row": row_idx,
				"fused_score": round(float(fscore), 6),
				"chunk_id": m["chunk_id"],
				"doc_id": m["doc_id"],
				"page": m["page"],
				"source_path": m["source_path"],
				"text": self._get_text_by_row(row_idx)
			})
		
		if not rerank:
			# Trim to k_final and attach a short snippet for readability
			out = candidates[:k_final]
			for it in out:
				t = it["text"].replace("\n", " ").strip()
				it["snippet"] = (t[:240] + "...") if len(t) > 240 else t
			return out

		self._ensure_reranker()
		top_pool = candidates[:top_m]
		reranked = self._reranker.rerank(query, top_pool, text_key="text", top_n=k_final)

		for it in reranked:
			t = it["text"].replace("\n", " ").strip()
			it["snippet"] = (t[:240] + "...") if len(t) > 240 else t
		return reranked

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

	def _materialize_items(self, pairs, include_text: bool = True) -> List[Dict]:
		out = []
		for row_idx, fscore in pairs:
			m = self.metas[row_idx]
			item = {
				"row": row_idx,
				"score": float(fscore),
				"chunk_id": m["chunk_id"],
				"doc_id": m["doc_id"],
				"page": m["page"],
				"source_path": m["source_path"],
			}
			if include_text:
				# full text for downstream (LLM or reranker)
				with open(os.path.join(self.art_dir, "chunks.jsonl"), "rb") as f:
					for i, l in enumerate(f):
						if i == row_idx:
							item["text"] = orjson.loads(l)["text"]
							break
			t = item.get("text", "")
			t = t.replace("\n", " ").strip()
			item["snippet"] = (t[:240] + "...") if len(t) > 240 else t
			out.append(item)

		return out

	def search(self, query: str, mode: str = "hybrid",
				k: int = 8, k_dense: int = 20, k_bm25: int = 20,
				rerank: bool = False, top_m: int = 50):
		"""
		mode: 'bm25' | 'dense' | 'hybrid' | 'hybrid_rerank'
		Returns a list of hit dicts aligned with existing /search.
		"""
		t0 = time.time()
		t_dense = t_bm25 = t_rrf = t_rerank = 0
		mode = mode.lower()
		if mode == "bm25":
			s0 = time.time()
			b = self.bm25_search(query, max(k_bm25, k))
			t_bm25 = time.time() - s0
			hits = self._materialize_items(b[:k])
			timings = {
				"t_bm25_ms": int(t_bm25*1000),
				"t_dense_ms": 0,
				"t_rrf_ms": 0,
				"t_rerank_ms": 0
			}
			return hits, timings

		if mode == "dense":
			s0 = time.time()
			d = self.dense_search(query, max(k_dense, k))
			t_dense = time.time() - s0
			hits = self._materialize_items(d[:k])
			timings = {
				"t_bm25_ms": 0,
				"t_dense_ms": int(t_dense*1000),
				"t_rrf_ms": 0,
				"t_rerank_ms": 0
			}
			return hits, timings

		# hybrid family
		s0 = time.time(); d = self.dense_search(query, k_dense); t_dense = time.time() - s0
		s0 = time.time(); b = self.bm25_search(query, k_bm25); t_bm25 = time.time() - s0
		s0 = time.time(); fused = self.rrf_fuse(d, b, k=max(k, top_m)); t_rrf = time.time() - s0
		candidates = self._materialize_items(fused)

		if mode in ("hybrid_rerank",) or rerank:
			rr = Reranker("BAAI/bge-reranker-base")
			pool = candidates[:top_m]
			texts = [it["text"] for it in pool]
			s0 = time.time()
			scores = rr.score_pairs(query, texts, batch_size=32)
			t_rerank = time.time() - s0
			for it, s in zip(pool, scores):
				it["rerank_score"] = float(s)
			pool.sort(key=lambda x: -x["rerank_score"])
			hits = pool[:k]
		else:
			hits = candidates[:k]

		timings = {
			"t_bm25_ms": int(t_bm25*1000),
			"t_dense_ms": int(t_dense*1000),
			"t_rrf_ms": int(t_rrf*1000),
			"t_rerank_ms": int(t_rerank*1000),
		}
		# plain hybrid (RRF only)
		return hits, timings

















