from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder

class Reranker:
	"""
	Cross-encoder reranker: given a query and a list of candidate chunks (texts),
	returns the same candidates sorted by a direct relevance score.
	"""

	def __init__(self, model_name: str = "BAAI/bge-reranker-base", max_length: int = 512):
		self.model = CrossEncoder(model_name, max_length=max_length)

	def score_pairs(self, query: str, texts: List[str], batch_size: int = 32) -> List[float]:
		pairs = [(query, t) for t in texts]
		scores = self.model.predict(pairs, batch_size=batch_size).tolist()
		return [float(s) for s in scores]

	def rerank(self, query: str, items: List[Dict], text_key: str = "text",
				top_n: int = 8, batch_size: int = 32) -> List[Dict]:
		"""
		items: list of dicts each containing at least {text: "..."} plus your metadata.
		Returns the same items sorted by cross-encoder score (desc), truncated to top_n, and with score attached.
		"""
		if not items:
			return []
		texts = [it[text_key] for it in items]
		scores = self.score_pairs(query, texts, batch_size=batch_size)
		for it, s in zip(items, scores):
			it["rerank_score"] = round(s, 6)
		items = sorted(items, key=lambda x: -x["rerank_score"])
		return items[:top_n]
		