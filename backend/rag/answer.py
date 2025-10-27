from typing import List, Dict, Optional
import statistics, time

from backend.rag.retrieve import Retriever
from backend.rag.generate import build_prompt
from backend.models.llm import get_llm
from backend.obs.logger import log_event

class Answerer:
	"""
	Orchestrates: retrieve -> gate -> generate -> package
	"""
	def __init__(self, art_dir: str = "artifacts"):
		self.retriever = Retriever(art_dir=art_dir)
		self.llm = get_llm()

	def _contains_any(self, text: str, terms: List[str]) -> bool:
		t = text.lower()
		return any(term in t for term in terms if term)

	def _compute_fused_stats(self, hits: List[Dict]) -> Dict:
		fs = [h.get("fused_score", 0.0) for h in hits]
		return {
			"max": max(fs) if fs else 0.0,
			"mean": statistics.mean(fs) if fs else 0.0,
			"n": len(fs),
		}

	def _should_abstain(self, q: str, hits: List[Dict], rerank: bool) -> Optional[str]:
		# "no context" logic
		if not hits:
			return "no hits"

		# Require at least one hit that contains a rare query term
        # Simple heuristic: split query and filter to words > 3 char
		terms = [w.lower() for w in q.split() if len(w) > 3]
		has_overlap = any(self._contains_any(h.get("text", ""), terms) for h in hits[:3])
		if not has_overlap:
			return "no_term_overlap"

		# Optional rerank threshold (if rerank is on and score is very low)
		if rerank and hits:
			top = hits[0].get("rerank_score", 0.0)
			if top < 0.02:
				return "low_rerank_score"

		return None

	# ---- public entry ----
	def answer(self, q: str, k: int = 4, rerank: bool = True, top_m: int = 24,
				max_tokens: int = 384, temperature: float = 0.1,
				retrieval_mode: str = "hybrid") -> Dict:

		t0 = time.time()
		require_terms = [t for t in q.lower().split() if len(t) > 3]
		mode = retrieval_mode.lower()
		hits, rt = self.retriever.search(
			q,
			mode=("hybrid_rerank" if (mode == "hybrid" and rerank) else mode),
			k=k,
			k_dense=max(20, k*3),
			k_bm25=max(20, k*3),
			rerank=rerank,
			top_m=top_m
		)
		t_retrieve_ms = int((time.time() - t0) * 1000)
		hits = [h for h in hits if any(t in h["text"].lower() for t in require_terms)]

		abstain_reason = self._should_abstain(q, hits, rerank)
		if abstain_reason:
			res = {
				"status": "no_context",
				"reason": abstain_reason,
				"answer": "I don't have enough information in the provided documents.",
				"query": q,
				"hits": [],
				"metrics": {**rt, "t_retrieve_ms": t_retrieve_ms}
			}
			log_event({"route": "chat", "stage": "answer", "status": "no_context", "q": q, **res["metrics"]})
			return res

		# build rpompt with context
		prompt = build_prompt(q, hits)
		t1 = time.time()
		# call LLM
		text, usage = self.llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
		t_gen_ms = int((time.time() - t1) * 1000)

		metrics = {
			**rt,
			"t_retrieve_ms": t_retrieve_ms,
			"t_gen_ms": max(usage.get("gen_ms", t_gen_ms), t_gen_ms),
			"prompt_tokens": usage.get("prompt_tokens"),
			"completion_tokens": usage.get("completion_tokens"),
			"total_tokens": usage.get("total_tokens"),
		}

		log_event({"route": "chat", "stage": "answer", "status": "ok", "q": q, **metrics})

		return{
			"status": "ok",
			"query": q,
			"rerank": rerank,
			"top_m": top_m,
			"hits": [
				{k: v for k, v in h.items() if k != "text"}
				for h in hits
			],
			"answer": text,
			"metrics": metrics,
	}



