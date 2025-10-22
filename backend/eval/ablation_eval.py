import os, orjson, argparse, csv
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from backend.rag.answer import Answerer
from backend.rag.retrieve import Retriever

MODES = [
	("bm25", "BM25"),
	("dense", "Dense"),
	("hybrid", "Hybrid"),
	("hybrid_rerank", "Hybrid+Rerank"),	
]

def load_samples(path: str) -> List[Dict]:
	items = []
	with open(path, "rb") as f:
		for line in f:
			line = line.strip()
			if not line: continue
			items.append(orjson.loads(line))
	return items

def get_context_texts(art_dir: str, hits: List[Dict]) -> List[str]:
	out = []
	with open(os.path.join(art_dir, "chunks.jsonl"), "rb") as f:
		lines = f.readlines()
	for h in hits:
		idx = h["row"]
		rec = orjson.loads(lines[idx])
		out.append(rec["text"])
	return out

def run_once(samples: List[Dict], art_dir: str, retrieval_mode: str) -> Dict:
	ans = Answerer(art_dir=art_dir)
	rows = []
	for item in samples:
		q = item["question"]
		gold = item.get("reference_answer", "")
		# rerank only for hybrid if we're in hybrid_rerank mode
		rerank_flag = (retrieval_mode == "hybrid_rerank")
		res = ans.answer(q, k=6, rerank=rerank_flag, top_m=40,
						max_tokens=512, temperature=0.2,
						retrieval_mode=retrieval_mode)
		if res["status"] != "ok":
			rows.append({
				"question": q,
				"answer": res.get("answer", ""),
				"contexts": [],
				"ground_truth": gold,
			})
		else:
			ctx_texts = get_context_texts(art_dir, res["hits"])
			rows.append({
				"question": q,
				"answer": res["answer"],
				"contexts": ctx_texts,
				"ground_truth": gold,
			})
	ds = Dataset.from_list(rows)
	results = evaluate(
		ds,
		metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
	)
	return results

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--samples", default="backend/eval/samples.jsonl")
	ap.add_argument("--art", default="artifacts")
	ap.add_argument("--csv", default="backend/eval/ablation_results.csv")
	args = ap.parse_args()

	samples = load_samples(args.samples)
	table = []
	print("\n=== ABLATION RESULTS ===")
	print(f"n_samples = {len(samples)}\n")

	for mode_key, mode_name in MODES:
		scores = run_once(samples, args.art, mode_key)
		table.append(f"{mode_name:15s} -> {scores}")
	print("\n".join(table))

if __name__ == "__main__":
	main()












