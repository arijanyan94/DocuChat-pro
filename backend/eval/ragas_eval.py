import os, orjson, argparse
from typing import List, Dict
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate

from backend.rag.retrieve import Retriever
from backend.rag.answer import Answerer

def load_samples(path: str) -> List[Dict]:
	items = []
	with open(path, "rb") as f:
		for line in f:
			if not line.strip(): continue
			items.append(orjson.loads(line))
	return items

def run_pipeline(answerer: Answerer, q: str, k: int = 6, rerank: bool = True, top_m: int = 40):
	res = answerer.answer(q, k=k, rerank=rerank, top_m=top_m, max_tokens=512, temperature=0.2)
	# Build fields expected by RAGAS
	if res["status"] != "ok":
		return {
			"question": q,
			"answer": res.get("answer", ""),
			"contexts": [],
			"status": res["status"]
		}
	contexts = []
	for h in res["hits"]:
		pass
	return res

def get_context_texts(art_dir: str, hits: List[Dict]) -> List[str]:
	"""Helper to map hits (row indices) to raw chunk texts"""
	r = Retriever(art_dir=art_dir)
	out = []
	for h in hits:
		idx = h["row"]
		with open(os.path.join(art_dir, "chunks.jsonl"), "rb") as f:
			for i, l in enumerate(f):
				if i == idx:
					out.append(orjson.loads(l)["text"])
					break
	return out

def prepare_ragas_dataset(samples_path: str, art_dir: str = "artifacts") -> Dataset:
	ans = Answerer(art_dir=art_dir)
	rows = []
	for item in load_samples(samples_path):
		q = item["question"]
		gold = item.get("reference_answer", "")
		res = ans.answer(q, k=6, rerank=True, top_m=40, max_tokens=512, temperature=0.2)
		if res['status'] != "ok":
			rows.append({
				"question": q,
				"answer": res.get("answer", ""),
				"contexts": [],
				"ground_truth": gold
				})
		else:
			ctx_texts = get_context_texts(art_dir, res["hits"])
			rows.append({
				"question": q,
				"answer": res["answer"],
				"contexts": ctx_texts,
				"ground_truth": gold
				})
	return Dataset.from_list(rows)

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--samples", default="backend/eval/samples.jsonl")
	ap.add_argument("--art", default="artifacts")
	args = ap.parse_args()

	ds = prepare_ragas_dataset(args.samples, args.art)

	results = evaluate(
		ds,
		metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
	)
	print("\n=== RAGAS RESULTS ===")
	print(results)

if __name__ == "__main__":
	main()






