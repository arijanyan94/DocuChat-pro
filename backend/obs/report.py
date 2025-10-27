import json, argparse, statistics as stats
from collections import defaultdict

def load(path):
	with open(path, "r") as f:
		for line in f:
			line = line.strip()
			if not line: continue
			try:
				yield json.loads(line)
			except Exception:
				continue

def fmt(ms_list):
	if not ms_list: return "-"
	p50 = f"p50={int(stats.median(ms_list))}ms"
	p95 = f"p95={int(stats.quantiles(ms_list, n=20)[18])}ms"
	n = f"n={len(ms_list)}"
	return  f"{p50} {p95} {n}"

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--log", default="runtime/requests.log")
	args = ap.parse_args()

	rows = [e for e in load(args.log) if e.get("route")=="chat" and e.get("stage")=="answer"]
	if not rows:
		print("No chat answer events found")
		return

	# Overall latency
	t_retrieve = [e.get("t_retrieve_ms") for e in rows if isinstance(e.get("t_retrieve_ms"), int)]
	t_gen = [e.get("t_gen_ms") for e in rows if instance(e.get("t_gen_ms"), int)]
	print("\n=== Latency Summary ===")
	print("retrieve:", fmt(t_retrieve))
	print("generate:", fmt(t_gen))

	# Sub-stage timings
	t_bm25 = [e.get("t_bm25_ms") for e in rows if isinstance(e.get("t_bm25_ms"), int)]
	t_dense= [e.get("t_dense_ms") for e in rows if isinstance(e.get("t_dense_ms"), int)]
	t_rrf  = [e.get("t_rrf_ms") for e in rows if isinstance(e.get("t_rrf_ms"), int)]
	t_rer  = [e.get("t_rerank_ms") for e in rows if isinstance(e.get("t_rerank_ms"), int)]
	print("\nsub-stages:  bm25:", fmt(t_bm25), " dense:", fmt(t_dense),
		  " rrf:", fmt(t_rrf), " rerank:", fmt(t_rer))

	# Token usage
	ptoks = [e.get("prompt_tokens") for e in rows if isinstance(e.get("prompt_tokens"), int)]
	ctoks = [e.get("completion_tokens") for e in rows if isinstance(e.get("completion_tokens"), int)]
	ttoks = [e.get("total_tokens") for e in rows if isinstance(e.get("total_tokens"), int)]
	if ttoks:
		print("\n=== Token Usage ===")
		print(f"prompt: mean={int(stats.mean(ptoks))}  completion: mean={int(stats.mean(ctoks))}  total: mean={int(stats.mean(ttoks))}")

	# Status breakdown
	by_status = defaultdict(int)
	for e in rows: by_status[e.get("status")] += 1
	print("\n=== Status Counts ===")
	for k,v in by_status.items(): print(f"{k}: {v}")

if __name__ == "__main__":
	main()