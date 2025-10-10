import os, orjson, argparse, numpy as np, faiss, pickle, math
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

ART = "artifacts"

def read_chunks(chunks_path: str):
	texts, metas = [], []
	with open(chunks_path, "rb") as f:
		for line in f:
			rec = orjson.loads(line)
			texts.append(rec["text"])
			metas.append({
				"chunk_id": rec["chunk_id"],
				"doc_id": rec["doc_id"],
				"page": rec["page"],
				"source_path": rec["source_path"],
				"n_tokens": rec["n_tokens"],
				})
	return texts, metas

def build_bm25(texts: List[str], out_dir: str):
	tokenized = [t.split() for t in texts]
	bm25 = BM25Okapi(tokenized)
	os.makedirs(out_dir, exist_ok=True)
	with open(os.path.join(out_dir, "bm25_tokens.pkl"), "wb") as f:
		pickle.dump(tokenized, f)
	return len(tokenized)

def build_dense(texts: List[str], out_dir: str, model_name="BAAI/bge-small-en-v1.5", batch_size=64):
	model = SentenceTransformer(model_name)
	# Encode in batches
	embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
	embs = np.asarray(embs, dtype="float32")
	# FAISS for cosine -> use inner product with normalized vectors
	index = faiss.IndexFlatIP(embs.shape[1])
	index.add(embs)
	os.makedirs(out_dir, exist_ok=True)
	faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
	np.save(os.path.join(out_dir, "embeddings.npy"), embs) # for testing
	return embs.shape

def write_meta(metas: List[Dict], out_dir: str):
	with open(os.path.join(out_dir, "meta_rows.jsonl"), "wb") as f:
		for m in metas:
			f.write(orjson.dumps(m) + b"\n")
	with open(os.path.join(out_dir, "meta_count.json"), "wb") as f:
		f.write(orjson.dumps({"n_rows": len(metas)}))

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--chunks", default=os.path.join(ART, "chunks.jsonl"))
	ap.add_argument("--out", default=ART)
	ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
	ap.add_argument("--batch", type=int, default=64)
	args = ap.parse_args()

	texts, metas = read_chunks(args.chunks)
	print(f"Loaded chunks: {len(texts)}")

	n_tok = build_bm25(texts, args.out)
	print(f"BM25 tokens prepared: {n_tok}")

	shape = build_dense(texts, args.out, args.model, args.batch)
	print(f"Dense index built: {shape}")

	write_meta(metas, args.out)
	print("Meta written.")	

