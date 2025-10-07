import os
import glob
from dataclasses import dataclass, asdict
from typing import List, Dict, Iterator, Tuple
import fitz  # PyMuPDF
import re
import orjson
from transformers import AutoTokenizer
import nltk

# ------------------------------
# Config
# ------------------------------
EMB_MODEL = "BAAI/bge-small-en-v1.5"
TARGET_TOKENS = 500
OVERLAP_TOKENS = 100

# ------------------------------
# Tokenizer (use embedding model's tokenizer for consistency)
# ------------------------------
_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL)
_tokenizer.model_max_length = 100_000

def count_tokens(text: str) -> int:
	return len(_tokenizer.encode(text, add_special_tokens=False, truncation=True))


# ------------------------------
# Basic text cleaning
# ------------------------------
_whitespace_re = re.compile(r"[ \t\u00A0]+")

def clean_text(t: str) -> str:
	t = t.replace("\r", "\n")
	t = re.sub(r"\n{3,}", "\n\n", t)
	t = _whitespace_re.sub(" ", t).strip()
	return t

# ------------------------------
# Sentence splitting (NLTK)
# ------------------------------
def split_by_regex(text: str) -> List[str]:
	splitter = re.compile(r'(?<=[.!?])\s+')
	parts = splitter.split(text.strip())
	return [p for p in parts if p.strip()]

def split_long_text_to_token_windows(text: str, window_tokens: int, overlap_tokens: int) -> List[str]:
	"""Hard-split an oversized segment into token windows to keep lengths under control."""
	ids = _tokenizer.encode(text, add_special_tokens=False)
	out = []
	stride = max(1, window_tokens - overlap_tokens)
	for start in range(0, len(ids), stride):
		end = start + window_tokens
		sub_ids = ids[start:end]
		if not sub_ids:
			break
		out.append(_tokenizer.decode(sub_ids, skip_special_tokens=True))
	return out

def split_sentences(text: str) -> List[str]:
	# fall back if language detection is needed later; for now, assuming English
	base = nltk.sent_tokenize(text)

	safe: List[str] = []
	MAX_SENT_TOKENS = 256 # if a "sentence" exceeds this, it will be sub-splited

	for seg in base:
		if count_tokens(seg) > MAX_SENT_TOKENS:
			# break it into safe windows; use smaller window (e.g., 200) and overlap (50)
			safe.extend(split_long_text_to_token_windows(seg, window_tokens=200, overlap_tokens=50))
		else:
			safe.append(seg)
	return [s for s in safe if s.strip()]


# ------------------------------
# Chunking: build ~TARGET_TOKENS chunks with OVERLAP_TOKENS
# Strategy: add sentences until the target is passed; start next chunk with overlap tail
# ------------------------------
def chunk_by_sentences(sentences: List[str],
					   target_tokens: int = TARGET_TOKENS,
					   overlap_tokens: int = OVERLAP_TOKENS) -> List[str]:
	
	# ensure no single "sentence" exceeds target by forcibly windowing it
	normalized: List[str] = []
	for s in sentences:
		if count_tokens(s) > target_tokens:
			normalized.extend(split_long_text_to_token_windows(s, window_tokens=target_tokens, overlap_tokens=overlap_tokens))
		else:
			normalized.append(s)
	sentences = normalized

	chunks = []
	buf: List[str] = []
	buf_tok = 0

	def flush_chunk():
		nonlocal buf
		if not buf:
			return None
		chunk = " ".join(buf).strip()
		chunks.append(chunk)
		return chunk

	i = 0
	while i < len(sentences):
		s = sentences[i]
		s_tokens = count_tokens(s)
		if buf_tok + s_tokens <= target_tokens or not buf:
			buf.append(s)
			buf_tok += s_tokens
			i += 1
		else:
			# flush current
			chunk = flush_chunk()
			# build overlap tail
			if overlap_tokens > 0:
				tail = []
				tail_tok = 0
				# walk backwards through buf until overlap budget
				for sent in reversed(buf):
					st = count_tokens(sent)
					if tail_tok + st > overlap_tokens and tail:
						break
					tail.append(sent)
					tail_tok += st
				tail = list(reversed(tail))
				buf = tail[:] # start next buffer with overlap tail
				buf_tok = sum(count_tokens(x) for x in buf)
			else:
				buf = []
				buf_tok = 0
	
	# flush remainder
	flush_chunk()
	return chunks

# ------------------------------
# Data model
# ------------------------------
@dataclass
class ChunkRecord:
	doc_id: str
	source_path: str
	page: int
	chunk_id: str
	start_char: int
	end_char: int
	text: str
	n_tokens: int

# ------------------------------
# PDF extraction (page-level)
# ------------------------------
def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
	doc = fitz.open(pdf_path)
	pages = []
	for pno in range(len(doc)):
		text = doc.load_page(pno).get_text("text")
		pages.append((pno + 1, clean_text(text)))
	doc.close()
	return pages

# ------------------------------
# Main pipeline
# ------------------------------
def ingest_folder(input_dir: str, artifacts_dir: str = "artifacts") -> Dict[str, Dict]:
	os.makedirs(artifacts_dir, exist_ok=True)
	out_jsonl = os.path.join(artifacts_dir, "chunks.jsonl")
	meta_path = os.path.join(artifacts_dir, "meta.json")

	pdfs = sorted(glob.glob(os.path.join(input_dir, "**", "*.pdf"), recursive=True))
	meta: Dict[str, Dict] = {}
	n_chunks = 0

	with open(out_jsonl, "wb") as f_out:
		for pdf_path in pdfs:
			doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
			pages = extract_pages(pdf_path)
			meta[doc_id] = {"source_path": pdf_path, "n_pages": len(pages)}
			for page_no, page_text in pages:
				if not page_text:
					continue
				sents = split_sentences(page_text)
				if not sents:
					continue
				chunks = chunk_by_sentences(sents)
				# Map back to char spans for this page 
				page_concat = " ".join(sents)
				for idx, chunk in enumerate(chunks):
					# naive span find; if duplicates exist, find first occurrence then mark used
					start = page_concat.find(chunk)
					end = start + len(chunk) if start != -1 else -1
					rec = ChunkRecord(
										doc_id=doc_id,
										source_path=pdf_path,
										page=page_no,
										chunk_id=f"{doc_id}:{page_no}:{idx+1}",
										start_char=start,
										end_char=end,
										text=chunk,
										n_tokens=count_tokens(chunk),
									)
					f_out.write(orjson.dumps(asdict(rec)) + b"\n")
					n_chunks += 1

	with open(meta_path, "wb") as f_meta:
		f_meta.write(orjson.dumps({"docs": meta, "n_chunks": n_chunks}))

	return {"n_docs": len(pdfs), "n_chunks": n_chunks, "artifacts": {"chunks": out_jsonl, "meta": meta_path}}

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
	import argparse
	ap = argparse.ArgumentParser(description="Ingest PDFs and produce chunk artifacts")
	ap.add_argument("--input", default="data", help="Folder with PDFs")
	ap.add_argument("--out", default="artifacts", help="Artifacts output folder")
	args = ap.parse_args()
	stats = ingest_folder(args.input, args.out)
	print(stats)










