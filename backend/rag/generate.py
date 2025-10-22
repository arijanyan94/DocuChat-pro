from typing import List, Dict

SYSTEM_PROMPT = """
You are a cautious assistant that answers using ONLY the provided context.
Rules:
- If the context is insufficient, say exactly: "I don't have enough information in the provided documents."
- Cite sources inline as [doc_id:page] next to each claim.
- Do not include any statement that is not explicitly supported by the context text.
- Prefer concise, well-structured answers; omit unrelated details from the context.
"""

USER_TEMPLATE = """Question:
{question}

Context (each item is a chunk from a document):
{context}

Write a helpful answer grounded ONLY in the context, with inline citations like [doc:page].
If information is missing, explicitly say you don't have enough information.
"""

def format_context(chunks: List[Dict], max_chars_per_chunk: int = 1200) -> str:
	lines = []
	for i, c in enumerate(chunks, 1):
		doc = c.get("doc_id", "doc")
		page = c.get("page", "?")
		text = c.get("text", "")[:max_chars_per_chunk].replace("\n", " ").strip()
		lines.append(f"[{doc}:{page}] {text}")
	return "\n".join(lines)

def build_prompt(question: str, chunks: List[Dict]) -> str:
	return SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(
		question=question,
		context=format_context(chunks)
	)