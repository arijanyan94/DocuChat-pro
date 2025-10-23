import os, re
from typing import Dict, List, Optional

USE_OPENAI_MODE = os.getenv("GUARD_USE_OPENAI_MOD", "false").lower() == "true"

JAILBREAK_PATTERNS = [
	r"(?i)\bignore\s+all\s+previous\s+instructions\b",
	r"(?i)\bact\s+as\b.*(system|developer|root|jailbreak)",
	r"(?i)\bdo\s+not\s+follow\s+the\s+rules\b",
	r"(?i)\bdeveloper\s+mode\b",
]
PROMPT_INJECTION_PAT = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]

# Primitive PII detector (expand as needed)
PII_PAT = re.compile(r"(?i)\b(ssn|passport|credit\s*card|cvv|iban|routing\s*number|swift)\b")

# Very basic link policy: disallow remote file/system instructions
DANGEROUS_PAT = re.compile(r"(?i)\b(download|execute|run|shell|system|rm -rf|chmod|curl|wget)\b")

def contains_prompt_injection(text: str) -> bool:
	t = text or ""
	return any(rx.search(t) for rx in PROMPT_INJECTION_PAT)

def contains_pii(text: str) -> bool:
	return bool(PII_PAT.search(text or ""))

def contains_dangerous_ops(text: str) -> bool:
	return bool(DANGEROUS_PAT.search(text or ""))

def too_long(text: str, max_chars: int = 2000) -> bool:
	return len(text or "") > max_chars

def openai_moderation_flag(text: str) -> Optional[str]:
	"""
	Optional moderation with OpenAI if you want a stronger safety check.
	Set GUARD_USE_OPENAI_MOD=true in .env to enable.
	"""
	if not USE_OPENAI_MODE:
		return None
	try:
		from openai import OpenAI
		client = OpenAI()
		resp = client.moderations.create(
			model=os.getenv("OPENAI_MODERATION_MODEL", "omni-moderation-latest"),
			input=text or "",
		)
		result = resp.results[0]
		if result.flagged:
			# return the first flagged category as reason
			for k, v in result.category_scores.items():
				if getattr(result.categories, k, False):
					return f"moderation:{k}"
			return "moderation:flagged"
		return None
	except Exception as e:
		return None

def guard_query(query: str) -> Dict:
	"""
	Returns {ok: bool, reason: Optional[str]}.
	"""
	if too_long(query):
		return {"ok": False, "reason": "too_long"}
	if contains_prompt_injection(query):
		return {"ok": False, "reason": "prompt_injection"}
	if contains_dangerous_ops(query):
		return {"ok": False, "reason": "dangerous_instructions"}
	if contains_pii(query):
		return {"ok": False, "reason": "pii_detected"}

	mod = openai_moderation_flag(query)
	if mod:
		return {"ok": False, "reason": mod}

	return {"ok": True, "reason": None}














