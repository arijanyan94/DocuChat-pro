import os, time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class LLMBase:
	def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
		raise NotImplementedError

class OpenAIChat(LLMBase):
	"""
	Simple wrapper for OpenAI Chat Completions (o4-mini / gpt-4o-mini / gpt-4o).
	Replace with your preferred model. Requires OPENAI_API_KEY in env.
	"""
	def __init__(self, model: str = None):
		from openai import OpenAI
		self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
		self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

	def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
		t0 = time.time()
		resp = self.client.chat.completions.create(
			model=self.model,
			temperature=temperature,
			max_tokens=max_tokens,
			messages=[
				{"role": "system", "content": "You are a careful assistant that cites sources."},
				{"role": "user", "content": prompt}
			],
		)
		dt = time.time() - t0
		text = resp.choices[0].message.content.strip() or ""
		u = getattr(resp, "usage", None)

		def _get_u(attr, default=None):
			# pydantic model attributes
			if u is not None and hasattr(u, attr):
				try:
					return int(getattr(u, attr))
				except Exception:
					return default
			if isinstance(u, dict):
				try:
					return int(u.get(attr, default))
				except Exception:
					return default
			return default

		usage = {
			"prompt_tokens": _get_u("prompt_tokens"),
			"completions_tokens": _get_u("completions_tokens"),
			"total_tokens": _get_u("total_tokens"),
			"gen_ms": int((time.time() - dt) * 1000),
		}
		return text, usage

class OllamaChat(LLMBase):
	"""
	Local alternative via Ollama (e.g., llama3). Requires `pip install ollama`.
	"""
	def __init__(self, model: str = None):
		import ollama
		self.ollama = ollama
		self.model = model or os.getenv("OLLAMA_MODEL", "llama3")

	def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
		r = self.ollama.chat(model=self.model, message=[
			{"role": "system", "content": "You are a careful assistant that cites sources."},
			{"role": "user", "content": prompt}
		], options={"temperature": temperature, "num_predict": max_tokens})
		return r["message"]["content"].strip()

def get_llm():
	provider = os.getenv("MODEL_PROVIDER", "openai").lower()
	if provider == "ollama":
		return OllamaChat()
	return OpenAIChat()




