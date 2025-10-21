import os
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
		r = self.client.chat.completions.create(
			model=self.model,
			temperature=temperature,
			max_tokens=max_tokens,
			messages=[
				{"role": "system", "content": "You are a careful assistant that cites sources."},
				{"role": "user", "content": prompt}
			],
		)
		return r.choices[0].message.content.strip()

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




