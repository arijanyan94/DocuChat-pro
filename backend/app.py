from fastapi import FastAPI, BackgroundTasks, Query
from pydantic import BaseModel
from typing import List, Optional
from backend.rag.ingest import ingest_folder
from backend.rag.retrieve import Retriever
from backend.rag.answer import Answerer

app = FastAPI(title="DocuChat Pro", version="0.4.0")
RET = None
ANS = None

def retriever() -> Retriever:
    global RET
    if RET is None:
        RET = Retriever(art_dir="artifacts") # lazy-load on first call
    return RET

def answerer() -> Answerer:
    global ANS
    if ANS is None:
        ANS = Answerer(art_dir="artifacts")
    return ANS

@app.get("/health")
def health():
    return {"status": "ok", "service": "docuchat-pro", "version": "0.1.0"}

class ChatRequest(BaseModel):
    query: str
    k: int = 6
    rerank: bool = True
    top_m: int = 40
    max_tokens: int = 512
    temperature: float = 0.2

@app.post("/dev/ingest")
def dev_ingest(background_tasks: BackgroundTasks, input_dir: str = "data"):
    def _job():
        ingest_folder(input_dir, "artifacts")
    background_tasks.add_task(_job)
    return {"status": "started", "input_dir": input_dir}

@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = 8,
            rerank: bool = False, top_m: int = 50):
    r = retriever().hybrid(q, k_dense=max(20, k*3), k_bm25=max(20, k*3),
                            k_final=k, rerank=rerank, top_m=top_m)
    return {"query": q, "k": k, "rerank": rerank, "top_m": top_m, "hits": r}

@app.post("/chat")
def chat(req: ChatRequest):
    # Route to the Answerer, which internally calls retriever
    res = answerer().answer(
        q=req.query, k=req.k, rerank=req.rerank, top_m=req.top_m,
        max_tokens=req.max_tokens, temperature=req.temperature
    )
    return res



