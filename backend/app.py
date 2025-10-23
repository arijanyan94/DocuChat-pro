from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from backend.rag.ingest import ingest_folder
from backend.rag.retrieve import Retriever
from backend.rag.answer import Answerer
from backend.guard.rails import guard_query
from backend.obs.logger import log_event

app = FastAPI(title="DocuChat Pro", version="0.4.0")
RET = None
ANS = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    # Guard input
    verdict = guard_query(req.query)
    if not verdict['ok']:
        log_event({"route": "chat", "action": "blocked",
            "reason": verdict["reason"], "q": req.query})
        return {
            "status": "blocked",
            "reason": verdict["reason"],
            "message": "Your request violates the assistant's safety rules or includes sensitive content."
        }

    # Route to the Answerer, which internally calls retriever
    res = answerer().answer(
        q=req.query, k=req.k, rerank=req.rerank, top_m=req.top_m,
        max_tokens=req.max_tokens, temperature=req.temperature
    )

    # Log outcome
    out = {
        "route": "chat",
        "action": "answered",
        "status": res.get("status"),
        "q": req.query,
        "rerank": req.rerank,
        "k": req.k,
        "top_m": req.top_m,
        "n_hits": len(res.get("hits", [])),
        "reason": res.get("reason"),
    }
    log_event(out)

    return res



