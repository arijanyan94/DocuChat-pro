from fastapi import FastAPI, BackgroundTasks
from backend.rag.ingest import ingest_folder

app = FastAPI(title="DocuChat Pro", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "service": "docuchat-pro", "version": "0.1.0"}

@app.post("/dev/ingest")
def dev_ingest(background_tasks: BackgroundTasks, input_dir: str = "data"):
    def _job():
        ingest_folder(input_dir, "artifacts")
    background_tasks.add_task(_job)
    return {"status": "started", "input_dir": input_dir}