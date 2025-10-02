from fastapi import FastAPI

app = FastAPI(title="DocuChat Pro", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "service": "docuchat-pro", "version": "0.1.0"}
