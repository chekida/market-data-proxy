# app.py — minimal sanity app for Render
import os
from fastapi import FastAPI

app = FastAPI(title="Sanity App")

@app.get("/")
def root():
    return {"ok": True}

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

@app.get("/getHealth", include_in_schema=False)
def get_health():
    return {"status": "ok"}

@app.get("/debug/routes", include_in_schema=False)
def routes():
    return [{"path": r.path, "methods": sorted(list(r.methods))} for r in app.routes]

if _name_ == "_main_":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","10000")))

