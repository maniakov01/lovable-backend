from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

class RunRequest(BaseModel):
    week: Optional[str] = None
    scenario: Optional[str] = None

@app.post("/run")
def run(req: RunRequest):
    try:
        from optimizer import run_optimization
        results = run_optimization(week=req.week, scenario=req.scenario)
        return {"status": "done", "results": results}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))