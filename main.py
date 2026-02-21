from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

app = FastAPI()

# ---------
# 1) Health
# ---------
@app.get("/health")
def health():
    return {"ok": True}

# -----------------------------
# 2) Request model for /run
# -----------------------------
class RunRequest(BaseModel):
    # These are optional inputs you can send from Lovable later
    week: Optional[str] = None
    scenario: Optional[str] = None

# -----------------------------
# 3) Response shape for /run
# -----------------------------
@app.post("/run")
def run(req: RunRequest):

    from optimizer import run_optimization

    results = run_optimization(
        week=req.week,
        scenario=req.scenario
    )

    return {
        "status": "done",
        "results": results
    }