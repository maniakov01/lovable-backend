from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------
# 1) Health
# ---------
@app.get("/health")
def health():
    return {"ok": True}

# -----------------------------
# 2) Request model for /run
#    week can be int OR string OR null
# -----------------------------
class RunRequest(BaseModel):
    week: Optional[Union[int, str]] = None
    scenario: Optional[str] = None

# -----------------------------
# 3) Run endpoint
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