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

@app.get("/health")
def health():
    return {"ok": True}

class RunRequest(BaseModel):
    # ✅ accept both 1 and "W01" and "2026-W08"
    week: Optional[Union[int, str]] = None
    scenario: Optional[str] = None

@app.post("/run")
def run(req: RunRequest):
    from optimizer import run_optimization

    # Normalize week into something optimizer can parse
    week_value = None
    if req.week is not None:
        week_value = str(req.week).strip()   # "1" or "W01" or "2026-W08"

    results = run_optimization(
        week=week_value,
        scenario=req.scenario
    )

    return {
        "status": "done",
        "results": results
    }