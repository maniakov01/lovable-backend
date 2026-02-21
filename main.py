from fastapi import FastAPI, HTTPException
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
    # Accept either 1 or "1" or "W01" or "2026-W01"
    week: Optional[Union[int, str]] = None
    scenario: Optional[str] = "base"

@app.post("/run")
def run(req: RunRequest):
    from optimizer import run_optimization

    # IMPORTANT: If Lovable sends nothing, default to week 1
    week_value = req.week
    if week_value is None:
        week_value = 1

    # Pass week as string OR int — optimizer can decide, but we normalize to string safely
    # (If your optimizer expects int, change this to: week_value = int(week_value))
    week_value = str(week_value)

    try:
        results = run_optimization(week=week_value, scenario=req.scenario)
        return {"status": "done", "results": results}
    except ValueError as e:
        # Your optimizer raises ValueError for out-of-range weeks, etc.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")