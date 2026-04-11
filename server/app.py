"""FastAPI server — ContractReviewEnv."""
from __future__ import annotations
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Any, Dict, Optional
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

try:
    from server.environment import TASKS, ContractReviewEnvironment, grade_task
except ImportError:
    from environment import TASKS, ContractReviewEnvironment, grade_task

app = FastAPI(title="ContractReviewEnv", version="1.0.0",
              description="OpenEnv — AI-driven legal contract review")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_envs: Dict[str, ContractReviewEnvironment] = {}

def _get(task_id: str) -> ContractReviewEnvironment:
    if task_id not in _envs:
        _envs[task_id] = ContractReviewEnvironment(task_id)
    return _envs[task_id]


@app.get("/health")
def health():
    return {"status": "ok", "service": "ContractReviewEnv", "version": "1.0.0"}

@app.get("/")
def root():
    return {"name": "ContractReviewEnv",
            "tasks": list(TASKS),
            "endpoints": ["/reset","/step","/state","/grade","/tasks","/health"]}

@app.get("/tasks")
def list_tasks():
    return {tid: {"description": cfg["description"],
                  "difficulty":  cfg["difficulty"],
                  "max_steps":   cfg["max_steps"],
                  "num_clauses": len(cfg["clauses"])}
            for tid, cfg in TASKS.items()}

@app.post("/reset")
def reset(body: Optional[Dict[str, Any]] = None):
    task_id = (body or {}).get("task_id", "task_easy")
    if task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id '{task_id}'")
    return _get(task_id).reset()

@app.post("/step")
def step(body: Dict[str, Any]):
    task_id = body.get("task_id", "task_easy")
    action  = body.get("action", {})
    if task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id '{task_id}'")
    e = _get(task_id)
    if e._state is None:
        raise HTTPException(400, "Call /reset first")
    if e._state["done"]:
        raise HTTPException(400, "Episode done. Call /reset")
    obs, reward, done, info = e.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.get("/state")
def state(task_id: str = "task_easy"):
    if task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id '{task_id}'")
    e = _get(task_id)
    if e._state is None:
        raise HTTPException(400, "Call /reset first")
    return e.state()

@app.post("/grade")
def grade(body: Dict[str, Any]):
    task_id = body.get("task_id", "task_easy")
    if task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id '{task_id}'")
    e = _get(task_id)
    if e._state is None:
        raise HTTPException(400, "Call /reset first")
    score, bd = grade_task(task_id, e.state())
    return {"task_id": task_id, "score": score, "breakdown": bd}

@app.get("/openenv.yaml", response_class=PlainTextResponse)
def openenv_yaml():
    return yaml.dump({
        "name": "ContractReviewEnv", "version": "1.0.0",
        "tasks": [{"id": tid, "difficulty": v["difficulty"],
                   "max_steps": v["max_steps"]}
                  for tid, v in TASKS.items()],
        "api": {"reset":"POST /reset","step":"POST /step",
                "state":"GET /state","grade":"POST /grade"},
    })


def main() -> None:
    """Entry point — uv run server / openenv validator."""
    import uvicorn
    uvicorn.run("server.app:app",
                host=os.getenv("HOST","0.0.0.0"),
                port=int(os.getenv("PORT","7860")),
                workers=1, log_level="info")

if __name__ == "__main__":
    main()
