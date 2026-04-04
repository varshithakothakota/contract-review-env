"""
FastAPI server for ContractReviewEnv.

Uses openenv-core's create_fastapi_app when available,
falls back to a hand-rolled FastAPI app with identical endpoints.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

# Ensure project root is importable (critical for HF Spaces)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from server.environment import (
    TASKS,
    ContractAction,
    ContractObservation,
    ContractReviewEnvironment,
    ReviewState,
    grade,
)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ContractReviewEnv",
    description="OpenEnv environment for AI-driven legal contract review",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per task (stateful, single-session)
_envs: Dict[str, ContractReviewEnvironment] = {}

def _get_env(task_id: str) -> ContractReviewEnvironment:
    if task_id not in _envs:
        _envs[task_id] = ContractReviewEnvironment(task_id=task_id)
    return _envs[task_id]


# ── Request / Response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_easy"


class StepRequest(BaseModel):
    task_id: str = "task_easy"
    action: ContractAction


class StepResult(BaseModel):
    observation: ContractObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class GradeRequest(BaseModel):
    task_id: str = "task_easy"


class GradeResult(BaseModel):
    task_id: str
    score: float
    breakdown: Dict[str, Any]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "ContractReviewEnv", "version": "1.0.0"}


@app.get("/")
def root():
    return {
        "name": "ContractReviewEnv",
        "description": "OpenEnv environment for AI-driven legal contract review",
        "tasks": list(TASKS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks", "/health"],
    }


@app.get("/tasks")
def list_tasks():
    return {
        tid: {
            "description": cfg["description"],
            "difficulty": cfg["difficulty"],
            "max_steps": cfg["max_steps"],
            "num_clauses": len(cfg["contract"]["clauses"]),
        }
        for tid, cfg in TASKS.items()
    }


@app.post("/reset", response_model=ContractObservation)
def reset(req: Optional[ResetRequest] = None):
    """Reset the environment and return initial observation."""
    task_id = (req.task_id if req else None) or "task_easy"
    if task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id '{task_id}'")
    obs = _get_env(task_id).reset()
    return obs


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    """Execute one action and return (observation, reward, done, info)."""
    if req.task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id '{req.task_id}'")
    env = _get_env(req.task_id)
    if env._state is None:
        raise HTTPException(400, "Call /reset first.")
    if env._state.done:
        raise HTTPException(400, "Episode done. Call /reset.")
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return StepResult(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state(task_id: str = "task_easy"):
    """Return full internal episode state."""
    if task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id '{task_id}'")
    env = _get_env(task_id)
    if env._state is None:
        raise HTTPException(400, "Call /reset first.")
    return env.state()


@app.post("/grade", response_model=GradeResult)
def grade_episode(req: GradeRequest):
    """Grade current episode. Returns score in [0.0, 1.0]."""
    if req.task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id '{req.task_id}'")
    env = _get_env(req.task_id)
    if env._state is None:
        raise HTTPException(400, "Call /reset first.")
    score, breakdown = grade(req.task_id, env.state())
    return GradeResult(task_id=req.task_id, score=score, breakdown=breakdown)


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def openenv_yaml():
    data = {
        "name": "ContractReviewEnv",
        "version": "1.0.0",
        "description": (
            "An AI agent acts as a junior legal analyst reviewing contracts for "
            "missing clauses, risk indicators, and compliance issues."
        ),
        "tags": ["openenv", "legal", "contract-review", "nlp", "real-world"],
        "tasks": [
            {"id": tid, "description": cfg["description"],
             "difficulty": cfg["difficulty"], "max_steps": cfg["max_steps"]}
            for tid, cfg in TASKS.items()
        ],
        "api": {"reset": "POST /reset", "step": "POST /step",
                "state": "GET /state", "grade": "POST /grade"},
    }
    return yaml.dump(data, default_flow_style=False)
