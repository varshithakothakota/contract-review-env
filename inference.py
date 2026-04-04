"""
Inference Script — ContractReviewEnv
=====================================
MANDATORY environment variables:
    API_BASE_URL     The API endpoint for the LLM.
    MODEL_NAME       The model identifier to use for inference.
    HF_TOKEN         Your Hugging Face / API key.
    LOCAL_IMAGE_NAME (optional) Docker image name.

Defaults set ONLY for API_BASE_URL and MODEL_NAME (not HF_TOKEN):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ── Env vars (defaults ONLY for API_BASE_URL and MODEL_NAME) ────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
API_KEY          = HF_TOKEN or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK        = "contract_review"
TASKS            = ["task_easy", "task_medium", "task_hard"]
MAX_STEPS_MAP    = {"task_easy": 25, "task_medium": 35, "task_hard": 45}
TEMPERATURE      = 0.1
MAX_TOKENS       = 500
SUCCESS_THRESHOLD = 0.3


# ── Mandatory stdout helpers ─────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action.replace(chr(10),' ')[:100]} "
        f"reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ── HTTP client ──────────────────────────────────────────────────────────────

class ContractReviewClient:
    def __init__(self, base_url: str, task_id: str):
        self.base_url = base_url.rstrip("/")
        self.task_id  = task_id
        self._http    = httpx.AsyncClient(timeout=30.0)

    async def reset(self) -> Dict[str, Any]:
        r = await self._http.post(f"{self.base_url}/reset",
                                  json={"task_id": self.task_id})
        r.raise_for_status()
        return r.json()

    async def step(self, action: Dict) -> Dict[str, Any]:
        r = await self._http.post(f"{self.base_url}/step",
                                  json={"task_id": self.task_id, "action": action})
        r.raise_for_status()
        return r.json()

    async def grade(self) -> Dict[str, Any]:
        r = await self._http.post(f"{self.base_url}/grade",
                                  json={"task_id": self.task_id})
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        await self._http.aclose()

    @classmethod
    async def from_docker_image(cls, image: str, task_id: str,
                                port: int = 7860) -> "ContractReviewClient":
        import subprocess, time
        subprocess.Popen(["docker", "run", "--rm", "-d",
                          "--name", f"contract-review-{task_id}",
                          "-p", f"{port}:{port}", image])
        time.sleep(6)
        return cls(f"http://localhost:{port}", task_id)


# ── LLM Agent ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert junior legal analyst reviewing contracts.
Respond with ONE valid JSON action per step.

Available action_types:
  view_clause      → {"action_type":"view_clause","clause_id":"c01"}
  flag_issue       → {"action_type":"flag_issue","clause_id":"c01","severity":"critical|high|medium|low","issue_description":"..."}
  approve_clause   → {"action_type":"approve_clause","clause_id":"c01"}
  assess_risk      → {"action_type":"assess_risk","clause_id":"overall","risk_level":"red|amber|green"}
  recommend_action → {"action_type":"recommend_action","recommendation":"request_revision|approve|escalate"}
  request_revision → {"action_type":"request_revision"}
  finalize_review  → {"action_type":"finalize_review"}

Strategy:
1. View each clause systematically
2. Flag issues with correct severity:
   - critical: missing clause, unlimited liability, one-sided IP assignment
   - high: unfavourable payment terms, short notice periods, weak SLA
   - medium: vague language, short confidentiality period
3. Approve clean clauses
4. Assess overall risk (red/amber/green)
5. Recommend action and finalize

Severity guide:
- [MISSING CLAUSE] in text → always critical
- Unlimited liability/damages → critical
- IP fully assigned to other party → critical
- Payment >60 days / interest >10% → high
- Notice <14 days → high

Respond with ONE JSON on a single line.
""").strip()


def _build_prompt(obs: Dict, step: int, history: List[str]) -> str:
    clauses = obs.get("clauses", [])
    lines = [
        f"Contract: {obs.get('contract_title')} ({obs.get('contract_type')})",
        f"Step {step}/{obs.get('max_steps',25)} | "
        f"Issues found: {len(obs.get('issues_found',[]))} | "
        f"Approved: {len(obs.get('approved_clauses',[]))}",
        f"Last: {obs.get('last_action_result','none')}",
        "",
        "CLAUSES:",
    ]
    for c in clauses:
        status = ("❌MISSING" if c.get("is_missing")
                  else "🚩FLAGGED" if c.get("is_flagged")
                  else "✅APPROVED" if c.get("is_approved")
                  else "⬜UNREVIEWED")
        lines.append(f"  [{c['clause_id']}] {status} {c['title']}")
        lines.append(f"    {c['text'][:120]}")
    if obs.get("issues_found"):
        lines.append("")
        lines.append("Issues flagged: " + ", ".join(
            f"{i['clause_id']}({i['severity']})" for i in obs["issues_found"]
        ))
    if history:
        lines.append("Recent: " + " | ".join(history[-3:]))
    lines.append("\nRespond with ONE JSON action.")
    return "\n".join(lines)


def _call_llm(client: OpenAI, obs: Dict, step: int, history: List[str]) -> Optional[Dict]:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(obs, step, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = "\n".join(l for l in text.split("\n")
                             if not l.startswith("```")).strip()
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return None


def _fallback(obs: Dict) -> Dict:
    """Heuristic agent when LLM fails."""
    clauses = obs.get("clauses", [])
    flagged = {i["clause_id"] for i in obs.get("issues_found", [])}
    approved = set(obs.get("approved_clauses", []))

    # Find next unreviewed clause
    for c in clauses:
        cid = c["clause_id"]
        if cid in flagged or cid in approved:
            continue
        if c.get("is_missing") or "MISSING" in c.get("text", ""):
            return {"action_type": "flag_issue", "clause_id": cid,
                    "severity": "critical",
                    "issue_description": "Missing clause — not present in contract"}
        # Simple keyword heuristics
        text = c.get("text", "").lower()
        if any(w in text for w in ["90 days", "unlimited", "all rights", "25%"]):
            return {"action_type": "flag_issue", "clause_id": cid,
                    "severity": "high",
                    "issue_description": "Potentially unfavourable terms detected"}
        return {"action_type": "view_clause", "clause_id": cid}

    # All clauses reviewed — finalise
    if not obs.get("risk_assessments"):
        n_issues = len(obs.get("issues_found", []))
        risk = "red" if n_issues >= 3 else "amber" if n_issues >= 1 else "green"
        return {"action_type": "assess_risk", "clause_id": "overall",
                "risk_level": risk}
    if not obs.get("recommendation"):
        return {"action_type": "recommend_action",
                "recommendation": "request_revision"}
    return {"action_type": "finalize_review"}


# ── Task runner ──────────────────────────────────────────────────────────────

async def run_task(task_id: str, llm_client: OpenAI) -> float:
    max_steps   = MAX_STEPS_MAP.get(task_id, 25)
    score       = 0.0
    success     = False
    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    env = (await ContractReviewClient.from_docker_image(LOCAL_IMAGE_NAME, task_id)
           if LOCAL_IMAGE_NAME
           else ContractReviewClient(ENV_BASE_URL, task_id))

    history: List[str] = []

    try:
        obs  = await env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            step += 1
            action = _call_llm(llm_client, obs, step, history) or _fallback(obs)
            action_str = json.dumps(action, separators=(",", ":"))

            try:
                result = await env.step(action)
                reward = float(result.get("reward", 0.0))
                done   = bool(result.get("done", False))
                obs    = result.get("observation", obs)
                error  = (None if obs.get("last_action_success", True)
                          else obs.get("last_action_result", "error"))
            except Exception as exc:
                reward, done, error = 0.0, False, str(exc)[:60]

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward,
                     done=done, error=error)
            history.append(
                f"s{step}:{action.get('action_type','?')}→{reward:+.2f}"
            )

        try:
            gr = await env.grade()
            score = float(gr.get("score", 0.0))
        except Exception:
            score = sum(rewards) / max(1, len(rewards))

        score   = max(0.0, min(1.0, score))
        success = score >= SUCCESS_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] close error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Entry point ──────────────────────────────────────────────────────────────

async def main() -> None:
    llm_client   = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_env     = os.getenv("TASK_ID")
    tasks_to_run = [task_env] if task_env and task_env in TASKS else TASKS

    scores: List[float] = []
    for tid in tasks_to_run:
        s = await run_task(tid, llm_client)
        scores.append(s)

    if len(scores) > 1:
        print(f"[INFO] Avg score: {sum(scores)/len(scores):.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
