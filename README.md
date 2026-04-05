---
title: ContractReviewEnv
emoji: 📋
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - legal
  - contract-review
  - nlp
  - agent-evaluation
---

# 📋 ContractReviewEnv

**An OpenEnv environment for AI-driven legal contract review.**

An AI agent acts as a junior legal analyst, systematically reviewing contracts for missing clauses, unfavourable terms, compliance gaps, and risk indicators. Every company signs contracts — making this one of the most universally applicable agent evaluation domains possible.

---

## Why This Environment?

- **Universal applicability**: Every business reviews contracts
- **Clear success criteria**: Missing clauses and risky terms have objective definitions
- **Rich partial-credit reward**: Correct severity classification is rewarded at each step
- **Novel domain**: Legal review hasn't been tackled in OpenEnv before
- **Genuine RL training value**: Teaches agents to read carefully, reason about risk, and prioritise

---

## 🎯 Action Space

| Action | Description | Key Fields |
|--------|-------------|------------|
| `view_clause` | Read a clause in detail | `clause_id` |
| `flag_issue` | Flag a problem | `clause_id`, `severity`, `issue_description` |
| `approve_clause` | Mark clause as acceptable | `clause_id` |
| `assess_risk` | Set risk level | `clause_id` (or "overall"), `risk_level` |
| `add_comment` | Add review note | `clause_id`, `comment` |
| `recommend_action` | Set overall recommendation | `recommendation` |
| `request_revision` | Shorthand to request changes | — |
| `finalize_review` | End the episode | — |

**Severity levels**: `critical` | `high` | `medium` | `low`  
**Risk levels**: `red` | `amber` | `green`

---

## 👁️ Observation Space

```
ContractObservation:
  contract_title: str
  contract_type: str
  clauses: List[ClauseItem]      # Each with id, title, text, status flags
  issues_found: List[Dict]       # Issues flagged so far
  approved_clauses: List[str]
  risk_assessments: Dict[str, str]
  recommendation: Optional[str]
  step_number: int
  max_steps: int
  done: bool
  last_action_result: str
  last_action_success: bool
```

---

## 🏆 Tasks

### task_easy ⭐ — NDA Review
- **7 clauses** including 1 missing + 1 vague  
- **Goal**: Find the critical missing clause and high-severity vague exclusions  
- **Max steps**: 25 | **Scoring**: 55% critical + 25% high + 20% risk/rec

### task_medium ⭐⭐ — Software Services Agreement
- **8 clauses** with 3 critical + 2 high + 2 medium issues  
- **Goal**: Full issue detection + risk assessment + recommendation  
- **Max steps**: 35 | **Scoring**: 40% critical + 25% high + 10% medium + 25% risk/rec

### task_hard ⭐⭐⭐ — Enterprise License Agreement
- **10 clauses** with 3 critical + 3 high + 1 medium + 3 missing clauses  
- **Goal**: Complete review with missing clause detection + efficiency  
- **Max steps**: 45 | **Scoring**: Multi-component with missing clause bonus

---

## 💰 Reward Function

Dense per-step signals — no sparse rewards:

| Signal | Range | Trigger |
|--------|-------|---------|
| Correct issue detection | +0.3 to +1.0 | Flag real issue (partial credit for severity) |
| False positive | −0.15 | Flag a clean clause as problematic |
| Wrong approval | −0.20 | Approve a problematic clause |
| Risk assessment | +0.4 | Correct overall risk level |
| Recommendation | +0.4 | Correct final action |
| Finalization | 0 to +0.5 | Coverage-weighted completion bonus |

---

## 🚀 Setup

```bash
# Clone and run locally
pip install -r requirements.txt   # or: uv sync
uv run server                     # starts on :7860

# Docker
docker build -t contract-review-env .
docker run -p 7860:7860 contract-review-env

# Run baseline inference
export API_BASE_URL=https://router.huggingface.co/v1
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
python inference.py
```

## API Quick Start

```python
import requests
BASE = "http://localhost:7860"

# Reset
obs = requests.post(f"{BASE}/reset", json={"task_id": "task_medium"}).json()

# Step
result = requests.post(f"{BASE}/step", json={
    "task_id": "task_medium",
    "action": {
        "action_type": "flag_issue",
        "clause_id": "s04",
        "severity": "critical",
        "issue_description": "No limitation of liability clause"
    }
}).json()
print(result["reward"])   # 0.5

# Grade
score = requests.post(f"{BASE}/grade", json={"task_id": "task_medium"}).json()
print(score["score"])     # 0.0 – 1.0
```

## 📁 Structure

```
contract-review-env/
├── inference.py          # Baseline agent (mandatory)
├── openenv.yaml          # Environment metadata
├── pyproject.toml        # Dependencies + entry point
├── Dockerfile
├── README.md
└── server/
    ├── __init__.py
    ├── __main__.py       # Entry point: def main()
    ├── app.py            # FastAPI server
    └── environment.py    # All env logic: models, tasks, graders, rewards
```

## License
MIT
