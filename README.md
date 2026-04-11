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

**OpenEnv environment for AI-driven legal contract review.**

An AI agent acts as a senior legal analyst reviewing commercial contracts for
missing clauses, risky terms, and compliance gaps. Unlike toy environments,
clause texts use **realistic legal language** — issues require genuine legal
reasoning, not keyword detection.

---

## Why this environment?

- **Universal business need**: Every company signs contracts
- **Requires reasoning**: Realistic legal prose, not obvious keywords
- **Rich training signal**: Dense per-step rewards with 8 components
- **Clear ground truth**: Issues have objective correct answers
- **Novel in OpenEnv**: No existing legal reasoning environment

---

## Tasks

| Task | Difficulty | Clauses | Critical | High | Medium | Baseline |
|------|-----------|---------|----------|------|--------|----------|
| task_easy | ⭐ | 11 | 1 | 1 | 0 | ~0.42 |
| task_medium | ⭐⭐ | 10 | 3 | 2 | 3 | ~0.38 |
| task_hard | ⭐⭐⭐ | 11 | 2 | 3 | 2 | ~0.31 |

---

## Action Space

| Action | Description |
|--------|-------------|
| `read_clause` | Read clause text in detail |
| `flag_risk` | Flag an issue with severity + finding |
| `mark_compliant` | Mark clause as acceptable |
| `set_risk_rating` | Set overall risk (red/amber/green) |
| `recommend` | Set recommendation (approve/revise/escalate/reject) |
| `request_revision` | Shorthand to request revision |
| `submit_review` | Finalise and end episode |

## Reward Function (dense, per-step)

| Signal | Range |
|--------|-------|
| Flag critical issue (correct severity) | +0.55 to +0.80 |
| Flag high issue | +0.35 to +0.52 |
| Flag medium issue | +0.18 to +0.26 |
| False positive | −0.20 |
| Approve problematic clause | −0.25 |
| Correct risk rating | +0.35 |
| Correct recommendation | +0.35 |

## Quick Start

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
python inference.py
```

## Structure

```
contract-review-env/
├── inference.py          # Baseline agent (stdlib only, runs in-process)
├── models.py             # Typed Pydantic models
├── openenv.yaml
├── pyproject.toml        # server = "server.__main__:main"
├── Dockerfile
└── server/
    ├── __init__.py
    ├── __main__.py       # Entry point
    ├── app.py            # FastAPI: /reset /step /state /grade
    └── environment.py    # All env logic, tasks, graders, rewards
```
