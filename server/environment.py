"""
ContractReviewEnv — complete environment logic.

An AI agent acts as a junior legal analyst, reviewing contracts for:
  - Missing critical clauses
  - Risk indicators (unfavourable terms)
  - Compliance flags (GDPR, IP, liability)
  - Overall completeness

All models, tasks, graders, and reward logic live here so
the server package is fully self-contained.
"""

from __future__ import annotations

import copy
import random
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class ClauseSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RiskLevel(str, Enum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"

class ActionType(str, Enum):
    VIEW_CLAUSE       = "view_clause"
    FLAG_ISSUE        = "flag_issue"
    APPROVE_CLAUSE    = "approve_clause"
    REQUEST_REVISION  = "request_revision"
    ADD_COMMENT       = "add_comment"
    ASSESS_RISK       = "assess_risk"
    RECOMMEND_ACTION  = "recommend_action"
    FINALIZE_REVIEW   = "finalize_review"


# ─────────────────────────────────────────────
# Pydantic Models (OpenEnv Action / Observation)
# ─────────────────────────────────────────────

class ContractAction(BaseModel):
    action_type: ActionType
    clause_id: Optional[str]       = None
    issue_description: Optional[str] = None
    severity: Optional[ClauseSeverity] = None
    risk_level: Optional[RiskLevel]    = None
    comment: Optional[str]         = None
    recommendation: Optional[str]  = None

    class Config:
        use_enum_values = True


class ClauseItem(BaseModel):
    clause_id: str
    title: str
    text: str
    is_missing: bool = False
    is_flagged: bool = False
    is_approved: bool = False
    flags: List[str] = Field(default_factory=list)


class ContractObservation(BaseModel):
    # Contract info
    contract_title: str
    contract_type: str
    clauses: List[ClauseItem]

    # Review state
    step_number: int = 0
    max_steps: int = 25
    issues_found: List[Dict[str, Any]] = Field(default_factory=list)
    approved_clauses: List[str]        = Field(default_factory=list)
    risk_assessments: Dict[str, str]   = Field(default_factory=dict)
    recommendation: Optional[str]      = None

    # Feedback
    last_action_result: str  = ""
    last_action_success: bool = True

    # Score tracking
    total_clauses: int = 0
    critical_issues_found: int = 0
    done: bool = False


class ReviewState(BaseModel):
    episode_id: str
    task_id: str
    difficulty: str
    step_count: int = 0
    done: bool = False
    cumulative_reward: float = 0.0

    # Full contract data
    contract: Dict[str, Any]             = Field(default_factory=dict)
    ground_truth: Dict[str, Any]         = Field(default_factory=dict)
    issues_found: List[Dict[str, Any]]   = Field(default_factory=list)
    approved_clauses: List[str]          = Field(default_factory=list)
    risk_assessments: Dict[str, str]     = Field(default_factory=dict)
    recommendation: Optional[str]        = None
    actions_log: List[Dict[str, Any]]    = Field(default_factory=list)


# ─────────────────────────────────────────────
# Contract Templates (synthetic data)
# ─────────────────────────────────────────────

NDA_CONTRACT = {
    "title": "Non-Disclosure Agreement",
    "type": "NDA",
    "clauses": [
        {
            "clause_id": "c01",
            "title": "Parties",
            "text": "This Agreement is entered into between Acme Corp ('Disclosing Party') and Beta LLC ('Receiving Party').",
            "is_missing": False,
        },
        {
            "clause_id": "c02",
            "title": "Definition of Confidential Information",
            "text": "Confidential Information means any data or information that is proprietary to the Disclosing Party.",
            "is_missing": False,
        },
        {
            "clause_id": "c03",
            "title": "Obligations of Receiving Party",
            "text": "The Receiving Party agrees to hold the Confidential Information in strict confidence.",
            "is_missing": False,
        },
        {
            "clause_id": "c04",
            "title": "Term",
            "text": "This Agreement shall remain in effect for two (2) years from the Effective Date.",
            "is_missing": False,
        },
        {
            "clause_id": "c05",
            "title": "Exclusions from Confidential Information",
            "text": "[PLACEHOLDER - This clause is intentionally left vague with no specific exclusions listed.]",
            "is_missing": False,
            "_has_issue": True,
            "_issue": "No specific exclusions defined (public domain, prior knowledge, etc.)",
            "_severity": "high",
        },
        {
            "clause_id": "c06",
            "title": "Return of Information",
            "text": "[MISSING CLAUSE]",
            "is_missing": True,
            "_issue": "No clause requiring return or destruction of confidential materials",
            "_severity": "critical",
        },
        {
            "clause_id": "c07",
            "title": "Governing Law",
            "text": "This Agreement shall be governed by the laws of the State of Delaware.",
            "is_missing": False,
        },
    ],
    "ground_truth": {
        "critical_issues": ["c06"],
        "high_issues": ["c05"],
        "overall_risk": "amber",
        "recommendation": "request_revision",
        "missing_clauses": ["c06"],
    },
}

SERVICE_AGREEMENT = {
    "title": "Software Services Agreement",
    "type": "Service Agreement",
    "clauses": [
        {
            "clause_id": "s01",
            "title": "Scope of Services",
            "text": "Provider agrees to deliver software development services as described in Schedule A.",
            "is_missing": False,
        },
        {
            "clause_id": "s02",
            "title": "Payment Terms",
            "text": "Client shall pay invoices within 90 days of receipt. Late payments accrue interest at 15% per annum.",
            "is_missing": False,
            "_has_issue": True,
            "_issue": "90-day payment terms highly unfavourable to Provider; 15% interest unusually high",
            "_severity": "high",
        },
        {
            "clause_id": "s03",
            "title": "Intellectual Property",
            "text": "All work product created under this Agreement shall remain the exclusive property of Client.",
            "is_missing": False,
            "_has_issue": True,
            "_issue": "IP clause gives all rights to Client — Provider retains no rights to pre-existing IP",
            "_severity": "critical",
        },
        {
            "clause_id": "s04",
            "title": "Limitation of Liability",
            "text": "[MISSING CLAUSE]",
            "is_missing": True,
            "_issue": "No limitation of liability — Provider exposed to unlimited damages",
            "_severity": "critical",
        },
        {
            "clause_id": "s05",
            "title": "Confidentiality",
            "text": "Both parties agree to keep proprietary information confidential for a period of 1 year.",
            "is_missing": False,
            "_has_issue": True,
            "_issue": "1-year confidentiality period is too short for a software services engagement",
            "_severity": "medium",
        },
        {
            "clause_id": "s06",
            "title": "Termination",
            "text": "Client may terminate this Agreement at any time with 7 days written notice.",
            "is_missing": False,
            "_has_issue": True,
            "_issue": "7-day notice too short; Provider has no reciprocal termination right",
            "_severity": "high",
        },
        {
            "clause_id": "s07",
            "title": "Dispute Resolution",
            "text": "Disputes shall be resolved by arbitration in Client's home jurisdiction.",
            "is_missing": False,
            "_has_issue": True,
            "_issue": "Arbitration venue biased toward Client; no neutral jurisdiction clause",
            "_severity": "medium",
        },
        {
            "clause_id": "s08",
            "title": "Data Protection",
            "text": "[MISSING CLAUSE]",
            "is_missing": True,
            "_issue": "No GDPR/data protection clause despite processing personal data",
            "_severity": "critical",
        },
    ],
    "ground_truth": {
        "critical_issues": ["s03", "s04", "s08"],
        "high_issues": ["s02", "s06"],
        "medium_issues": ["s05", "s07"],
        "overall_risk": "red",
        "recommendation": "request_revision",
        "missing_clauses": ["s04", "s08"],
    },
}

LICENSE_AGREEMENT = {
    "title": "Enterprise Software License & SaaS Agreement",
    "type": "Enterprise License",
    "clauses": [
        {
            "clause_id": "l01",
            "title": "License Grant",
            "text": "Licensor grants Licensee a non-exclusive, non-transferable license to use the Software.",
            "is_missing": False,
        },
        {
            "clause_id": "l02",
            "title": "Permitted Users",
            "text": "The license covers up to 50 named users. Additional users require written consent and additional fees.",
            "is_missing": False,
        },
        {
            "clause_id": "l03",
            "title": "Subscription Fees",
            "text": "Annual fees are $240,000. Licensor may increase fees by up to 25% annually with 30 days notice.",
            "is_missing": False,
            "_has_issue": True,
            "_issue": "25% annual price increase with only 30 days notice is excessive for enterprise",
            "_severity": "high",
        },
        {
            "clause_id": "l04",
            "title": "Service Level Agreement",
            "text": "Licensor guarantees 99% uptime. Penalties for downtime: credit of 5% of monthly fee per hour of outage.",
            "is_missing": False,
            "_has_issue": True,
            "_issue": "99% uptime (87 hrs downtime/yr) too low for enterprise; penalties are capped too low",
            "_severity": "high",
        },
        {
            "clause_id": "l05",
            "title": "Data Ownership",
            "text": "All data uploaded to the platform remains property of Licensee. Licensor may use anonymised data for product improvement.",
            "is_missing": False,
            "_has_issue": True,
            "_issue": "Anonymised data usage clause lacks definition of anonymisation standard",
            "_severity": "medium",
        },
        {
            "clause_id": "l06",
            "title": "Security & Compliance",
            "text": "[MISSING CLAUSE]",
            "is_missing": True,
            "_issue": "No security standards clause (SOC 2, ISO 27001) for enterprise SaaS",
            "_severity": "critical",
        },
        {
            "clause_id": "l07",
            "title": "Audit Rights",
            "text": "[MISSING CLAUSE]",
            "is_missing": True,
            "_issue": "No audit rights for Licensee to verify compliance",
            "_severity": "critical",
        },
        {
            "clause_id": "l08",
            "title": "Indemnification",
            "text": "Licensee shall indemnify Licensor against all third-party claims arising from Licensee's use.",
            "is_missing": False,
            "_has_issue": True,
            "_issue": "One-sided indemnification — Licensor has no IP infringement indemnity obligation",
            "_severity": "critical",
        },
        {
            "clause_id": "l09",
            "title": "Termination for Convenience",
            "text": "Either party may terminate with 90 days written notice.",
            "is_missing": False,
        },
        {
            "clause_id": "l10",
            "title": "Data Portability & Exit",
            "text": "[MISSING CLAUSE]",
            "is_missing": True,
            "_issue": "No data export/portability clause — vendor lock-in risk",
            "_severity": "high",
        },
    ],
    "ground_truth": {
        "critical_issues": ["l06", "l07", "l08"],
        "high_issues": ["l03", "l04", "l10"],
        "medium_issues": ["l05"],
        "overall_risk": "red",
        "recommendation": "request_revision",
        "missing_clauses": ["l06", "l07", "l10"],
    },
}

TASKS = {
    "task_easy": {
        "description": (
            "Review a Non-Disclosure Agreement (NDA) for completeness and issues. "
            "Find the 1 critical missing clause and 1 high-severity issue within 25 steps."
        ),
        "difficulty": "easy",
        "max_steps": 25,
        "contract": NDA_CONTRACT,
        "seed": 42,
    },
    "task_medium": {
        "description": (
            "Review a Software Services Agreement with 3 critical issues, 2 high, and 2 medium issues. "
            "Flag all critical and high issues, assess overall risk, and recommend action within 35 steps."
        ),
        "difficulty": "medium",
        "max_steps": 35,
        "contract": SERVICE_AGREEMENT,
        "seed": 99,
    },
    "task_hard": {
        "description": (
            "Review an Enterprise Software License Agreement with 3 critical, 3 high, and 1 medium issue. "
            "Find all issues, assess risk per clause, provide compliance recommendations within 45 steps."
        ),
        "difficulty": "hard",
        "max_steps": 45,
        "contract": LICENSE_AGREEMENT,
        "seed": 7,
    },
}


# ─────────────────────────────────────────────
# Reward Calculator
# ─────────────────────────────────────────────

def compute_reward(
    action: ContractAction,
    state: ReviewState,
    action_result: Dict[str, Any],
) -> float:
    """Dense per-step reward with partial credit."""
    task     = TASKS[state.task_id]
    contract = task["contract"]
    gt       = contract["ground_truth"]
    clause_id = action.clause_id

    reward = 0.0

    if action.action_type == ActionType.FLAG_ISSUE:
        clause = next((c for c in contract["clauses"] if c["clause_id"] == clause_id), None)
        if clause and clause.get("_has_issue"):
            true_sev = clause.get("_severity", "low")
            pred_sev = action.severity or "low"
            if pred_sev == true_sev:
                sev_score = 1.0
            else:
                sev_map = {"critical": 3, "high": 2, "medium": 1, "low": 0}
                dist = abs(sev_map.get(pred_sev, 0) - sev_map.get(true_sev, 0))
                sev_score = max(0.0, 1.0 - dist * 0.35)
            if clause_id in gt.get("critical_issues", []):
                reward = 0.5 + 0.5 * sev_score
            elif clause_id in gt.get("high_issues", []):
                reward = 0.3 + 0.3 * sev_score
            else:
                reward = 0.1 + 0.1 * sev_score
        elif clause and clause.get("is_missing"):
            true_sev = clause.get("_severity", "critical")
            pred_sev = action.severity or "low"
            reward = 0.6 if pred_sev == true_sev else 0.3
        else:
            reward = -0.15  # false positive

    elif action.action_type == ActionType.APPROVE_CLAUSE:
        clause = next((c for c in contract["clauses"] if c["clause_id"] == clause_id), None)
        if clause and not clause.get("_has_issue") and not clause.get("is_missing"):
            reward = 0.1
        elif clause and (clause.get("_has_issue") or clause.get("is_missing")):
            reward = -0.2  # approved a problematic clause

    elif action.action_type == ActionType.ASSESS_RISK:
        true_risk = gt.get("overall_risk", "amber")
        pred_risk = action.risk_level or "green"
        if pred_risk == true_risk:
            reward = 0.4
        elif (pred_risk == "amber" and true_risk == "red") or \
             (pred_risk == "red" and true_risk == "amber"):
            reward = 0.15
        else:
            reward = -0.1

    elif action.action_type == ActionType.RECOMMEND_ACTION:
        true_rec = gt.get("recommendation", "request_revision")
        pred_rec = action.recommendation or ""
        reward = 0.4 if pred_rec == true_rec else -0.05

    elif action.action_type == ActionType.REQUEST_REVISION:
        true_rec = gt.get("recommendation", "request_revision")
        reward = 0.3 if true_rec == "request_revision" else -0.1

    elif action.action_type == ActionType.FINALIZE_REVIEW:
        # Bonus based on completeness
        total_issues = (
            len(gt.get("critical_issues", [])) +
            len(gt.get("high_issues", [])) +
            len(gt.get("medium_issues", []))
        )
        found_issues = len(state.issues_found)
        coverage = min(1.0, found_issues / max(1, total_issues))
        has_risk = bool(state.risk_assessments)
        has_rec = state.recommendation is not None
        reward = coverage * 0.5 + (0.25 if has_risk else 0) + (0.25 if has_rec else 0)

    elif action.action_type == ActionType.VIEW_CLAUSE:
        # Small exploration reward, but penalise repeated viewing
        views = sum(1 for a in state.actions_log
                    if a.get("action_type") == "view_clause"
                    and a.get("clause_id") == clause_id)
        reward = 0.02 if views <= 1 else -0.01

    return max(-1.0, min(1.0, reward))


# ─────────────────────────────────────────────
# Graders
# ─────────────────────────────────────────────

def _grader_base(state: ReviewState) -> Tuple[float, Dict]:
    task     = TASKS[state.task_id]
    contract = task["contract"]
    gt       = contract["ground_truth"]

    critical_ids = set(gt.get("critical_issues", []))
    high_ids     = set(gt.get("high_issues", []))
    medium_ids   = set(gt.get("medium_issues", []))
    all_issues   = critical_ids | high_ids | medium_ids

    found_ids = {i["clause_id"] for i in state.issues_found}

    found_critical = len(found_ids & critical_ids)
    found_high     = len(found_ids & high_ids)
    found_medium   = len(found_ids & medium_ids)

    # Issue detection score
    critical_score = found_critical / max(1, len(critical_ids))
    high_score     = found_high     / max(1, len(high_ids))     if high_ids else 1.0
    medium_score   = found_medium   / max(1, len(medium_ids))   if medium_ids else 1.0

    # False positives
    false_pos = len(found_ids - all_issues)
    fp_penalty = min(0.3, false_pos * 0.05)

    # Risk assessment
    true_risk = gt.get("overall_risk", "amber")
    risk_score = 0.0
    if state.risk_assessments:
        overall_pred = max(state.risk_assessments.values(),
                           key=lambda x: {"red": 2, "amber": 1, "green": 0}.get(x, 0))
        if overall_pred == true_risk:
            risk_score = 1.0
        elif (overall_pred == "amber" and true_risk == "red"):
            risk_score = 0.4
        else:
            risk_score = 0.1

    # Recommendation
    true_rec = gt.get("recommendation", "request_revision")
    rec_score = 1.0 if state.recommendation == true_rec else 0.0

    return (critical_score, high_score, medium_score,
            fp_penalty, risk_score, rec_score, found_ids, all_issues)


def grade_easy(state: ReviewState) -> Tuple[float, Dict]:
    (critical_score, high_score, medium_score,
     fp_penalty, risk_score, rec_score, found_ids, all_issues) = _grader_base(state)

    score = (
        0.55 * critical_score
        + 0.25 * high_score
        + 0.10 * risk_score
        + 0.10 * rec_score
        - fp_penalty
    )
    score = round(max(0.0, min(1.0, score)), 3)
    return score, {
        "critical_detection": round(critical_score, 3),
        "high_detection": round(high_score, 3),
        "risk_assessment": round(risk_score, 3),
        "recommendation": round(rec_score, 3),
        "false_positive_penalty": round(fp_penalty, 3),
        "final_score": score,
    }


def grade_medium(state: ReviewState) -> Tuple[float, Dict]:
    (critical_score, high_score, medium_score,
     fp_penalty, risk_score, rec_score, found_ids, all_issues) = _grader_base(state)

    score = (
        0.40 * critical_score
        + 0.25 * high_score
        + 0.10 * medium_score
        + 0.15 * risk_score
        + 0.10 * rec_score
        - fp_penalty
    )
    score = round(max(0.0, min(1.0, score)), 3)
    return score, {
        "critical_detection": round(critical_score, 3),
        "high_detection": round(high_score, 3),
        "medium_detection": round(medium_score, 3),
        "risk_assessment": round(risk_score, 3),
        "recommendation": round(rec_score, 3),
        "false_positive_penalty": round(fp_penalty, 3),
        "final_score": score,
    }


def grade_hard(state: ReviewState) -> Tuple[float, Dict]:
    (critical_score, high_score, medium_score,
     fp_penalty, risk_score, rec_score, found_ids, all_issues) = _grader_base(state)

    task   = TASKS[state.task_id]
    gt     = task["contract"]["ground_truth"]
    missing_ids = set(gt.get("missing_clauses", []))
    found_missing = len({i["clause_id"] for i in state.issues_found} & missing_ids)
    missing_score = found_missing / max(1, len(missing_ids)) if missing_ids else 1.0

    steps_used = state.step_count
    efficiency = max(0.0, 1.0 - max(0, steps_used - 25) / 20)

    score = (
        0.30 * critical_score
        + 0.20 * high_score
        + 0.10 * medium_score
        + 0.15 * missing_score
        + 0.15 * risk_score
        + 0.10 * rec_score
        - fp_penalty
    )
    score = round(max(0.0, min(1.0, score)), 3)
    return score, {
        "critical_detection": round(critical_score, 3),
        "high_detection": round(high_score, 3),
        "medium_detection": round(medium_score, 3),
        "missing_clause_detection": round(missing_score, 3),
        "risk_assessment": round(risk_score, 3),
        "recommendation": round(rec_score, 3),
        "false_positive_penalty": round(fp_penalty, 3),
        "steps_used": steps_used,
        "final_score": score,
    }


GRADERS = {
    "task_easy":   grade_easy,
    "task_medium": grade_medium,
    "task_hard":   grade_hard,
}


def grade(task_id: str, state: ReviewState) -> Tuple[float, Dict]:
    return GRADERS[task_id](state)


# ─────────────────────────────────────────────
# Core Environment Class
# ─────────────────────────────────────────────

class ContractReviewEnvironment:
    """
    Contract Review OpenEnv environment.
    Implements reset() / step() / state() per OpenEnv spec.
    """

    def __init__(self, task_id: str = "task_easy"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose: {list(TASKS)}")
        self.task_id  = task_id
        self._state: Optional[ReviewState] = None

    def reset(self) -> ContractObservation:
        task     = TASKS[self.task_id]
        contract = copy.deepcopy(task["contract"])
        self._state = ReviewState(
            episode_id=str(uuid.uuid4()),
            task_id=self.task_id,
            difficulty=task["difficulty"],
            contract=contract,
            ground_truth=contract["ground_truth"],
        )
        return self._build_observation()

    def step(self, action: ContractAction):
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode done. Call reset().")

        s = self._state
        s.step_count += 1
        result: Dict[str, Any] = {}
        success = True
        message = ""

        contract = s.contract
        clauses  = contract["clauses"]

        def get_clause(cid):
            return next((c for c in clauses if c["clause_id"] == cid), None)

        # ── Execute action ──────────────────────────────────────
        atype = action.action_type

        if atype == ActionType.VIEW_CLAUSE:
            clause = get_clause(action.clause_id)
            if clause:
                result["clause"] = clause
                message = f"Viewing clause '{clause['title']}'"
            else:
                success = False
                message = f"Clause {action.clause_id} not found"

        elif atype == ActionType.FLAG_ISSUE:
            clause = get_clause(action.clause_id)
            if not clause:
                success = False
                message = f"Clause {action.clause_id} not found"
            elif any(i["clause_id"] == action.clause_id for i in s.issues_found):
                success = False
                message = f"Issue already flagged for {action.clause_id}"
            else:
                issue = {
                    "clause_id": action.clause_id,
                    "description": action.issue_description or "",
                    "severity": action.severity or "medium",
                }
                s.issues_found.append(issue)
                clause["is_flagged"] = True
                if action.clause_id in s.ground_truth.get("critical_issues", []):
                    s.step_count  # just track
                message = f"Issue flagged on clause {action.clause_id}"

        elif atype == ActionType.APPROVE_CLAUSE:
            clause = get_clause(action.clause_id)
            if clause:
                clause["is_approved"] = True
                if action.clause_id not in s.approved_clauses:
                    s.approved_clauses.append(action.clause_id)
                message = f"Clause {action.clause_id} approved"
            else:
                success = False
                message = f"Clause {action.clause_id} not found"

        elif atype == ActionType.ASSESS_RISK:
            clause_id = action.clause_id or "overall"
            s.risk_assessments[clause_id] = action.risk_level or "amber"
            message = f"Risk assessed as {action.risk_level} for {clause_id}"

        elif atype == ActionType.ADD_COMMENT:
            result["comment_added"] = True
            message = f"Comment added to {action.clause_id}"

        elif atype == ActionType.RECOMMEND_ACTION:
            s.recommendation = action.recommendation
            message = f"Recommendation set: {action.recommendation}"

        elif atype == ActionType.REQUEST_REVISION:
            s.recommendation = "request_revision"
            message = "Revision requested"

        elif atype == ActionType.FINALIZE_REVIEW:
            s.done = True
            message = "Review finalised"

        # ── Compute reward ──────────────────────────────────────
        s.actions_log.append({
            "step": s.step_count,
            "action_type": atype,
            "clause_id": action.clause_id,
        })
        reward = compute_reward(action, s, result)
        s.cumulative_reward += reward

        # ── Check done ──────────────────────────────────────────
        max_steps = TASKS[self.task_id]["max_steps"]
        if s.step_count >= max_steps:
            s.done = True

        obs  = self._build_observation(message, success)
        info = {
            "step": s.step_count,
            "reward": reward,
            "cumulative_reward": s.cumulative_reward,
            "done": s.done,
        }
        return obs, reward, s.done, info

    def state(self) -> ReviewState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def _build_observation(
        self,
        last_action_result: str = "",
        last_action_success: bool = True,
    ) -> ContractObservation:
        s        = self._state
        task     = TASKS[s.task_id]
        contract = s.contract
        clauses  = [
            ClauseItem(
                clause_id=c["clause_id"],
                title=c["title"],
                text=c["text"] if not c.get("is_missing") else "[MISSING — not present in contract]",
                is_missing=c.get("is_missing", False),
                is_flagged=c.get("is_flagged", False),
                is_approved=c.get("is_approved", False),
            )
            for c in contract["clauses"]
        ]
        gt = contract["ground_truth"]
        return ContractObservation(
            contract_title=contract["title"],
            contract_type=contract["type"],
            clauses=clauses,
            step_number=s.step_count,
            max_steps=task["max_steps"],
            issues_found=s.issues_found,
            approved_clauses=s.approved_clauses,
            risk_assessments=s.risk_assessments,
            recommendation=s.recommendation,
            last_action_result=last_action_result,
            last_action_success=last_action_success,
            total_clauses=len(clauses),
            critical_issues_found=sum(
                1 for i in s.issues_found
                if i["clause_id"] in gt.get("critical_issues", [])
            ),
            done=s.done,
        )
