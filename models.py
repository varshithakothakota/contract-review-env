"""
Typed models for ContractReviewEnv.
Follows OpenEnv Action / Observation / State pattern.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


class RiskLevel(str, Enum):
    RED    = "red"
    AMBER  = "amber"
    GREEN  = "green"


class ActionType(str, Enum):
    READ_CLAUSE      = "read_clause"
    FLAG_RISK        = "flag_risk"
    MARK_COMPLIANT   = "mark_compliant"
    REQUEST_REVISION = "request_revision"
    SET_RISK_RATING  = "set_risk_rating"
    RECOMMEND        = "recommend"
    SUBMIT_REVIEW    = "submit_review"


class ContractAction(BaseModel):
    """Action the agent takes during contract review."""
    action_type:   ActionType
    clause_id:     Optional[str]       = None
    severity:      Optional[Severity]  = None
    risk_level:    Optional[RiskLevel] = None
    finding:       Optional[str]       = None   # description of issue found
    recommendation: Optional[str]      = None   # approve | revise | escalate | reject

    class Config:
        use_enum_values = True


class ClauseView(BaseModel):
    """A single contract clause as seen by the agent."""
    clause_id:   str
    title:       str
    text:        str
    status:      str = "unreviewed"   # unreviewed | flagged | compliant | missing


class FindingRecord(BaseModel):
    clause_id:   str
    severity:    str
    finding:     str


class ContractObservation(BaseModel):
    """What the agent observes at each step."""
    episode_id:     str
    task_id:        str
    contract_type:  str
    contract_title: str

    clauses:        List[ClauseView]
    findings:       List[FindingRecord] = Field(default_factory=list)
    compliant_ids:  List[str]           = Field(default_factory=list)

    risk_rating:    Optional[str]  = None
    recommendation: Optional[str] = None

    step:           int  = 0
    max_steps:      int  = 30
    done:           bool = False

    last_action_ok:     bool = True
    last_action_msg:    str  = ""
    current_score:      float = 0.0
    available_actions:  List[str] = Field(default_factory=list)


class EpisodeState(BaseModel):
    """Full internal state — returned by GET /state."""
    episode_id:   str
    task_id:      str
    step:         int
    done:         bool
    cumulative_reward: float
    findings:     List[Dict[str, Any]]
    compliant_ids: List[str]
    risk_rating:  Optional[str]
    recommendation: Optional[str]
    actions_taken: List[Dict[str, Any]]
