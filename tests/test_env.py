"""Tests for ContractReviewEnv — OpenEnv spec compliance."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from server.environment import (
    TASKS, ContractAction, ContractObservation,
    ContractReviewEnvironment, ReviewState,
    ActionType, ClauseSeverity, RiskLevel, grade,
)


@pytest.fixture
def easy_env():
    e = ContractReviewEnvironment("task_easy")
    return e

@pytest.fixture
def medium_env():
    e = ContractReviewEnvironment("task_medium")
    return e

@pytest.fixture
def hard_env():
    e = ContractReviewEnvironment("task_hard")
    return e


class TestReset:
    def test_returns_observation(self, easy_env):
        obs = easy_env.reset()
        assert isinstance(obs, ContractObservation)

    def test_step_zero(self, easy_env):
        obs = easy_env.reset()
        assert obs.step_number == 0

    def test_has_clauses(self, easy_env):
        obs = easy_env.reset()
        assert len(obs.clauses) > 0

    def test_is_deterministic(self, easy_env):
        obs1 = easy_env.reset()
        obs2 = easy_env.reset()
        assert [c.clause_id for c in obs1.clauses] == [c.clause_id for c in obs2.clauses]

    def test_clean_state(self, easy_env):
        easy_env.reset()
        easy_env.step(ContractAction(action_type=ActionType.VIEW_CLAUSE,
                                     clause_id=easy_env.state().contract["clauses"][0]["clause_id"]))
        obs2 = easy_env.reset()
        assert obs2.step_number == 0
        assert len(obs2.issues_found) == 0

    def test_all_tasks_reset(self):
        for tid in TASKS:
            env = ContractReviewEnvironment(tid)
            obs = env.reset()
            assert isinstance(obs, ContractObservation)
            assert obs.step_number == 0


class TestStep:
    def test_returns_correct_types(self, easy_env):
        easy_env.reset()
        cid = easy_env.state().contract["clauses"][0]["clause_id"]
        obs, reward, done, info = easy_env.step(
            ContractAction(action_type=ActionType.VIEW_CLAUSE, clause_id=cid)
        )
        assert isinstance(obs, ContractObservation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_reward_in_range(self, easy_env):
        easy_env.reset()
        for c in easy_env.state().contract["clauses"][:3]:
            cid = c["clause_id"]
            _, reward, done, _ = easy_env.step(
                ContractAction(action_type=ActionType.VIEW_CLAUSE, clause_id=cid)
            )
            assert -1.0 <= reward <= 1.0
            if done:
                break

    def test_step_increments_counter(self, easy_env):
        easy_env.reset()
        cid = easy_env.state().contract["clauses"][0]["clause_id"]
        for i in range(1, 4):
            obs, _, done, _ = easy_env.step(
                ContractAction(action_type=ActionType.VIEW_CLAUSE, clause_id=cid)
            )
            if done:
                break
            assert obs.step_number == i

    def test_flag_issue_updates_state(self, easy_env):
        easy_env.reset()
        # Find a clause with an issue
        issue_clause = next(
            c for c in easy_env.state().contract["clauses"]
            if c.get("_has_issue") or c.get("is_missing")
        )
        easy_env.step(ContractAction(
            action_type=ActionType.FLAG_ISSUE,
            clause_id=issue_clause["clause_id"],
            severity=ClauseSeverity.CRITICAL,
            issue_description="Test issue",
        ))
        assert len(easy_env.state().issues_found) == 1

    def test_approve_clause_updates_state(self, easy_env):
        easy_env.reset()
        # Find a clean clause
        clean_clause = next(
            c for c in easy_env.state().contract["clauses"]
            if not c.get("_has_issue") and not c.get("is_missing")
        )
        easy_env.step(ContractAction(
            action_type=ActionType.APPROVE_CLAUSE,
            clause_id=clean_clause["clause_id"],
        ))
        assert clean_clause["clause_id"] in easy_env.state().approved_clauses

    def test_finalize_ends_episode(self, easy_env):
        easy_env.reset()
        _, _, done, _ = easy_env.step(ContractAction(action_type=ActionType.FINALIZE_REVIEW))
        assert done is True

    def test_step_after_done_raises(self, easy_env):
        easy_env.reset()
        easy_env.step(ContractAction(action_type=ActionType.FINALIZE_REVIEW))
        with pytest.raises(RuntimeError):
            easy_env.step(ContractAction(action_type=ActionType.FINALIZE_REVIEW))


class TestState:
    def test_returns_review_state(self, easy_env):
        easy_env.reset()
        s = easy_env.state()
        assert isinstance(s, ReviewState)

    def test_without_reset_raises(self, easy_env):
        with pytest.raises(RuntimeError):
            easy_env.state()

    def test_ground_truth_not_in_observation(self, easy_env):
        obs = easy_env.reset()
        obs_dict = obs.model_dump()
        assert "ground_truth" not in obs_dict


class TestGraders:
    def test_score_in_range_all_tasks(self):
        for tid in TASKS:
            env = ContractReviewEnvironment(tid)
            env.reset()
            score, _ = grade(tid, env.state())
            assert 0.0 <= score <= 1.0, f"{tid}: {score}"

    def test_zero_agent_scores_zero(self):
        env = ContractReviewEnvironment("task_easy")
        env.reset()
        score, _ = grade("task_easy", env.state())
        assert score == 0.0

    def test_correct_flags_increase_score(self):
        env = ContractReviewEnvironment("task_easy")
        env.reset()
        gt = env.state().ground_truth

        for cid in gt.get("critical_issues", []):
            clause = next(c for c in env.state().contract["clauses"]
                          if c["clause_id"] == cid)
            sev = clause.get("_severity", "critical")
            env.step(ContractAction(
                action_type=ActionType.FLAG_ISSUE,
                clause_id=cid,
                severity=ClauseSeverity(sev),
                issue_description="Critical issue found",
            ))

        score_with_flags, _ = grade("task_easy", env.state())
        assert score_with_flags > 0.0

    def test_false_positives_reduce_score(self):
        env = ContractReviewEnvironment("task_easy")
        env.reset()
        clean_clause = next(
            c for c in env.state().contract["clauses"]
            if not c.get("_has_issue") and not c.get("is_missing")
        )
        env.step(ContractAction(
            action_type=ActionType.FLAG_ISSUE,
            clause_id=clean_clause["clause_id"],
            severity=ClauseSeverity.CRITICAL,
            issue_description="False positive",
        ))
        score, bd = grade("task_easy", env.state())
        assert bd["false_positive_penalty"] > 0

    def test_graders_deterministic(self):
        for tid in TASKS:
            e1 = ContractReviewEnvironment(tid)
            e1.reset()
            s1, _ = grade(tid, e1.state())
            e2 = ContractReviewEnvironment(tid)
            e2.reset()
            s2, _ = grade(tid, e2.state())
            assert s1 == s2

    def test_breakdown_has_final_score(self):
        for tid in TASKS:
            env = ContractReviewEnvironment(tid)
            env.reset()
            _, bd = grade(tid, env.state())
            assert "final_score" in bd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
