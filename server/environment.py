"""
ContractReviewEnvironment — server-side logic.

Key design decisions:
  - Clause texts use REALISTIC legal language (no "[MISSING CLAUSE]" hints)
  - Issues require reasoning to detect, not keyword matching
  - Reward provides dense partial-credit signal at every step
  - Grader is multi-component with strict severity accuracy requirement
  - Baseline heuristic scores ~0.35-0.55 (leaves room for LLM improvement)
"""
from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple

# Models are defined inline to keep server package self-contained
# (avoids import issues when running in-process from inference.py)


# ─────────────────────────────────────────────────────────────────────────────
# Contract data — realistic language, issues require legal reasoning
# ─────────────────────────────────────────────────────────────────────────────

_NDA_CLAUSES = [
    {
        "clause_id": "n01", "title": "Parties and Recitals",
        "text": (
            "This Non-Disclosure Agreement ('Agreement') is entered into as of the Effective Date "
            "between Acme Technologies Inc., a Delaware corporation ('Disclosing Party'), "
            "and Beta Consulting LLC, a California limited liability company ('Receiving Party')."
        ),
        "_clean": True,
    },
    {
        "clause_id": "n02", "title": "Definition of Confidential Information",
        "text": (
            "'Confidential Information' means any information disclosed by the Disclosing Party "
            "to the Receiving Party, either directly or indirectly, in writing, orally or by "
            "inspection of tangible objects, that is designated as 'Confidential' or that "
            "reasonably should be understood to be confidential given the nature of the information."
        ),
        "_clean": True,
    },
    {
        "clause_id": "n03", "title": "Exclusions from Confidential Information",
        "text": (
            "The obligations of confidentiality shall not apply to information that: "
            "(a) was publicly known prior to disclosure; (b) becomes publicly known through "
            "no breach by Receiving Party; or (c) was rightfully received from a third party "
            "without restriction. Receiving Party bears the burden of proving any exclusion applies."
        ),
        "_clean": True,
    },
    {
        "clause_id": "n04", "title": "Obligations of Receiving Party",
        "text": (
            "The Receiving Party agrees to hold Confidential Information in strict confidence "
            "and to take all reasonable precautions to protect such information. "
            "The Receiving Party may disclose Confidential Information solely to its employees "
            "who have a need to know such information for the purpose of evaluating a potential "
            "business relationship between the parties."
        ),
        "_clean": True,
    },
    {
        "clause_id": "n05", "title": "Term and Duration",
        "text": (
            "This Agreement shall be effective as of the date first written above and "
            "shall continue in full force and effect until terminated by either party "
            "upon thirty (30) days' written notice to the other party. "
            "Confidentiality obligations shall survive termination of this Agreement."
        ),
        # ISSUE: 30-day termination notice is far too short for NDA;
        # also no fixed minimum term — can be terminated immediately after 30 days
        "_has_issue": True, "_severity": "high",
        "_issue": (
            "Termination on 30 days notice with no minimum term is commercially unreasonable "
            "for an NDA; the agreement can be ended before confidential discussions conclude"
        ),
    },
    {
        "clause_id": "n06", "title": "Return and Destruction of Materials",
        "text": (
            "At the Disclosing Party's written request, the Receiving Party shall promptly "
            "return or destroy all tangible materials containing Confidential Information. "
            "The Receiving Party shall certify in writing the completion of such destruction "
            "within five (5) business days of the request."
        ),
        "_clean": True,
    },
    {
        "clause_id": "n07", "title": "No Licence Granted",
        "text": (
            "Nothing in this Agreement shall be construed as granting any rights, by licence "
            "or otherwise, to any Confidential Information, or to any invention or patent, "
            "copyright, trade secret, or other intellectual property right of the Disclosing Party."
        ),
        "_clean": True,
    },
    {
        "clause_id": "n08", "title": "Remedies",
        "text": (
            "Each party acknowledges that any breach of this Agreement may cause irreparable harm "
            "for which monetary damages may be an inadequate remedy. Accordingly, each party "
            "agrees that equitable relief, including injunction and specific performance, "
            "is an appropriate remedy for any actual or threatened breach. "
            "Pursuit of equitable relief shall not limit any other rights or remedies available."
        ),
        "_clean": True,
    },
    {
        "clause_id": "n09", "title": "Governing Law",
        "text": (
            "This Agreement shall be governed by and construed in accordance with the laws "
            "of the State of Delaware, without regard to conflicts of law principles. "
            "Any disputes shall be resolved in the federal or state courts located in "
            "New Castle County, Delaware."
        ),
        "_clean": True,
    },
    {
        "clause_id": "n10", "title": "Entire Agreement",
        "text": (
            "This Agreement constitutes the entire agreement between the parties with respect "
            "to the subject matter hereof and supersedes all prior agreements and understandings, "
            "written or oral, relating to the subject matter hereof."
        ),
        "_clean": True,
    },
    # MISSING: No injunctive relief clause — wait, that's n08 above
    # ACTUALLY MISSING: No specific penalty/liquidated damages clause
    # and critically — no mutual reciprocity clause (currently one-sided)
    {
        "clause_id": "n11", "title": "Mutual Obligations",
        "text": (
            "The obligations set forth in Section 4 (Obligations of Receiving Party) apply "
            "solely to the Receiving Party and do not apply to information disclosed by "
            "the Receiving Party to the Disclosing Party in the course of their discussions."
        ),
        # CRITICAL: This makes NDA one-sided only. In a mutual NDA both parties
        # should have identical obligations. This clause explicitly makes it one-sided.
        "_has_issue": True, "_severity": "critical",
        "_issue": (
            "Agreement purports to be mutual but this clause makes confidentiality "
            "obligations one-sided — only the Receiving Party is bound; "
            "Disclosing Party can freely share anything the Receiving Party discloses"
        ),
    },
]

_NDA_GT = {
    "critical_issues": ["n11"],
    "high_issues":     ["n05"],
    "medium_issues":   [],
    "clean_clauses":   ["n01","n02","n03","n04","n06","n07","n08","n09","n10"],
    "overall_risk":    "amber",
    "recommendation":  "revise",
}

# ── Software Services Agreement ──────────────────────────────────────────────

_SSA_CLAUSES = [
    {
        "clause_id": "a01", "title": "Services",
        "text": (
            "Provider shall perform the software development and consulting services "
            "described in Schedule A ('Services'). Provider shall assign qualified personnel "
            "to perform the Services and may use subcontractors with Client's prior written consent."
        ),
        "_clean": True,
    },
    {
        "clause_id": "a02", "title": "Fees and Invoicing",
        "text": (
            "Client shall pay Provider the fees set forth in Schedule B. Provider shall invoice "
            "Client monthly in arrears. Payment shall be due within ninety (90) calendar days "
            "of invoice receipt. Undisputed amounts outstanding beyond the payment period "
            "shall accrue interest at the rate of one and one-half percent (1.5%) per month."
        ),
        # ISSUE: 90-day payment terms are highly unfavourable to Provider
        # (industry standard is net-30). 1.5%/month (18% APR) interest is high.
        "_has_issue": True, "_severity": "high",
        "_issue": (
            "Net-90 payment terms are three times longer than the industry standard of net-30, "
            "creating severe cash flow risk for Provider; 18% APR interest on late payments "
            "may be unenforceable in some jurisdictions"
        ),
    },
    {
        "clause_id": "a03", "title": "Intellectual Property Ownership",
        "text": (
            "All work product, inventions, developments, improvements, and other materials "
            "created, conceived, or developed by Provider in connection with the Services "
            "('Work Product') shall be the sole and exclusive property of Client. "
            "Provider hereby irrevocably assigns to Client all right, title, and interest "
            "in and to all Work Product, including all intellectual property rights therein."
        ),
        # CRITICAL: Blanket IP assignment with no carve-out for pre-existing IP.
        # Provider loses all rights to tools, frameworks, and methodologies developed
        # independently. This is a major negotiating red flag.
        "_has_issue": True, "_severity": "critical",
        "_issue": (
            "Blanket IP assignment to Client with no carve-out for Provider's pre-existing "
            "IP, proprietary tools, or general know-how; Provider loses rights to independently "
            "developed methodologies and reusable components used across multiple clients"
        ),
    },
    {
        "clause_id": "a04", "title": "Representations and Warranties",
        "text": (
            "Provider represents and warrants that: (a) it has the right to enter into "
            "this Agreement; (b) the Services will be performed in a professional and "
            "workmanlike manner consistent with industry standards; and (c) the Work Product "
            "will not infringe any third-party intellectual property rights to Provider's knowledge."
        ),
        # MEDIUM: "to Provider's knowledge" qualifier significantly weakens the IP warranty
        "_has_issue": True, "_severity": "medium",
        "_issue": (
            "IP non-infringement warranty is qualified by 'to Provider's knowledge', "
            "which substantially weakens the warranty and may leave Client unprotected "
            "against third-party IP claims"
        ),
    },
    {
        "clause_id": "a05", "title": "Confidentiality",
        "text": (
            "Each party agrees to maintain the confidentiality of the other party's "
            "Confidential Information and not to disclose such information to any third party "
            "without the prior written consent of the disclosing party. "
            "These confidentiality obligations shall survive termination of this Agreement "
            "for a period of twelve (12) months."
        ),
        # MEDIUM: 12-month post-termination confidentiality is too short
        "_has_issue": True, "_severity": "medium",
        "_issue": (
            "Post-termination confidentiality of only 12 months is inadequate for a "
            "software services engagement where proprietary technical architecture "
            "and business logic may remain competitively sensitive for years"
        ),
    },
    {
        "clause_id": "a06", "title": "Term and Termination",
        "text": (
            "This Agreement shall commence on the Effective Date and continue for an initial "
            "term of one (1) year, automatically renewing for successive one-year periods "
            "unless terminated by either party upon seven (7) days' prior written notice. "
            "Client may terminate for convenience at any time upon such notice."
        ),
        # HIGH: 7-day notice period is commercially unreasonable
        "_has_issue": True, "_severity": "high",
        "_issue": (
            "Seven-day termination notice is commercially unreasonable — industry standard "
            "is 30-90 days. Provider cannot wind down work, transition staff, or protect "
            "revenue with one week's notice. Provider has no reciprocal for-cause termination right."
        ),
    },
    {
        "clause_id": "a07", "title": "Dispute Resolution",
        "text": (
            "Any dispute arising under or relating to this Agreement shall be submitted "
            "to binding arbitration conducted in Client's principal place of business "
            "in accordance with the rules of the American Arbitration Association. "
            "The arbitration shall be conducted in the English language. "
            "The award of the arbitrator shall be final and binding."
        ),
        # MEDIUM: Arbitration venue mandated at Client's location is biased
        "_has_issue": True, "_severity": "medium",
        "_issue": (
            "Mandating arbitration at Client's principal place of business creates "
            "geographic disadvantage for Provider; a neutral venue or virtual arbitration "
            "is standard practice"
        ),
    },
    {
        "clause_id": "a08", "title": "Data Protection and Privacy",
        "text": (
            "Each party shall comply with applicable data protection laws in relation "
            "to any personal data processed in connection with this Agreement."
        ),
        # CRITICAL: This clause is dangerously vague. No mention of GDPR, CCPA,
        # data processing agreements, sub-processor obligations, breach notification,
        # data subject rights, or data transfer mechanisms.
        "_has_issue": True, "_severity": "critical",
        "_issue": (
            "Data protection clause is dangerously vague — no specification of applicable "
            "regulations (GDPR, CCPA), no Data Processing Agreement, no data breach "
            "notification obligations, no data subject rights procedures, "
            "and no cross-border transfer mechanisms"
        ),
    },
    {
        "clause_id": "a09", "title": "Governing Law",
        "text": (
            "This Agreement shall be governed by and construed in accordance with the laws "
            "of the State of New York, without regard to its conflict of law provisions."
        ),
        "_clean": True,
    },
    # MISSING CLAUSE: No Limitation of Liability
    # (omitted — not stated anywhere in the contract)
    {
        "clause_id": "a10", "title": "General Provisions",
        "text": (
            "This Agreement may be amended only by a written instrument signed by both parties. "
            "If any provision is found to be unenforceable, the remaining provisions shall "
            "continue in full force and effect. This Agreement may be executed in counterparts, "
            "each of which shall be deemed an original."
        ),
        # CRITICAL: No limitation of liability clause anywhere in this contract.
        # This general provisions clause does NOT address liability caps.
        # Provider is exposed to unlimited consequential damages.
        "_has_issue": True, "_severity": "critical",
        "_issue": (
            "The contract contains no limitation of liability clause anywhere. "
            "Provider is exposed to unlimited consequential, indirect, and special damages. "
            "This is one of the most dangerous omissions in a commercial services agreement."
        ),
    },
]

_SSA_GT = {
    "critical_issues": ["a03", "a08", "a10"],
    "high_issues":     ["a02", "a06"],
    "medium_issues":   ["a04", "a05", "a07"],
    "clean_clauses":   ["a01", "a09"],
    "overall_risk":    "red",
    "recommendation":  "revise",
}

# ── Enterprise SaaS License Agreement ────────────────────────────────────────

_SAAS_CLAUSES = [
    {
        "clause_id": "e01", "title": "Grant of License",
        "text": (
            "Subject to the terms of this Agreement, Licensor grants Licensee a limited, "
            "non-exclusive, non-transferable, non-sublicensable license to access and use "
            "the Software solely for Licensee's internal business purposes during the Term."
        ),
        "_clean": True,
    },
    {
        "clause_id": "e02", "title": "Subscription Fees and Adjustments",
        "text": (
            "Licensee shall pay the annual subscription fee of USD 240,000. "
            "Licensor reserves the right to adjust subscription fees upon each renewal "
            "by providing Licensee with thirty (30) days' advance written notice. "
            "Fee adjustments are not subject to any cap or limitation."
        ),
        # HIGH: Uncapped fee increases with only 30 days notice is unreasonable
        "_has_issue": True, "_severity": "high",
        "_issue": (
            "Licensor can increase fees by any amount with only 30 days notice, "
            "with no cap whatsoever. Enterprise contracts typically cap annual increases "
            "at CPI or 5-7%, and require 90-180 days notice for material changes."
        ),
    },
    {
        "clause_id": "e03", "title": "Service Availability",
        "text": (
            "Licensor shall use commercially reasonable efforts to make the Software "
            "available ninety-nine percent (99%) of the time, measured monthly, "
            "excluding scheduled maintenance. In the event of downtime exceeding the "
            "availability commitment, Licensee's sole remedy shall be a service credit "
            "equal to five percent (5%) of the monthly subscription fee per hour of excess downtime, "
            "up to a maximum of one month's subscription fee."
        ),
        # HIGH: 99% uptime = ~7.3 hrs/month downtime for enterprise SaaS.
        # 5% per hour credit is very low. Credit-only remedy removes right to terminate or claim damages.
        "_has_issue": True, "_severity": "high",
        "_issue": (
            "99% uptime permits ~7.3 hours of downtime per month, insufficient for enterprise. "
            "Credits-only remedy removes Licensee's right to terminate or claim actual damages. "
            "Credit cap of one month's fee provides no meaningful compensation for prolonged outages."
        ),
    },
    {
        "clause_id": "e04", "title": "Data Ownership",
        "text": (
            "As between the parties, Licensee retains all ownership rights to data "
            "uploaded to or generated by Licensee's use of the Software ('Licensee Data'). "
            "Licensor may use aggregated, anonymised data derived from Licensee Data "
            "for product improvement and analytical purposes."
        ),
        # MEDIUM: No definition of "anonymised", no opt-out, no audit right on data usage
        "_has_issue": True, "_severity": "medium",
        "_issue": (
            "Licensor's right to use 'anonymised' data is undefined — no standard referenced "
            "(ISO 29101, k-anonymity), no opt-out mechanism, and no audit right for Licensee "
            "to verify that proper anonymisation is applied before use"
        ),
    },
    {
        "clause_id": "e05", "title": "Security Standards",
        "text": (
            "Licensor represents that it maintains industry-standard security practices "
            "in connection with the Software and Licensee Data. "
            "Licensor will notify Licensee of any confirmed data breach affecting "
            "Licensee Data within seventy-two (72) hours of discovery."
        ),
        # CRITICAL: "Industry-standard" is undefined — no specific certifications required
        # (SOC 2 Type II, ISO 27001, PCI-DSS). 72-hour notification is acceptable but
        # overall clause is dangerously vague for enterprise SaaS.
        "_has_issue": True, "_severity": "critical",
        "_issue": (
            "Security clause commits only to undefined 'industry-standard practices' — "
            "no specific certifications required or represented (SOC 2 Type II, ISO 27001). "
            "Enterprise SaaS must contractually commit to specific standards with audit rights."
        ),
    },
    {
        "clause_id": "e06", "title": "Indemnification",
        "text": (
            "Licensee shall defend, indemnify, and hold harmless Licensor from and against "
            "any claims, damages, losses, and expenses arising from Licensee's use of the Software "
            "or breach of this Agreement. Licensor's obligation to indemnify Licensee is limited "
            "to claims that the Software, as provided by Licensor, infringes a valid patent, "
            "copyright, or trademark of a third party, provided Licensee promptly notifies "
            "Licensor and cooperates fully in the defence."
        ),
        # CRITICAL: Asymmetric indemnification heavily favouring Licensor.
        # Licensee has broad unlimited indemnification obligation; Licensor's IP indemnity
        # is narrow and subject to conditions.
        "_has_issue": True, "_severity": "critical",
        "_issue": (
            "Severely asymmetric indemnification: Licensee's obligation is broad and unconditional "
            "while Licensor's IP indemnity is narrow, conditional, and likely inadequate. "
            "Licensor should bear full IP indemnity for its own software without conditions "
            "not within Licensee's control."
        ),
    },
    {
        "clause_id": "e07", "title": "Limitation of Liability",
        "text": (
            "In no event shall either party be liable for any indirect, incidental, "
            "special, exemplary, or consequential damages. Each party's total liability "
            "shall not exceed the fees paid by Licensee in the twelve (12) months "
            "preceding the claim."
        ),
        "_clean": True,
    },
    {
        "clause_id": "e08", "title": "Term and Renewal",
        "text": (
            "This Agreement shall commence on the Effective Date and continue for an initial "
            "term of one (1) year ('Initial Term'). Thereafter, this Agreement shall "
            "automatically renew for successive one-year terms unless either party provides "
            "written notice of non-renewal at least ninety (90) days prior to the end of "
            "the then-current term."
        ),
        "_clean": True,
    },
    {
        "clause_id": "e09", "title": "Termination for Cause",
        "text": (
            "Either party may terminate this Agreement for material breach upon thirty (30) "
            "days' written notice, provided the breaching party fails to cure such breach "
            "within the notice period. Either party may terminate immediately upon the "
            "other party's insolvency or bankruptcy."
        ),
        "_clean": True,
    },
    {
        "clause_id": "e10", "title": "Effect of Termination",
        "text": (
            "Upon expiration or termination, Licensee's access to the Software shall "
            "immediately cease. Licensor shall retain Licensee Data for thirty (30) days "
            "following termination, after which it may be permanently deleted. "
            "Licensee is responsible for exporting its data prior to termination."
        ),
        # HIGH: 30-day data retention is very short. No guaranteed export format.
        # No data portability mechanism described. Vendor lock-in risk.
        "_has_issue": True, "_severity": "high",
        "_issue": (
            "Thirty-day data retention post-termination is insufficient for complex enterprise "
            "data migrations. No guaranteed export format or portability mechanism is specified. "
            "Licensor should provide structured data export (CSV/JSON) for at least 90 days."
        ),
    },
    {
        "clause_id": "e11", "title": "Audit Rights",
        "text": (
            "Licensor shall have the right to audit Licensee's use of the Software "
            "upon thirty (30) days' written notice, no more than once per calendar year, "
            "to verify compliance with this Agreement. Audit costs shall be borne by Licensor "
            "unless a material compliance failure is discovered."
        ),
        # MEDIUM: Audit rights are one-directional — only Licensor can audit Licensee.
        # Licensee has no audit right to verify Licensor's security practices or data handling.
        "_has_issue": True, "_severity": "medium",
        "_issue": (
            "Audit rights are unilateral — Licensor can audit Licensee's usage compliance, "
            "but Licensee has no reciprocal right to audit Licensor's security controls, "
            "data processing practices, or compliance certifications"
        ),
    },
]

_SAAS_GT = {
    "critical_issues": ["e05", "e06"],
    "high_issues":     ["e02", "e03", "e10"],
    "medium_issues":   ["e04", "e11"],
    "clean_clauses":   ["e01", "e07", "e08", "e09"],
    "overall_risk":    "red",
    "recommendation":  "revise",
}

# ─────────────────────────────────────────────────────────────────────────────
# Task registry
# ─────────────────────────────────────────────────────────────────────────────

TASKS: Dict[str, Dict] = {
    "task_easy": {
        "description": (
            "Review a 11-clause Non-Disclosure Agreement. Identify 1 critical issue "
            "(one-sided obligations) and 1 high issue (inadequate term protection). "
            "Issues are embedded in formal legal language requiring careful reading."
        ),
        "difficulty":  "easy",
        "max_steps":   30,
        "clauses":     _NDA_CLAUSES,
        "ground_truth": _NDA_GT,
    },
    "task_medium": {
        "description": (
            "Review a 10-clause Software Services Agreement with 3 critical issues, "
            "2 high, and 3 medium issues. Issues span IP assignment, data protection gaps, "
            "missing liability caps, and unfavourable commercial terms."
        ),
        "difficulty":  "medium",
        "max_steps":   40,
        "clauses":     _SSA_CLAUSES,
        "ground_truth": _SSA_GT,
    },
    "task_hard": {
        "description": (
            "Review a 11-clause Enterprise SaaS License Agreement. Identify 2 critical, "
            "3 high, and 2 medium issues embedded in complex legal language. "
            "Issues include asymmetric indemnification, vague security standards, "
            "and data portability gaps — all requiring expert legal reasoning."
        ),
        "difficulty":  "hard",
        "max_steps":   50,
        "clauses":     _SAAS_CLAUSES,
        "ground_truth": _SAAS_GT,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Reward calculator
# ─────────────────────────────────────────────────────────────────────────────

_SEV_SCORE = {"critical": 3, "high": 2, "medium": 1, "low": 0}


def _severity_match_score(predicted: str, true_sev: str) -> float:
    dist = abs(_SEV_SCORE.get(predicted, 0) - _SEV_SCORE.get(true_sev, 0))
    return max(0.0, 1.0 - dist * 0.4)


def compute_step_reward(action: Dict, state: Dict) -> Tuple[float, str]:
    """
    Dense per-step reward. Returns (reward, explanation).
    
    Design:
    - Correctly flagging a critical issue: up to +0.8 (base + severity accuracy bonus)
    - Correctly flagging a high issue: up to +0.5
    - Correctly flagging a medium issue: up to +0.3
    - False positive (flagging a clean clause): -0.2
    - Approving a problematic clause: -0.25 (worse than false positive)
    - Marking a clean clause compliant: +0.08
    - Correct risk rating: +0.35
    - Correct recommendation: +0.35
    - Finalising with good coverage: +0.2 to +0.5
    - Repeated read of same clause: -0.03
    """
    task    = TASKS[state["task_id"]]
    gt      = task["ground_truth"]
    atype   = action.get("action_type", "")
    cid     = action.get("clause_id")
    clause  = next((c for c in task["clauses"] if c.get("clause_id") == cid), None)

    crit    = set(gt["critical_issues"])
    high    = set(gt.get("high_issues", []))
    medium  = set(gt.get("medium_issues", []))
    clean   = set(gt.get("clean_clauses", []))
    all_iss = crit | high | medium

    if atype == "flag_risk":
        if not clause:
            return -0.1, f"Clause {cid} not found"
        if cid in clean:
            return -0.2, f"False positive: {cid} is a clean clause"
        if cid not in all_iss:
            return -0.1, f"Clause {cid} has no graded issue"

        # Already flagged?
        if any(f["clause_id"] == cid for f in state["findings"]):
            return -0.05, f"Clause {cid} already flagged"

        pred_sev = action.get("severity", "medium")
        true_sev = clause.get("_severity", "medium")
        sev_sc   = _severity_match_score(pred_sev, true_sev)

        if cid in crit:
            base = 0.55
        elif cid in high:
            base = 0.35
        else:
            base = 0.18

        reward = base + base * 0.45 * sev_sc
        return round(reward, 3), (
            f"Flagged {cid} ({pred_sev} vs true={true_sev}, "
            f"sev_match={sev_sc:.2f})"
        )

    if atype == "mark_compliant":
        if not clause:
            return -0.1, f"Clause {cid} not found"
        if cid in all_iss:
            return -0.25, f"Approved problematic clause {cid}"
        return 0.08, f"Correctly marked {cid} as compliant"

    if atype == "set_risk_rating":
        true_risk = gt["overall_risk"]
        pred_risk = action.get("risk_level", "")
        if pred_risk == true_risk:
            return 0.35, f"Correct risk rating: {pred_risk}"
        map_ = {"red": 2, "amber": 1, "green": 0}
        dist = abs(map_.get(pred_risk, 0) - map_.get(true_risk, 0))
        return round(max(-0.1, 0.15 - 0.1 * dist), 3), (
            f"Risk rating off: {pred_risk} vs {true_risk}"
        )

    if atype == "recommend":
        true_rec = gt["recommendation"]
        pred_rec = action.get("recommendation", "")
        return (0.35, f"Correct recommendation: {pred_rec}") \
               if pred_rec == true_rec else \
               (-0.05, f"Wrong recommendation: {pred_rec} vs {true_rec}")

    if atype == "request_revision":
        true_rec = gt["recommendation"]
        return (0.3, "Correct: revision requested") \
               if true_rec == "revise" else (-0.05, "Revision not appropriate")

    if atype == "submit_review":
        total   = len(all_iss)
        found   = {f["clause_id"] for f in state["findings"]}
        cov     = len(found & all_iss) / max(1, total)
        has_risk = state.get("risk_rating") is not None
        has_rec  = state.get("recommendation") is not None
        bonus    = cov * 0.35 + (0.1 if has_risk else 0) + (0.1 if has_rec else 0)
        return round(bonus, 3), f"Submission coverage={cov:.2f}"

    if atype == "read_clause":
        reads = sum(1 for a in state["actions_taken"]
                    if a.get("action_type") == "read_clause"
                    and a.get("clause_id") == cid)
        return (0.01, f"Reading {cid}") if reads <= 1 else \
               (-0.03, f"Repeated read of {cid}")

    return 0.0, "No signal"


# ─────────────────────────────────────────────────────────────────────────────
# Graders
# ─────────────────────────────────────────────────────────────────────────────

def _base_components(state: Dict, task_id: str) -> Dict:
    task    = TASKS[task_id]
    gt      = task["ground_truth"]
    crit    = set(gt["critical_issues"])
    high    = set(gt.get("high_issues", []))
    medium  = set(gt.get("medium_issues", []))
    clean   = set(gt.get("clean_clauses", []))

    findings = state.get("findings", [])
    found_ids = {f["clause_id"] for f in findings}

    # Detection
    c_det  = len(found_ids & crit)  / max(1, len(crit))
    h_det  = len(found_ids & high)  / max(1, len(high))  if high   else 1.0
    m_det  = len(found_ids & medium)/ max(1, len(medium)) if medium else 1.0

    # Severity accuracy (partial credit)
    sev_scores = []
    for f in findings:
        cid = f["clause_id"]
        clause = next((c for c in task["clauses"] if c.get("clause_id") == cid), None)
        if clause and clause.get("_severity"):
            sev_scores.append(_severity_match_score(f["severity"], clause["_severity"]))
    sev_acc = sum(sev_scores) / max(1, len(sev_scores)) if sev_scores else 0.0

    # False positives
    all_iss = crit | high | medium
    fp      = len(found_ids - all_iss)
    fp_pen  = min(0.25, fp * 0.08)

    # Wrong approvals
    comp_ids = set(state.get("compliant_ids", []))
    wrong_app = len(comp_ids & all_iss)
    wa_pen   = min(0.2, wrong_app * 0.1)

    # Risk rating
    true_risk = gt["overall_risk"]
    pred_risk = state.get("risk_rating")
    r_map     = {"red": 2, "amber": 1, "green": 0}
    if pred_risk == true_risk:
        risk_sc = 1.0
    elif pred_risk and abs(r_map.get(pred_risk,0) - r_map.get(true_risk,0)) == 1:
        risk_sc = 0.35
    else:
        risk_sc = 0.0

    # Recommendation
    rec_sc = 1.0 if state.get("recommendation") == gt["recommendation"] else 0.0

    return {
        "c_det": c_det, "h_det": h_det, "m_det": m_det,
        "sev_acc": sev_acc, "fp_pen": fp_pen, "wa_pen": wa_pen,
        "risk_sc": risk_sc, "rec_sc": rec_sc,
    }


def grade_task(task_id: str, state: Dict) -> Tuple[float, Dict]:
    c = _base_components(state, task_id)

    if task_id == "task_easy":
        raw = (0.45 * c["c_det"]
             + 0.20 * c["h_det"]
             + 0.15 * c["sev_acc"]
             + 0.10 * c["risk_sc"]
             + 0.10 * c["rec_sc"]
             - c["fp_pen"]
             - c["wa_pen"])
    elif task_id == "task_medium":
        raw = (0.35 * c["c_det"]
             + 0.20 * c["h_det"]
             + 0.10 * c["m_det"]
             + 0.15 * c["sev_acc"]
             + 0.10 * c["risk_sc"]
             + 0.10 * c["rec_sc"]
             - c["fp_pen"]
             - c["wa_pen"])
    else:  # hard
        raw = (0.30 * c["c_det"]
             + 0.22 * c["h_det"]
             + 0.10 * c["m_det"]
             + 0.15 * c["sev_acc"]
             + 0.12 * c["risk_sc"]
             + 0.11 * c["rec_sc"]
             - c["fp_pen"]
             - c["wa_pen"])

    score = round(max(0.0, min(1.0, raw)), 3)
    return score, {
        "critical_recall": round(c["c_det"], 3),
        "high_recall":     round(c["h_det"], 3),
        "medium_recall":   round(c["m_det"], 3),
        "severity_accuracy": round(c["sev_acc"], 3),
        "false_positive_penalty": round(c["fp_pen"], 3),
        "wrong_approval_penalty": round(c["wa_pen"], 3),
        "risk_rating_score": round(c["risk_sc"], 3),
        "recommendation_score": round(c["rec_sc"], 3),
        "final_score": score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class ContractReviewEnvironment:
    """
    OpenEnv environment for legal contract review.
    Implements reset() / step() / state() per spec.
    """

    def __init__(self, task_id: str = "task_easy"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}. Choose: {list(TASKS)}")
        self.task_id = task_id
        self._state: Optional[Dict] = None

    def reset(self) -> Dict:
        task = TASKS[self.task_id]
        self._state = {
            "episode_id":      str(uuid.uuid4()),
            "task_id":         self.task_id,
            "step":            0,
            "done":            False,
            "findings":        [],
            "compliant_ids":   [],
            "risk_rating":     None,
            "recommendation":  None,
            "actions_taken":   [],
            "cumulative_reward": 0.0,
            "clauses":         copy.deepcopy(task["clauses"]),
        }
        return self._build_obs()

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        if self._state["done"]:
            raise RuntimeError("Episode is done. Call reset().")

        s     = self._state
        atype = action.get("action_type", "")
        cid   = action.get("clause_id")
        clause = next((c for c in s["clauses"] if c.get("clause_id") == cid), None)
        s["step"] += 1

        success = True
        msg     = ""

        if atype == "flag_risk":
            if not clause:
                success, msg = False, f"Clause {cid} not found"
            elif any(f["clause_id"] == cid for f in s["findings"]):
                success, msg = False, f"Clause {cid} already flagged"
            else:
                s["findings"].append({
                    "clause_id": cid,
                    "severity":  action.get("severity", "medium"),
                    "finding":   action.get("finding", ""),
                })
                clause["status"] = "flagged"
                msg = f"Risk flagged on {cid} ({action.get('severity','medium')})"

        elif atype == "mark_compliant":
            if not clause:
                success, msg = False, f"Clause {cid} not found"
            else:
                clause["status"] = "compliant"
                if cid not in s["compliant_ids"]:
                    s["compliant_ids"].append(cid)
                msg = f"Marked {cid} compliant"

        elif atype == "set_risk_rating":
            s["risk_rating"] = action.get("risk_level")
            msg = f"Risk rating: {s['risk_rating']}"

        elif atype == "recommend":
            s["recommendation"] = action.get("recommendation")
            msg = f"Recommendation: {s['recommendation']}"

        elif atype == "request_revision":
            s["recommendation"] = "revise"
            msg = "Revision requested"

        elif atype == "submit_review":
            s["done"] = True
            msg = "Review submitted"

        elif atype == "read_clause":
            msg = f"Reading clause {cid}"

        s["actions_taken"].append({
            "step": s["step"], "action_type": atype, "clause_id": cid,
        })

        reward, reward_explanation = compute_step_reward(action, s)
        s["cumulative_reward"] = round(s["cumulative_reward"] + reward, 4)

        if s["step"] >= TASKS[self.task_id]["max_steps"]:
            s["done"] = True

        obs  = self._build_obs(msg, success)
        info = {
            "step":               s["step"],
            "reward":             reward,
            "reward_explanation": reward_explanation,
            "cumulative_reward":  s["cumulative_reward"],
            "done":               s["done"],
        }
        return obs, reward, s["done"], info

    def state(self) -> Dict:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def _build_obs(self, msg: str = "", success: bool = True) -> Dict:
        s    = self._state
        task = TASKS[s["task_id"]]
        score, _ = grade_task(s["task_id"], s)
        return {
            "episode_id":     s["episode_id"],
            "task_id":        s["task_id"],
            "contract_title": task["description"].split(".")[0],
            "contract_type":  task["difficulty"],
            "clauses": [
                {
                    "clause_id": c["clause_id"],
                    "title":     c["title"],
                    "text":      c["text"],
                    "status":    c.get("status", "unreviewed"),
                }
                for c in s["clauses"]
            ],
            "findings":        s["findings"],
            "compliant_ids":   s["compliant_ids"],
            "risk_rating":     s["risk_rating"],
            "recommendation":  s["recommendation"],
            "step":            s["step"],
            "max_steps":       task["max_steps"],
            "done":            s["done"],
            "last_action_ok":  success,
            "last_action_msg": msg,
            "current_score":   score,
            "available_actions": [
                "read_clause", "flag_risk", "mark_compliant",
                "set_risk_rating", "recommend", "request_revision", "submit_review"
            ],
        }
