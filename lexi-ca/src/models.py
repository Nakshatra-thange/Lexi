"""
Core data models for the compliance agent.

These are the contracts (pun intended) between every stage of the
pipeline: ingestion -> agent -> redlines -> eval. Keeping them strict
and typed now saves a lot of debugging later, since LLM output gets
parsed directly into these.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ClauseCategory(str, Enum):
    LIABILITY = "liability"
    INDEMNIFICATION = "indemnification"
    TERMINATION = "termination"
    IP_ASSIGNMENT = "ip_assignment"
    CONFIDENTIALITY = "confidentiality"
    GOVERNING_LAW = "governing_law"
    PAYMENT = "payment"
    WARRANTY = "warranty"
    DATA_PROTECTION = "data_protection"
    AUTO_RENEWAL = "auto_renewal"
    OTHER = "other"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"          # clause satisfies the playbook rule
    DEVIATION = "deviation"          # clause exists but violates the rule
    MISSING = "missing"              # rule requires a clause that isn't present
    AMBIGUOUS = "ambiguous"          # can't confidently classify — flag for human


# ---------------------------------------------------------------------------
# Playbook
# ---------------------------------------------------------------------------

class PlaybookRule(BaseModel):
    id: str
    category: ClauseCategory
    title: str
    requirement: str = Field(
        ..., description="Plain-English statement of what must be true."
    )
    risk_if_violated: RiskLevel
    rationale: str = Field(
        ..., description="Why this rule exists — used in explanations to the user."
    )
    fallback_language: Optional[str] = Field(
        None, description="Preferred replacement clause text, if the rule is violated."
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Hints for the ingestion/matching stage, not a substitute for LLM reasoning.",
    )


class Playbook(BaseModel):
    name: str
    description: str
    rules: list[PlaybookRule]

    def rule_by_id(self, rule_id: str) -> Optional[PlaybookRule]:
        return next((r for r in self.rules if r.id == rule_id), None)


# ---------------------------------------------------------------------------
# Contract / clauses
# ---------------------------------------------------------------------------

class ContractClause(BaseModel):
    id: str
    contract_id: str
    clause_number: Optional[str] = None
    category: Optional[ClauseCategory] = None
    text: str
    start_char: int
    end_char: int


class Contract(BaseModel):
    id: str
    filename: str
    raw_text: str
    clauses: list[ContractClause] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Findings / report
# ---------------------------------------------------------------------------

class ComplianceFinding(BaseModel):
    rule_id: str
    clause_id: Optional[str] = Field(
        None, description="None if status == MISSING (no matching clause exists)."
    )
    status: ComplianceStatus
    severity: RiskLevel
    explanation: str = Field(
        ..., description="Why the agent reached this conclusion."
    )
    source_quote: Optional[str] = Field(
        None, description="Verbatim text from the contract the finding is grounded in. "
                           "Required for DEVIATION/COMPLIANT — absence signals a possible hallucination."
    )
    suggested_redline: Optional[str] = None


class ComplianceReport(BaseModel):
    contract_id: str
    playbook_name: str
    findings: list[ComplianceFinding]
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    def summary(self) -> dict:
        counts = {status.value: 0 for status in ComplianceStatus}
        for f in self.findings:
            counts[f.status.value] += 1
        return counts