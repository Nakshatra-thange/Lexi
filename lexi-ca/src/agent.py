"""
Compliance agent core: maps a contract's clauses against a playbook and
produces structured ComplianceFinding objects.

Key design decision: the agent is asked to justify every non-MISSING
finding with a verbatim quote from the actual clause text. We then verify
that quote is a real substring of the source in code (see
_verify_grounding). This isn't optional decoration — it's the mechanism
that makes Part 5's eval meaningful. An agent that reasons correctly but
can't be checked against source text isn't trustworthy in a legal context,
and cheap wrapper demos never bother to enforce this.

Uses MODEL_REASONING (Sonnet) — this is the one place in the pipeline
where model quality genuinely matters, unlike segmentation.
"""

from __future__ import annotations

import json

import anthropic

from src.config import ANTHROPIC_API_KEY, MODEL_REASONING
from src.models import (
    Contract,
    Playbook,
    ComplianceFinding,
    ComplianceReport,
    ComplianceStatus,
    RiskLevel,
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


AGENT_SYSTEM_PROMPT = """You are a contract compliance reviewer checking a
contract against a negotiation playbook. You are careful, conservative,
and never invent facts.

For EACH playbook rule, determine one status:
- "compliant": a clause exists and satisfies the rule
- "deviation": a clause exists but violates or falls short of the rule
- "missing": the rule requires something no clause addresses at all
- "ambiguous": you cannot confidently determine status from the text given

Rules for your output:
1. For "compliant" or "deviation", you MUST include source_quote: a
   VERBATIM substring copied exactly from the provided clause text
   (character-for-character, no paraphrasing, no ellipsis-editing).
   If you cannot produce an exact quote, use status "ambiguous" instead.
2. For "missing", source_quote and clause_id must be null.
3. explanation must reference the SPECIFIC language in the clause, not
   generic restatement of the rule.
4. Do not flag a rule as violated based on assumptions about intent —
   only on what the text actually says.
5. severity should normally match the rule's risk_if_violated for
   "deviation"/"missing" findings, and should be "low" for "compliant".

Respond with ONLY a JSON array, no markdown fences, no preamble:
[
  {
    "rule_id": "LIAB-01",
    "clause_id": "sample_msa_clause_2",
    "status": "deviation",
    "severity": "high",
    "explanation": "...",
    "source_quote": "..."
  }
]
"""


def _build_user_prompt(contract: Contract, playbook: Playbook) -> str:
    clauses_block = "\n\n".join(
        f"[clause_id: {c.id}] [clause_number: {c.clause_number}] "
        f"[category: {c.category.value}]\n{c.text}"
        for c in contract.clauses
    )

    rules_block = "\n\n".join(
        f"[rule_id: {r.id}] {r.title}\n"
        f"Requirement: {r.requirement}\n"
        f"Category: {r.category.value}"
        for r in playbook.rules
    )

    return f"""PLAYBOOK RULES:
{rules_block}

CONTRACT CLAUSES:
{clauses_block}

Evaluate every playbook rule above against these clauses and return the
JSON array as instructed."""


def _verify_grounding(finding_dict: dict, contract: Contract) -> tuple[bool, str | None]:
    """
    Checks that source_quote is an exact substring of the referenced
    clause's text. Returns (is_grounded, matched_clause_text).

    This is the single most important function in the whole pipeline for
    trust purposes — it's the difference between "the agent said it found
    something" and "we verified the agent found something."
    """
    clause_id = finding_dict.get("clause_id")
    quote = finding_dict.get("source_quote")

    if clause_id is None or quote is None:
        return True, None  # nothing to verify (e.g. MISSING status)

    clause = next((c for c in contract.clauses if c.id == clause_id), None)
    if clause is None:
        return False, None  # agent referenced a clause_id that doesn't exist

    is_grounded = quote.strip() in clause.text
    return is_grounded, clause.text


def _parse_llm_output(raw_output: str) -> list[dict]:
    raw_output = raw_output.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
    return json.loads(raw_output)


def run_compliance_check(contract: Contract, playbook: Playbook) -> ComplianceReport:
    user_prompt = _build_user_prompt(contract, playbook)

    response = client.messages.create(
        model=MODEL_REASONING,
        max_tokens=4000,
        system=AGENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw_findings = _parse_llm_output(response.content[0].text)

    findings: list[ComplianceFinding] = []
    seen_rule_ids = set()

    for fd in raw_findings:
        is_grounded, _ = _verify_grounding(fd, contract)

        status = fd.get("status")
        # Downgrade ungrounded quotes rather than trust them silently.
        # A finding we can't verify against source text is not one we
        # should present to a lawyer as fact.
        if not is_grounded and status in ("compliant", "deviation"):
            status = "ambiguous"
            fd["explanation"] = (
                f"[GROUNDING FAILED — quote not found verbatim in source clause] "
                f"{fd.get('explanation', '')}"
            )

        rule = playbook.rule_by_id(fd["rule_id"])
        fallback = rule.fallback_language if rule else None

        findings.append(
            ComplianceFinding(
                rule_id=fd["rule_id"],
                clause_id=fd.get("clause_id"),
                status=ComplianceStatus(status),
                severity=RiskLevel(fd.get("severity", "low")),
                explanation=fd.get("explanation", ""),
                source_quote=fd.get("source_quote"),
                suggested_redline=fallback if status == "deviation" else None,
            )
        )
        seen_rule_ids.add(fd["rule_id"])

    # Safety net: if the model silently skipped a rule entirely, don't let
    # it vanish from the report — surface it as ambiguous so a human checks it.
    for rule in playbook.rules:
        if rule.id not in seen_rule_ids:
            findings.append(
                ComplianceFinding(
                    rule_id=rule.id,
                    clause_id=None,
                    status=ComplianceStatus.AMBIGUOUS,
                    severity=rule.risk_if_violated,
                    explanation="Agent did not return a finding for this rule.",
                    source_quote=None,
                    suggested_redline=None,
                )
            )

    return ComplianceReport(
        contract_id=contract.id,
        playbook_name=playbook.name,
        findings=findings,
    )


if __name__ == "__main__":
    import yaml
    from rich import print as rprint
    from rich.table import Table
    from rich.console import Console

    from src.config import PLAYBOOKS_DIR, CONTRACTS_DIR
    from src.ingestion import segment_contract

    playbook_data = yaml.safe_load(
        (PLAYBOOKS_DIR / "services_agreement_playbook.yaml").read_text()
    )
    playbook = Playbook(**playbook_data)

    raw_text = (CONTRACTS_DIR / "sample_msa.txt").read_text()
    contract = segment_contract("sample_msa", "sample_msa.txt", raw_text)

    report = run_compliance_check(contract, playbook)

    console = Console()
    table = Table(title=f"Compliance Report — {contract.filename}")
    table.add_column("Rule")
    table.add_column("Status")
    table.add_column("Severity")
    table.add_column("Explanation", max_width=60)

    status_colors = {
        "compliant": "green",
        "deviation": "red",
        "missing": "yellow",
        "ambiguous": "magenta",
    }

    for f in report.findings:
        color = status_colors.get(f.status.value, "white")
        table.add_row(
            f.rule_id,
            f"[{color}]{f.status.value}[/{color}]",
            f.severity.value,
            f.explanation,
        )

    console.print(table)
    rprint(f"\n[bold]Summary:[/bold] {report.summary()}")