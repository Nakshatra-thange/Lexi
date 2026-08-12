"""
Redline generation: for each DEVIATION finding, drafts a replacement
clause that fixes the compliance issue while staying consistent with the
contract's existing terminology and structure.

Deliberately scoped to DEVIATION only. MISSING findings need net-new
clause insertion (deciding where in the document it goes, renumbering,
etc.) — a different and heavier problem, out of scope here by design
rather than by oversight.

Uses MODEL_REASONING (Sonnet) — drafting quality matters here as much as
in Part 3's classification.
"""

from __future__ import annotations

import re

import anthropic

from src.config import ANTHROPIC_API_KEY, MODEL_REASONING
from src.models import Contract, Playbook, ComplianceFinding, ComplianceStatus

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


REDLINE_SYSTEM_PROMPT = """You are a contract drafter producing a redline
suggestion for a single clause that deviates from a negotiation playbook
rule.

You will be given:
- The original clause text (verbatim from the contract)
- The rule it violates and why
- Reference fallback language (a generic template — a starting point,
  NOT something to paste in unchanged)

Your job: rewrite the clause so it satisfies the rule, while:
1. Reusing the SAME defined terms the original clause uses (e.g. if the
   contract calls the parties "Vendor" and "Customer", your redline must
   too — don't switch to generic "Party A/Party B" language).
2. Preserving structure/formatting conventions from the original clause
   where reasonable (e.g. if it's in ALL CAPS, match that; if it's a
   numbered list, keep it a numbered list).
3. Changing ONLY what's needed to fix the specific deviation — don't
   rewrite unrelated parts of the clause.
4. Being drafted as real contract language a lawyer could paste in
   directly, not a description of what should change.

Respond with ONLY the redlined clause text. No preamble, no markdown
fences, no commentary, no "Here's the redline:" — just the clause text
itself, ready to paste into a document.
"""


def _extract_defined_terms(text: str) -> set[str]:
    """
    Cheap heuristic: capitalized single words that recur, as a stand-in
    for 'defined terms' (Vendor, Customer, Agreement, etc). Not a legal
    parser — just a consistency tripwire for the check below.
    """
    candidates = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    common_words = {"The", "This", "Any", "All", "Each", "Such", "Section"}
    return {w for w in candidates if w not in common_words}


def _check_term_consistency(original_clause_text: str, redline_text: str) -> list[str]:
    """
    Flags defined terms present in the original clause but missing from
    the redline — a cheap proxy for 'did the rewrite silently drift from
    the contract's own vocabulary.' Not proof of correctness, but a real
    check, and cheap enough to run on every redline for free.
    """
    original_terms = _extract_defined_terms(original_clause_text)
    redline_terms = _extract_defined_terms(redline_text)

    # Only flag terms that appear meaningfully often in the original
    # (avoid false positives on incidental capitalized words)
    frequent_terms = {
        t for t in original_terms
        if original_clause_text.count(t) >= 2
    }

    missing = frequent_terms - redline_terms
    return sorted(missing)


def generate_redline(
    finding: ComplianceFinding,
    contract: Contract,
    playbook: Playbook,
) -> tuple[str, list[str]]:
    """
    Returns (redline_text, consistency_warnings).
    consistency_warnings is non-empty when the redline may have dropped
    defined terms from the original — surface this to the user, don't
    silently swallow it.
    """
    rule = playbook.rule_by_id(finding.rule_id)
    clause = next((c for c in contract.clauses if c.id == finding.clause_id), None)

    if rule is None or clause is None:
        return "", ["Could not locate rule or clause — redline not generated."]

    user_prompt = f"""ORIGINAL CLAUSE:
{clause.text}

RULE VIOLATED: {rule.title}
Requirement: {rule.requirement}
Why this finding was flagged: {finding.explanation}

REFERENCE FALLBACK LANGUAGE (generic template, adapt don't paste):
{rule.fallback_language or "(none provided — draft from the requirement directly)"}

Draft the redline now."""

    response = client.messages.create(
        model=MODEL_REASONING,
        max_tokens=1000,
        system=REDLINE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    redline_text = response.content[0].text.strip()
    warnings = _check_term_consistency(clause.text, redline_text)

    return redline_text, warnings


def generate_all_redlines(
    findings: list[ComplianceFinding],
    contract: Contract,
    playbook: Playbook,
) -> dict[str, dict]:
    """
    Returns {rule_id: {"redline": str, "warnings": list[str]}} for every
    DEVIATION finding. Skipped findings (compliant/missing/ambiguous)
    are simply absent from the result — caller shouldn't expect entries
    for them.
    """
    results = {}
    for finding in findings:
        if finding.status != ComplianceStatus.DEVIATION:
            continue
        redline_text, warnings = generate_redline(finding, contract, playbook)
        results[finding.rule_id] = {
            "redline": redline_text,
            "warnings": warnings,
        }
    return results


if __name__ == "__main__":
    import yaml
    from rich import print as rprint
    from rich.panel import Panel
    from rich.console import Console

    from src.config import PLAYBOOKS_DIR, CONTRACTS_DIR
    from src.ingestion import segment_contract
    from src.agent import run_compliance_check

    playbook_data = yaml.safe_load(
        (PLAYBOOKS_DIR / "services_agreement_playbook.yaml").read_text()
    )
    playbook = Playbook(**playbook_data)

    raw_text = (CONTRACTS_DIR / "sample_msa.txt").read_text()
    contract = segment_contract("sample_msa", "sample_msa.txt", raw_text)
    report = run_compliance_check(contract, playbook)

    redlines = generate_all_redlines(report.findings, contract, playbook)

    console = Console()
    for rule_id, result in redlines.items():
        rule = playbook.rule_by_id(rule_id)
        console.print(Panel(
            result["redline"],
            title=f"[bold red]{rule_id}[/bold red] — {rule.title}",
            subtitle=(
                f"[yellow]⚠ term consistency warnings: {result['warnings']}[/yellow]"
                if result["warnings"] else "[green]✓ term consistency ok[/green]"
            ),
        ))