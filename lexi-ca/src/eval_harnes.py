"""
Eval harness: scores agent output against hand-labeled ground truth, and
independently verifies grounding (no hallucinated quotes) across any
report.

Two separate concerns kept deliberately separate:
1. Accuracy eval — did the agent reach the RIGHT compliance conclusion?
   Requires ground truth, one dataset at a time.
2. Grounding eval — is every claim the agent made actually TRUE of the
   source document, independent of whether the conclusion was "right"?
   Requires no ground truth at all — it's a property checkable on any
   report, on any contract, forever. This is what you'd run in CI on
   every new contract processed in production, since you won't have
   hand labels for those.

Keeping these separate matters: an agent can be perfectly grounded
(every quote is real) and still wrong (misjudges what the quote implies),
or it can reach the right conclusion while fabricating the quote that
"supports" it. A single blended score would hide which failure mode
you're looking at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from src.models import Contract, ComplianceReport, ComplianceStatus


# ---------------------------------------------------------------------------
# 1. Accuracy eval (needs ground truth)
# ---------------------------------------------------------------------------

@dataclass
class RuleScore:
    rule_id: str
    predicted: str | None
    actual: str
    correct: bool


@dataclass
class AccuracyEvalResult:
    rule_scores: list[RuleScore] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.rule_scores:
            return 0.0
        correct = sum(1 for r in self.rule_scores if r.correct)
        return correct / len(self.rule_scores)

    @property
    def deviation_recall(self) -> float:
        """
        Of the rules that SHOULD be flagged as deviation (the ones that
        actually matter to a lawyer), what fraction did the agent catch?
        This is the more important number than raw accuracy — missing a
        real deviation (false negative) is a materially worse failure
        than being over-cautious, since it's the kind of miss that costs
        a client money.
        """
        true_deviations = [r for r in self.rule_scores if r.actual == "deviation"]
        if not true_deviations:
            return 1.0
        caught = sum(1 for r in true_deviations if r.predicted == "deviation")
        return caught / len(true_deviations)

    @property
    def false_negatives(self) -> list[str]:
        """Rules that were真 deviations but the agent missed — highest priority to review."""
        return [
            r.rule_id for r in self.rule_scores
            if r.actual == "deviation" and r.predicted != "deviation"
        ]

    @property
    def false_positives(self) -> list[str]:
        """Rules the agent flagged as deviation but ground truth says otherwise."""
        return [
            r.rule_id for r in self.rule_scores
            if r.predicted == "deviation" and r.actual != "deviation"
        ]


def load_ground_truth(path) -> dict[str, str]:
    data = yaml.safe_load(open(path))
    return {label["rule_id"]: label["status"] for label in data["labels"]}


def score_against_ground_truth(
    report: ComplianceReport, ground_truth: dict[str, str]
) -> AccuracyEvalResult:
    result = AccuracyEvalResult()

    findings_by_rule = {f.rule_id: f for f in report.findings}

    for rule_id, actual_status in ground_truth.items():
        finding = findings_by_rule.get(rule_id)
        predicted_status = finding.status.value if finding else None
        result.rule_scores.append(
            RuleScore(
                rule_id=rule_id,
                predicted=predicted_status,
                actual=actual_status,
                correct=(predicted_status == actual_status),
            )
        )

    return result


# ---------------------------------------------------------------------------
# 2. Grounding eval (no ground truth needed — runs on any report)
# ---------------------------------------------------------------------------

@dataclass
class GroundingIssue:
    rule_id: str
    issue_type: str  # "missing_quote" | "quote_not_verbatim" | "clause_id_not_found"
    detail: str


@dataclass
class GroundingEvalResult:
    issues: list[GroundingIssue] = field(default_factory=list)
    total_checked: int = 0

    @property
    def grounding_rate(self) -> float:
        if self.total_checked == 0:
            return 1.0
        return 1 - (len(self.issues) / self.total_checked)


def check_grounding(report: ComplianceReport, contract: Contract) -> GroundingEvalResult:
    """
    Independently re-verifies every compliant/deviation finding's
    source_quote against the actual contract text. This duplicates the
    check already done inline in agent.py by design — that inline check
    can be bypassed or skipped if agent.py changes; this is the
    independent audit that should never be allowed to silently rot.
    """
    result = GroundingEvalResult()
    clauses_by_id = {c.id: c for c in contract.clauses}

    checkable_statuses = {ComplianceStatus.COMPLIANT, ComplianceStatus.DEVIATION}

    for finding in report.findings:
        if finding.status not in checkable_statuses:
            continue

        result.total_checked += 1

        if finding.clause_id not in clauses_by_id:
            result.issues.append(GroundingIssue(
                rule_id=finding.rule_id,
                issue_type="clause_id_not_found",
                detail=f"clause_id '{finding.clause_id}' does not exist in contract",
            ))
            continue

        if not finding.source_quote or not finding.source_quote.strip():
            result.issues.append(GroundingIssue(
                rule_id=finding.rule_id,
                issue_type="missing_quote",
                detail="status requires a source_quote but none was provided",
            ))
            continue

        clause_text = clauses_by_id[finding.clause_id].text
        if finding.source_quote.strip() not in clause_text:
            result.issues.append(GroundingIssue(
                rule_id=finding.rule_id,
                issue_type="quote_not_verbatim",
                detail=(
                    f"quote '{finding.source_quote[:60]}...' not found "
                    f"verbatim in clause {finding.clause_id}"
                ),
            ))

    return result


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint

    from src.config import PLAYBOOKS_DIR, CONTRACTS_DIR, EVALS_DIR
    from src.ingestion import segment_contract
    from src.agent import run_compliance_check
    from src.models import Playbook

    console = Console()

    playbook_data = yaml.safe_load(
        (PLAYBOOKS_DIR / "services_agreement_playbook.yaml").read_text()
    )
    playbook = Playbook(**playbook_data)

    raw_text = (CONTRACTS_DIR / "sample_msa.txt").read_text()
    contract = segment_contract("sample_msa", "sample_msa.txt", raw_text)
    report = run_compliance_check(contract, playbook)

    # --- Accuracy eval ---
    ground_truth = load_ground_truth(
        EVALS_DIR / "ground_truth" / "sample_msa_ground_truth.yaml"
    )
    accuracy_result = score_against_ground_truth(report, ground_truth)

    table = Table(title="Accuracy vs Ground Truth")
    table.add_column("Rule")
    table.add_column("Predicted")
    table.add_column("Actual")
    table.add_column("Correct")
    for r in accuracy_result.rule_scores:
        table.add_row(
            r.rule_id,
            r.predicted or "—",
            r.actual,
            "[green]✓[/green]" if r.correct else "[red]✗[/red]",
        )
    console.print(table)

    rprint(f"\n[bold]Overall accuracy:[/bold] {accuracy_result.accuracy:.0%}")
    rprint(f"[bold]Deviation recall:[/bold] {accuracy_result.deviation_recall:.0%} "
           f"(the number that actually matters)")
    if accuracy_result.false_negatives:
        rprint(f"[red bold]Missed deviations (false negatives):[/red bold] "
               f"{accuracy_result.false_negatives}")
    if accuracy_result.false_positives:
        rprint(f"[yellow bold]Over-flagged (false positives):[/yellow bold] "
               f"{accuracy_result.false_positives}")

    # --- Grounding eval ---
    grounding_result = check_grounding(report, contract)
    rprint(f"\n[bold]Grounding rate:[/bold] {grounding_result.grounding_rate:.0%} "
           f"({grounding_result.total_checked} findings checked)")
    for issue in grounding_result.issues:
        rprint(f"  [red]✗ {issue.rule_id}[/red] [{issue.issue_type}] {issue.detail}")