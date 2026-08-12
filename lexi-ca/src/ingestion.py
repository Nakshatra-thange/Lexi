"""
Clause segmentation: turns raw contract text into structured ContractClause
objects.

Design choices, and why:
- Uses Haiku, not Sonnet. Segmentation is extraction/formatting, not
  legal reasoning — paying reasoning-model prices for it is wasted spend.
  The compliance judgment in Part 3 is where model quality actually matters.
- One API call per contract, not one per clause. A single well-structured
  prompt with the whole contract (most contracts fit in one context window)
  is far cheaper than looping per-section.
- Disk-cached by a hash of the contract text. Re-running the pipeline
  during development costs zero extra tokens after the first run.
- The LLM returns clause boundaries as character offsets into the
  ORIGINAL text, not a re-typed copy of the clause. This is important:
  asking a model to reproduce full clause text verbatim is more tokens,
  more cost, and a chance for it to subtly paraphrase. Instead we ask
  it to point at spans, and we slice the original string ourselves —
  guaranteed byte-for-byte fidelity, which matters a lot when Part 5's
  hallucination checker has to verify grounding against source text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anthropic
from pydantic import BaseModel

from src.config import ANTHROPIC_API_KEY, MODEL_SEGMENTATION, CACHE_DIR
from src.models import Contract, ContractClause, ClauseCategory

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


SEGMENTATION_SYSTEM_PROMPT = """You segment legal contracts into clauses.

You will be given a contract with line numbers prefixed to help you locate
text. For each distinct clause/section, identify:
- clause_number: the section number/heading if present (e.g. "3", "3.1"), else null
- category: best-fit from this exact list: liability, indemnification,
  termination, ip_assignment, confidentiality, governing_law, payment,
  warranty, data_protection, auto_renewal, other
- start_char: character offset where the clause begins in the ORIGINAL
  text (not counting the line-number prefixes you were shown)
- end_char: character offset where the clause ends

Respond with ONLY a JSON array, no markdown fences, no preamble. Format:
[{"clause_number": "3", "category": "liability", "start_char": 412, "end_char": 810}]

Rules:
- Cover the whole document; don't skip boilerplate, categorize it "other"
- Offsets must be precise — they will be used to slice the original string
  directly, so an off-by-a-few-characters error will corrupt the clause text
- One entry per clause, do not split a single clause into fragments
"""


def _contract_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cache_path(contract_id: str, text_hash: str) -> Path:
    return CACHE_DIR / f"{contract_id}_{text_hash}.json"


def _build_offset_annotated_text(text: str) -> str:
    """
    Gives the model periodic character-offset markers inline, e.g.
    every 200 chars, so it can ground start/end offsets accurately
    instead of guessing. This meaningfully improves offset precision
    for longer documents without costing extra API calls.
    """
    marker_interval = 200
    annotated = []
    for i, ch in enumerate(text):
        if i % marker_interval == 0:
            annotated.append(f"[offset:{i}]")
        annotated.append(ch)
    return "".join(annotated)


def _call_llm_for_segments(raw_text: str) -> list[dict]:
    annotated_text = _build_offset_annotated_text(raw_text)

    response = client.messages.create(
        model=MODEL_SEGMENTATION,
        max_tokens=4000,
        system=SEGMENTATION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Contract text with offset markers:\n\n{annotated_text}",
            }
        ],
    )

    raw_output = response.content[0].text.strip()

    # Defensive cleanup in case the model wraps in fences despite instructions
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]

    return json.loads(raw_output)


def _validate_and_clip_offsets(segments: list[dict], text_len: int) -> list[dict]:
    """LLM offsets are estimates, not guarantees. Clip to valid range and
    drop anything degenerate rather than letting it corrupt downstream data."""
    valid = []
    for seg in segments:
        start = max(0, min(seg.get("start_char", 0), text_len))
        end = max(0, min(seg.get("end_char", 0), text_len))
        if end <= start:
            continue
        seg["start_char"] = start
        seg["end_char"] = end
        valid.append(seg)
    return valid


def segment_contract(contract_id: str, filename: str, raw_text: str) -> Contract:
    text_hash = _contract_hash(raw_text)
    cache_file = _cache_path(contract_id, text_hash)

    if cache_file.exists():
        segments = json.loads(cache_file.read_text())
    else:
        segments = _call_llm_for_segments(raw_text)
        segments = _validate_and_clip_offsets(segments, len(raw_text))
        cache_file.write_text(json.dumps(segments, indent=2))

    clauses = []
    for i, seg in enumerate(segments):
        category = seg.get("category", "other")
        try:
            category_enum = ClauseCategory(category)
        except ValueError:
            category_enum = ClauseCategory.OTHER

        clause_text = raw_text[seg["start_char"]:seg["end_char"]]

        clauses.append(
            ContractClause(
                id=f"{contract_id}_clause_{i}",
                contract_id=contract_id,
                clause_number=seg.get("clause_number"),
                category=category_enum,
                text=clause_text.strip(),
                start_char=seg["start_char"],
                end_char=seg["end_char"],
            )
        )

    return Contract(
        id=contract_id,
        filename=filename,
        raw_text=raw_text,
        clauses=clauses,
    )


if __name__ == "__main__":
    from src.config import CONTRACTS_DIR
    from rich import print as rprint

    sample_path = CONTRACTS_DIR / "sample_msa.txt"
    raw = sample_path.read_text()

    contract = segment_contract(
        contract_id="sample_msa",
        filename=sample_path.name,
        raw_text=raw,
    )

    rprint(f"[bold green]Segmented into {len(contract.clauses)} clauses[/bold green]\n")
    for clause in contract.clauses:
        rprint(
            f"[cyan]{clause.clause_number or '?'}[/cyan] "
            f"[yellow]({clause.category.value})[/yellow] "
            f"— {len(clause.text)} chars"
        )
        rprint(f"  {clause.text[:80]}...\n")