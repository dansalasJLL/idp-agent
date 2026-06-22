"""
IDP Agent — Obligation Extraction Core
======================================
Pydantic schema + extraction prompt + per-clause extraction (via a ModelProvider).

The model call is delegated to a provider (see providers.py), so this module is
endpoint-agnostic: sandbox Claude for the hackathon, a JLL-sanctioned endpoint for
production — same schema, same validation, same priority logic.

Pipeline position
-----------------
    parse -> chunk -> [THIS MODULE per chunk] -> reduce/dedup -> checklist -> export

Quick start
-----------
    pip install anthropic pydantic
    export ANTHROPIC_API_KEY=sk-ant-...        # sandbox / synthetic contracts only
    python idp_extraction.py

Author: Daniel Salas Castro — JLL Hackathon 2026
"""

from __future__ import annotations

import json
import os
from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

REVIEW_CONFIDENCE_THRESHOLD = 0.70  # items below this get flagged for human review


# ---------------------------------------------------------------------------
# Enums  (the shared vocabulary — keep in sync with the design doc taxonomy)
# ---------------------------------------------------------------------------
class Category(str, Enum):
    FINANCIAL = "Financial"
    INSURANCE = "Insurance"
    REPORTING = "Reporting"
    SLA = "Service Level (SLA)"
    COMPLIANCE = "Compliance & Regulatory"
    NOTICE = "Notice"
    TERM_RENEWAL = "Term & Renewal"
    TERMINATION = "Termination"
    INDEMNITY = "Indemnity & Liability"
    CONFIDENTIALITY = "Confidentiality & Data"


class Party(str, Enum):
    JLL = "JLL"
    CLIENT = "Client"
    VENDOR = "Vendor"
    BOTH = "Both"


class TriggerType(str, Enum):
    SPECIFIC_DATE = "Specific date"
    RECURRING = "Recurring"
    EVENT_DRIVEN = "Event-driven"
    CONDITIONAL = "Conditional"


class Priority(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


# ---------------------------------------------------------------------------
# Input model — one clause chunk produced by the parser/chunker
# ---------------------------------------------------------------------------
class ClauseChunk(BaseModel):
    """A structure-aware unit of the MSA. The metadata becomes the citation."""
    section_id: str = Field(..., description="e.g. '8.3', 'Schedule C', 'Exhibit 2'")
    heading: Optional[str] = Field(None, description="Clause/section heading if available")
    page_range: str = Field(..., description="e.g. '142' or '142-144'")
    text: str = Field(..., description="Raw text of this clause")


# ---------------------------------------------------------------------------
# Output model — what the model returns PER obligation
# (obligation_id and priority are derived later, NOT extracted by the model)
# ---------------------------------------------------------------------------
class ExtractedObligation(BaseModel):
    description: str = Field(..., description="Plain-language statement of what must be done.")
    category: Category
    responsible_party: Party
    trigger_type: TriggerType
    deadline: Optional[date] = Field(None, description="Absolute due date (YYYY-MM-DD) if stated.")
    frequency: Optional[str] = Field(None, description="Cadence for recurring obligations.")
    penalty: Optional[str] = Field(None, description="Consequence of non-compliance, incl. amount.")
    verbatim_snippet: str = Field(
        ...,
        description="EXACT text copied from the clause that supports this obligation. "
                    "Required — an obligation with no supporting text must not be returned.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model confidence this is a genuine, correctly-parsed obligation.",
    )

    @field_validator("verbatim_snippet")
    @classmethod
    def snippet_must_be_substantive(cls, v: str) -> str:
        if not v or len(v.strip()) < 8:
            raise ValueError("verbatim_snippet too short to be a real citation")
        return v.strip()


# ---------------------------------------------------------------------------
# Finalized record — enriched after extraction (id + source + priority)
# ---------------------------------------------------------------------------
class Obligation(ExtractedObligation):
    obligation_id: str
    source_section: str
    source_page: str
    priority: Priority
    needs_review: bool

    @classmethod
    def from_extracted(cls, ext: ExtractedObligation, *, obligation_id: str, chunk: ClauseChunk) -> "Obligation":
        return cls(
            obligation_id=obligation_id,
            source_section=chunk.section_id,
            source_page=chunk.page_range,
            priority=derive_priority(ext),
            needs_review=ext.confidence < REVIEW_CONFIDENCE_THRESHOLD,
            **ext.model_dump(),
        )


# ---------------------------------------------------------------------------
# Priority derivation  (in code, not by the model — keeps it consistent)
# ---------------------------------------------------------------------------
_MONEY_HINTS = ("$", "usd", "fee", "penalt", "credit", "interest", "%", "per day", "per diem")


def derive_priority(ext: ExtractedObligation) -> Priority:
    has_penalty = bool(ext.penalty and ext.penalty.strip())
    monetary = has_penalty and any(h in ext.penalty.lower() for h in _MONEY_HINTS)
    firm_trigger = ext.trigger_type in (
        TriggerType.SPECIFIC_DATE, TriggerType.RECURRING, TriggerType.EVENT_DRIVEN
    )
    if monetary and firm_trigger:
        return Priority.HIGH
    if has_penalty or (firm_trigger and ext.deadline is not None):
        return Priority.MEDIUM
    return Priority.LOW


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a contract analyst for JLL's Commercial Real Estate team. \
You extract obligations from one clause of a Master Service Agreement (MSA).

An OBLIGATION is any binding requirement a party must perform or satisfy: payments, \
insurance coverage, reports, service levels, notices, renewals, terminations, \
indemnities, confidentiality duties, and similar commitments.

Rules:
1. Extract EVERY obligation in the clause. A single clause may contain several, or none.
2. For each obligation, copy the EXACT supporting text into verbatim_snippet. \
Never paraphrase the snippet. If you cannot point to supporting text, do not return the item.
3. Do NOT invent obligations. If the clause is purely definitional, recital, or \
boilerplate with no binding requirement, return an empty list.
4. Capture penalties and amounts whenever the clause states a consequence of non-compliance.
5. Assign responsible_party from the clause's perspective (JLL, Client, Vendor, or Both).
6. Set confidence honestly — lower it when the clause is ambiguous or you are inferring.
7. Return results ONLY by calling the record_obligations tool."""


def build_user_prompt(chunk: ClauseChunk) -> str:
    heading = f" — {chunk.heading}" if chunk.heading else ""
    return (
        f"Clause {chunk.section_id}{heading} (page {chunk.page_range}):\n\n"
        f'"""\n{chunk.text}\n"""\n\n'
        f"Extract all obligations from this clause."
    )


# ---------------------------------------------------------------------------
# Extraction (delegates the model call to a ModelProvider)
# ---------------------------------------------------------------------------
def extract_obligations_from_chunk(chunk: ClauseChunk, provider) -> List["Obligation"]:
    """Map step: one clause -> validated obligations, via the given provider.
    Malformed items are skipped, not allowed to poison the run."""
    raw_items = provider.extract(SYSTEM_PROMPT, build_user_prompt(chunk),
                                 ExtractedObligation.model_json_schema())
    return _validate(raw_items, chunk)


def _validate(raw_items: list, chunk: ClauseChunk) -> List["Obligation"]:
    out: List[Obligation] = []
    for i, item in enumerate(raw_items):
        try:
            ext = ExtractedObligation.model_validate(item)
        except ValidationError as e:
            print(f"  [skip] clause {chunk.section_id} item {i}: {e.error_count()} error(s)")
            continue
        out.append(Obligation.from_extracted(ext, obligation_id=f"{chunk.section_id}-{i + 1}", chunk=chunk))
    return out


def extract_all(chunks: List[ClauseChunk], provider) -> List["Obligation"]:
    """Run extraction across many chunks (sequential). For production throughput,
    parallelize with asyncio + an async provider implementation."""
    results: List[Obligation] = []
    for n, chunk in enumerate(chunks, 1):
        found = extract_obligations_from_chunk(chunk, provider)
        results.extend(found)
        print(f"[{n}/{len(chunks)}] {chunk.section_id}: {len(found)} obligation(s)")
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
SAMPLE_CHUNK = ClauseChunk(
    section_id="8.3",
    heading="Insurance Requirements",
    page_range="142",
    text=(
        "Service Provider shall maintain Commercial General Liability insurance with "
        "limits of not less than $5,000,000 per occurrence throughout the Term, and "
        "shall furnish Client with a certificate of insurance evidencing such coverage "
        "within ten (10) business days of the Effective Date and upon each renewal. "
        "Failure to maintain the required coverage shall entitle Client to assess a "
        "penalty of $1,000 for each day the coverage lapses."
    ),
)


def _demo_offline():
    print("No ANTHROPIC_API_KEY found — running OFFLINE schema/validation check.\n")
    sample = {
        "description": "Maintain Commercial General Liability insurance of at least "
                       "$5,000,000 per occurrence for the full Term.",
        "category": "Insurance", "responsible_party": "Vendor", "trigger_type": "Recurring",
        "deadline": None, "frequency": "continuous / each renewal",
        "penalty": "$1,000 per day the coverage lapses",
        "verbatim_snippet": "shall maintain Commercial General Liability insurance with "
                            "limits of not less than $5,000,000 per occurrence",
        "confidence": 0.95,
    }
    ext = ExtractedObligation.model_validate(sample)
    ob = Obligation.from_extracted(ext, obligation_id="8.3-1", chunk=SAMPLE_CHUNK)
    print(json.dumps(ob.model_dump(mode="json"), indent=2))
    print(f"\nDerived priority: {ob.priority.value}   needs_review: {ob.needs_review}")


def _demo_live():
    from providers import ClaudeProvider, assert_data_allowed
    provider = ClaudeProvider()
    # SANDBOX guardrail: this sample is synthetic, so real-data flag is False.
    assert_data_allowed(provider, contains_real_client_data=False)
    print(f"Calling provider '{provider.name}' on the sample clause...\n")
    for ob in extract_obligations_from_chunk(SAMPLE_CHUNK, provider):
        print(json.dumps(ob.model_dump(mode="json"), indent=2)); print("-" * 60)


if __name__ == "__main__":
    if os.getenv("ANTHROPIC_API_KEY"):
        _demo_live()
    else:
        _demo_offline()
