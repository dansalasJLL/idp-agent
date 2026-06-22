"""
IDP Agent — Model Provider Adapters
===================================
One swappable seam between the pipeline and whatever LLM endpoint is sanctioned.

Why this exists
---------------
The hackathon's Claude access is SANDBOX-ONLY — cleared for synthetic contracts, NOT
real client MSAs. Production must run against a JLL-sanctioned endpoint (Falcon) that
is cleared for real data. Everything else in the pipeline (parse, chunk, schema,
reduce, checklist, UI, export) is identical across both. This module is the only place
that changes when the endpoint changes.

    parse -> chunk -> [provider.extract] -> validate -> reduce -> checklist -> UI

Governance rule enforced here
-----------------------------
`cleared_for_real_data` marks whether an endpoint may receive real client text.
`assert_data_allowed()` blocks real-data runs against a sandbox endpoint.

Author: Daniel Salas Castro — JLL Hackathon 2026
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List

DEFAULT_MODEL = "claude-sonnet-4-6"   # fast + cheap + strong structured output
MAX_TOKENS = 4096
MAX_RETRIES = 3

_TOOL_NAME = "record_obligations"


def _wrap_schema(item_schema: dict) -> dict:
    """The extraction tool returns {"obligations": [ <item_schema>, ... ]}."""
    return {
        "type": "object",
        "properties": {"obligations": {"type": "array", "items": item_schema}},
        "required": ["obligations"],
    }


def parse_tool_obligations(content_blocks) -> list:
    """Pull the obligations array out of a tool-use response (find by type, not index).
    Pure function — no SDK dependency — so it's unit-testable on its own."""
    for block in content_blocks:
        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        bname = getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else None)
        if btype == "tool_use" and bname == _TOOL_NAME:
            inp = getattr(block, "input", None)
            if inp is None and isinstance(block, dict):
                inp = block.get("input", {})
            return (inp or {}).get("obligations", [])
    return []


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #
class ModelProvider(ABC):
    name: str = "base"
    cleared_for_real_data: bool = False   # may this endpoint receive real client MSAs?

    @abstractmethod
    def extract(self, system_prompt: str, user_prompt: str, item_schema: dict) -> List[dict]:
        """Return the raw list of obligation dicts for ONE clause (pre-validation).
        Implementations must honor the item_schema so the rest of the pipeline is
        endpoint-agnostic."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Claude via hackathon sponsorship — SANDBOX ONLY
# --------------------------------------------------------------------------- #
class ClaudeProvider(ModelProvider):
    """Anthropic API through the hackathon sponsorship.

    \u26a0 SANDBOX ONLY. Not cleared for real client MSA text — use synthetic contracts.
    For production with real data, swap to FalconProvider (or another sanctioned,
    data-cleared endpoint)."""
    name = "claude-sandbox"
    cleared_for_real_data = False

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = MAX_TOKENS,
                 max_retries: int = MAX_RETRIES):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def extract(self, system_prompt: str, user_prompt: str, item_schema: dict) -> List[dict]:
        tool = {
            "name": _TOOL_NAME,
            "description": "Record every obligation found in the clause.",
            "input_schema": _wrap_schema(item_schema),
        }
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": _TOOL_NAME},
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return parse_tool_obligations(resp.content)
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Claude extraction failed after {self.max_retries} tries: {last_err}")


# --------------------------------------------------------------------------- #
# JLL Falcon — sanctioned, cleared for real client data  (production target)
# --------------------------------------------------------------------------- #
class FalconProvider(ModelProvider):
    """JLL Falcon inference endpoint — the sanctioned production target, cleared for
    real client MSAs (data stays inside JLL's governed envelope).

    STUB: wire to Falcon's inference API when the team has its interface. The only
    contract that matters: return the same `obligations` list shape, honoring
    item_schema. If Falcon exposes an OpenAI-compatible or tool/function-calling
    surface, mirror ClaudeProvider.extract; if it returns plain JSON text, parse the
    JSON and return its `obligations` array. Nothing downstream changes."""
    name = "jll-falcon"
    cleared_for_real_data = True

    def __init__(self, endpoint: str = "", **kwargs):
        self.endpoint = endpoint
        self.kwargs = kwargs

    def extract(self, system_prompt: str, user_prompt: str, item_schema: dict) -> List[dict]:
        raise NotImplementedError(
            "FalconProvider is a stub. Wire it to the JLL Falcon inference endpoint, "
            "returning the obligations list per item_schema. Keep the schema contract "
            "identical so the rest of the pipeline is unchanged."
        )


# --------------------------------------------------------------------------- #
# JLL GPT — assistant layer (optional; better for interactive Q&A than batch)
# --------------------------------------------------------------------------- #
class JLLGPTProvider(ModelProvider):
    """JLL GPT — the CRE assistant layer on Falcon. Sanctioned and data-cleared, but
    geared to interactive assistance rather than hundreds of batch extraction calls.
    Provided for completeness / a conversational front end. STUB."""
    name = "jll-gpt"
    cleared_for_real_data = True

    def extract(self, system_prompt: str, user_prompt: str, item_schema: dict) -> List[dict]:
        raise NotImplementedError(
            "JLLGPTProvider is a stub. If JLL GPT exposes a programmatic/Skills API, "
            "send the prompt + schema and return the obligations array. For batch "
            "per-clause extraction, FalconProvider is the better fit."
        )


# --------------------------------------------------------------------------- #
# Factory + governance guard
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "claude": ClaudeProvider,
    "falcon": FalconProvider,
    "jllgpt": JLLGPTProvider,
}


def get_provider(name: str = "claude", **kwargs) -> ModelProvider:
    key = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    key = {"claudesandbox": "claude", "jllfalcon": "falcon", "jllgptprovider": "jllgpt"}.get(key, key)
    if key not in _REGISTRY:
        raise ValueError(f"Unknown provider '{name}'. Options: {list(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def assert_data_allowed(provider: ModelProvider, contains_real_client_data: bool) -> None:
    """Hard governance gate: refuse to send real client MSA text to a sandbox endpoint."""
    if contains_real_client_data and not provider.cleared_for_real_data:
        raise PermissionError(
            f"Endpoint '{provider.name}' is NOT cleared for real client data. "
            f"Use synthetic contracts here, or switch to a sanctioned endpoint "
            f"(e.g. Falcon) for real MSAs."
        )


# --------------------------------------------------------------------------- #
# Self-test: response parsing + governance guard (no SDK / network needed)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # fake tool-use response shaped like the Anthropic SDK's resp.content
    class Block:
        def __init__(self, **kw): self.__dict__.update(kw)

    blocks = [
        Block(type="text", text="Here are the obligations."),
        Block(type="tool_use", name=_TOOL_NAME, input={"obligations": [
            {"description": "Maintain insurance", "category": "Insurance"},
            {"description": "Pay within 30 days", "category": "Financial"},
        ]}),
    ]
    got = parse_tool_obligations(blocks)
    assert len(got) == 2 and got[0]["category"] == "Insurance", got
    # also works on plain dict blocks
    assert parse_tool_obligations([{"type": "tool_use", "name": _TOOL_NAME,
                                    "input": {"obligations": [{"x": 1}]}}]) == [{"x": 1}]
    # empty when no tool block
    assert parse_tool_obligations([Block(type="text", text="hi")]) == []
    print("parse_tool_obligations: OK")

    # governance guard
    sandbox = ClaudeProvider.__new__(ClaudeProvider)  # don't init SDK
    sandbox.name, sandbox.cleared_for_real_data = "claude-sandbox", False
    assert_data_allowed(sandbox, contains_real_client_data=False)  # fine
    try:
        assert_data_allowed(sandbox, contains_real_client_data=True)
        raise AssertionError("guard should have blocked real data on sandbox")
    except PermissionError:
        print("assert_data_allowed: blocked real data on sandbox endpoint as expected")

    falcon = FalconProvider()
    assert falcon.cleared_for_real_data is True
    assert_data_allowed(falcon, contains_real_client_data=True)  # allowed
    print("Falcon cleared for real data: OK")
    print("\nAll provider self-tests passed.")
