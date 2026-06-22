# Intelligent Document Processing Agent

Turns a Master Service Agreement (PDF) into a categorized, source-linked compliance
checklist. Built for the JLL hackathon, designed to live entirely within
JLL-sanctioned environments.

`parse → chunk → extract → reduce → checklist → UI / export`

---

## System requirements

- **Python 3.10–3.12** (3.11 recommended)
- **pip** and ~1 GB free disk for dependencies
- **Anthropic API key** (hackathon sponsorship) — only needed for live mode
- **No Node.js needed** to run the app (Node was only used to generate the docs/deck)
- Any OS (macOS / Linux / Windows)

## Files

| File | Role |
|------|------|
| `app.py` | Streamlit UI — upload, checklist, click-to-source, export |
| `providers.py` | Model adapters: Claude (sandbox) / Falcon (prod) / JLL GPT + governance guard |
| `idp_extraction.py` | Obligation schema, prompt, per-clause extraction + priority |
| `parse_chunk.py` | PDF → structure-aware `ClauseChunk` list |
| `reduce_obligations.py` | Dedup + build the final checklist |
| `demo_obligations.json` | Cached **synthetic** demo data (UI runs with zero setup) |
| `requirements.txt` | Python dependencies |

---

## First steps

### 1. Put all files in one folder, then set up a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify the pieces — no API key required
Each module has a self-test. Run them top to bottom; all should pass.
```bash
python parse_chunk.py          # parse/chunk self-tests
python reduce_obligations.py   # dedup demo (16 → 14 records)
python providers.py            # provider + governance-guard self-tests
python idp_extraction.py       # offline schema/validation check
```

### 3. Run the app in demo mode — still no API key
```bash
streamlit run app.py
```
Opens `localhost:8501` with the **Demo dataset** selected. Fully clickable — this is
your judging-safe path and proves the whole UI end to end.

### 4. Enable live mode (sandbox — synthetic contracts only)
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Windows: set ANTHROPIC_API_KEY=...
streamlit run app.py
```
Choose **Upload MSA (live)** and upload a **clean, text-based sample** PDF.

---

## ⚠️ The sandbox rule (non-negotiable)

The sponsored Claude access is **cleared for synthetic / sample contracts only**.
**Never upload real client MSAs** in this app. The code enforces this with a
governance guard (`assert_data_allowed`) that blocks real-flagged data from any
endpoint not cleared for it. Production swaps the adapter to **JLL Falcon**, which is
sanctioned and data-cleared, so real contracts stay inside JLL's governed envelope.

---

## Architecture & run order

```
parse_chunk.parse_and_chunk(pdf)          → List[ClauseChunk]
        ↓
idp_extraction.extract_all(chunks, provider)   → List[Obligation]   (map)
        ↓
reduce_obligations.reduce_obligations(records) → deduped records    (reduce)
        ↓
reduce_obligations.build_checklist(...)        → checklist dict
        ↓
app.py                                          → UI + Excel/CSV export
```

The **only** endpoint-specific code is `providers.py`. Everything else is identical
between sandbox and production.

## Production path (post-hackathon)

1. Implement `FalconProvider.extract()` in `providers.py` against JLL Falcon's
   inference endpoint — return the same `obligations` list shape per the schema.
2. In `app.py` `run_pipeline`, swap `ClaudeProvider()` for `get_provider("falcon")`.
3. Nothing downstream changes — parse, schema, reduce, checklist, UI, export all stand.

---

## Troubleshooting

- **`import fitz` fails** → `pip install pymupdf` (the package imports as `fitz`). Make
  sure no unrelated package literally named `fitz` is installed.
- **Model name mismatch** → `providers.py` sets `DEFAULT_MODEL = "claude-sonnet-4-6"`.
  If the sponsorship exposes a different model id, change it there.
- **Port in use** → `streamlit run app.py --server.port 8502`.
- **`tiktoken` not installed** → optional; token counts fall back to a safe heuristic.
- **Live run errors** → the app auto-falls back to the demo dataset and shows the
  reason, so a failed live call never breaks a presentation. Check the key and model name.

## Demo-day playbook

- Present from the **Demo dataset** — it can't be broken by a network hiccup.
- Show the live upload **once** with a synthetic PDF as the "it runs on real PDFs" beat.
- Land on the **expanded obligation view** (source clause + verbatim snippet) — that's
  the trust moment that answers "why would legal believe an AI?"
