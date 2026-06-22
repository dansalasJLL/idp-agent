"""
app.py — Streamlit UI for the IDP Agent
JLL Hackathon 2026

Run:
    streamlit run app.py

Modes:
  - Demo mode (no API key): loads demo_obligations.json instantly
  - Live mode (ANTHROPIC_API_KEY set): upload a PDF, run full pipeline

Features:
  - Upload MSA PDF → extract obligations
  - Filter by Priority / Category / Party / Needs Review
  - Click any row to see source clause + verbatim snippet
  - Export to Excel or CSV
  - Governance guard: blocks real client data from sandbox endpoint
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="IDP Agent · JLL Hackathon 2026",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Local imports (after page config)
# ---------------------------------------------------------------------------

from idp_extraction import Obligation, Priority, Category, Party
from reduce_obligations import reduce, build_checklist, to_dataframe, export_excel

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------

DEMO_JSON_PATH = Path(__file__).parent / "demo_obligations.json"
PRIORITY_COLORS = {
    Priority.HIGH.value:   "🔴",
    Priority.MEDIUM.value: "🟡",
    Priority.LOW.value:    "🟢",
}
REVIEW_BADGE = "⚠️ Review"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_demo_obligations() -> List[Obligation]:
    """Load cached demo obligations from JSON."""
    if not DEMO_JSON_PATH.exists():
        st.error(f"Demo file not found: {DEMO_JSON_PATH}. Run the self-test to generate it.")
        return []
    with open(DEMO_JSON_PATH) as f:
        raw = json.load(f)
    return [Obligation.model_validate(item) for item in raw]


def _run_live_pipeline(pdf_bytes: bytes, pdf_name: str) -> List[Obligation]:
    """Run the full parse → extract → reduce pipeline on an uploaded PDF."""
    import tempfile
    from parse_chunk import parse_and_chunk
    from idp_extraction import extract_all
    from providers import get_provider, assert_data_allowed

    provider = get_provider("claude")
    assert_data_allowed(provider, contains_real_client_data=False)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        with st.spinner("📄 Parsing PDF into clause chunks…"):
            chunks = parse_and_chunk(tmp_path)
        st.info(f"Found **{len(chunks)}** clause chunks. Starting extraction…")

        progress = st.progress(0, text="Extracting obligations…")
        obligations: List[Obligation] = []
        for i, chunk in enumerate(chunks):
            from idp_extraction import extract_obligations_from_chunk
            obs = extract_obligations_from_chunk(chunk, provider)
            obligations.extend(obs)
            progress.progress((i + 1) / len(chunks),
                              text=f"[{i+1}/{len(chunks)}] {chunk.section_id}: {len(obs)} obligation(s)")

        progress.empty()
        return reduce(obligations)

    finally:
        os.unlink(tmp_path)


def _priority_sort_key(p: str) -> int:
    return {Priority.HIGH.value: 0, Priority.MEDIUM.value: 1, Priority.LOW.value: 2}.get(p, 9)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(obligations: List[Obligation]):
    """Render filter controls and return filtered list."""
    st.sidebar.header("🔍 Filters")

    # Priority
    all_priorities = [Priority.HIGH.value, Priority.MEDIUM.value, Priority.LOW.value]
    sel_priority = st.sidebar.multiselect(
        "Priority", all_priorities, default=all_priorities,
        format_func=lambda p: f"{PRIORITY_COLORS[p]} {p}"
    )

    # Category
    all_cats = sorted({ob.category for ob in obligations})
    sel_cats = st.sidebar.multiselect("Category", all_cats, default=all_cats)

    # Responsible party
    all_parties = sorted({ob.responsible_party for ob in obligations})
    sel_parties = st.sidebar.multiselect("Responsible Party", all_parties, default=all_parties)

    # Needs review
    review_only = st.sidebar.checkbox("⚠️ Needs Review only", value=False)

    st.sidebar.divider()
    st.sidebar.caption("JLL Hackathon 2026 · IDP Agent · Daniel Salas Castro")

    # Apply filters
    filtered = [
        ob for ob in obligations
        if ob.priority in sel_priority
        and ob.category in sel_cats
        and ob.responsible_party in sel_parties
        and (not review_only or ob.needs_review)
    ]
    return filtered


# ---------------------------------------------------------------------------
# Main checklist table
# ---------------------------------------------------------------------------

def render_checklist(filtered: List[Obligation]):
    """Render the main obligation checklist table."""
    if not filtered:
        st.warning("No obligations match the current filters.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    high   = sum(1 for o in filtered if o.priority == Priority.HIGH.value)
    medium = sum(1 for o in filtered if o.priority == Priority.MEDIUM.value)
    low    = sum(1 for o in filtered if o.priority == Priority.LOW.value)
    review = sum(1 for o in filtered if o.needs_review)

    col1.metric("🔴 High",     high)
    col2.metric("🟡 Medium",   medium)
    col3.metric("🟢 Low",      low)
    col4.metric("⚠️ Review",   review)

    st.divider()

    # Table header
    hcols = st.columns([1, 1.5, 2, 1, 1.5, 1, 1, 0.8])
    headers = ["ID", "Section", "Description", "Category", "Party", "Trigger", "Priority", "Review"]
    for col, h in zip(hcols, headers):
        col.markdown(f"**{h}**")
    st.divider()

    # Rows — clickable via expander
    for ob in filtered:
        pri_icon = PRIORITY_COLORS.get(ob.priority, "⚪")
        review_badge = REVIEW_BADGE if ob.needs_review else ""

        with st.expander(
            f"{pri_icon} **{ob.obligation_id}** · {ob.source_section} · "
            f"{ob.description[:80]}{'…' if len(ob.description) > 80 else ''} "
            f"{review_badge}",
            expanded=False,
        ):
            dcol1, dcol2 = st.columns([1, 1])

            with dcol1:
                st.markdown("**📋 Obligation Details**")
                st.markdown(f"- **ID:** `{ob.obligation_id}`")
                st.markdown(f"- **Section:** {ob.source_section}")
                st.markdown(f"- **Page:** {ob.source_page}")
                st.markdown(f"- **Category:** {ob.category}")
                st.markdown(f"- **Responsible Party:** {ob.responsible_party}")
                st.markdown(f"- **Trigger:** {ob.trigger_type}")
                if ob.deadline:
                    st.markdown(f"- **Deadline:** {ob.deadline}")
                if ob.frequency:
                    st.markdown(f"- **Frequency:** {ob.frequency}")
                if ob.penalty:
                    st.markdown(f"- **⚠️ Penalty:** {ob.penalty}")
                st.markdown(f"- **Priority:** {pri_icon} {ob.priority}")
                st.markdown(f"- **Confidence:** {ob.confidence:.0%}"
                            + (" ⚠️ *Flagged for review*" if ob.needs_review else ""))

            with dcol2:
                st.markdown("**📄 Full Description**")
                st.info(ob.description)
                st.markdown("**🔗 Verbatim Source**")
                st.code(ob.verbatim_snippet, language=None)


# ---------------------------------------------------------------------------
# Export section
# ---------------------------------------------------------------------------

def render_export(filtered: List[Obligation]):
    """Render Excel and CSV export buttons."""
    st.subheader("📤 Export")
    ecol1, ecol2 = st.columns(2)

    df = to_dataframe(filtered)
    if df is None:
        st.warning("pandas not installed — export unavailable. `pip install pandas openpyxl`")
        return

    # CSV
    with ecol1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name="obligations.csv",
            mime="text/csv",
        )

    # Excel
    with ecol2:
        import io
        try:
            import openpyxl  # noqa: F401
            buf = io.BytesIO()
            with __import__("pandas").ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Obligations")
            st.download_button(
                label="⬇️ Download Excel",
                data=buf.getvalue(),
                file_name="obligations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except ImportError:
            st.info("Install openpyxl for Excel export: `pip install openpyxl`")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.title("📄 IDP Agent — Intelligent Document Processing")
    st.caption("JLL Hackathon 2026 · Automated MSA Obligation Extraction")

    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # --- Mode selector ---
    mode = st.radio(
        "Mode",
        options=["🎬 Demo (no API key)", "🚀 Live (upload PDF)"],
        horizontal=True,
        disabled=not has_api_key and False,  # demo always available
    )

    obligations: List[Obligation] = []

    # -----------------------------------------------------------------------
    # DEMO MODE
    # -----------------------------------------------------------------------
    if mode == "🎬 Demo (no API key)":
        st.info(
            "Running in **demo mode** — using cached synthetic obligations. "
            "No API key or PDF required.",
            icon="ℹ️",
        )
        obligations = _load_demo_obligations()
        if obligations:
            st.success(f"Loaded **{len(obligations)}** synthetic obligations from demo dataset.")

    # -----------------------------------------------------------------------
    # LIVE MODE
    # -----------------------------------------------------------------------
    else:
        if not has_api_key:
            st.error(
                "⚠️ `ANTHROPIC_API_KEY` environment variable not set. "
                "Set it and restart the app to use live mode.",
                icon="🔐",
            )
            st.stop()

        st.warning(
            "⚠️ **Sandbox mode — synthetic/sample PDFs only.** "
            "Do NOT upload real client MSAs. The governance guard will block it.",
            icon="🔒",
        )

        uploaded = st.file_uploader(
            "Upload MSA PDF (text-based, synthetic/sample only)",
            type=["pdf"],
            help="Scanned PDFs require Azure OCR — see README.",
        )

        if uploaded:
            if st.button("🚀 Extract Obligations", type="primary"):
                try:
                    obligations = _run_live_pipeline(uploaded.read(), uploaded.name)
                    st.success(f"✅ Extracted **{len(obligations)}** obligations.")
                    # Cache in session so filters don't re-run the pipeline
                    st.session_state["live_obligations"] = obligations
                except PermissionError as e:
                    st.error(f"🔒 Governance block: {e}")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.info("Falling back to demo dataset…")
                    obligations = _load_demo_obligations()

        elif "live_obligations" in st.session_state:
            obligations = st.session_state["live_obligations"]

    # -----------------------------------------------------------------------
    # CHECKLIST (shared by both modes)
    # -----------------------------------------------------------------------
    if obligations:
        st.divider()
        filtered = render_sidebar(obligations)
        st.subheader(f"📋 Obligation Checklist ({len(filtered)} of {len(obligations)})")
        render_checklist(filtered)
        st.divider()
        render_export(filtered)
    else:
        st.info("Upload a PDF and click **Extract Obligations** to begin, or switch to Demo mode.")


if __name__ == "__main__":
    main()
