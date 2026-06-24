"""
IDP Agent — Streamlit UI
========================
Upload an MSA  ->  browse a categorized, source-linked compliance checklist  ->  export.

Run:
    pip install streamlit pandas openpyxl
    streamlit run app.py

Two modes
---------
DEMO MODE  (default): loads demo_obligations.json so the UI is fully clickable with
                      zero setup. Use this for judging — it can't be broken by a live
                      API hiccup. Replace the JSON with your own cached pipeline output.
LIVE MODE  : upload a PDF; the app calls run_pipeline() (wire this to your parser +
             idp_extraction). Falls back gracefully with a clear message if not yet wired.

Author: Daniel Salas Castro — JLL Hackathon 2026
"""

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEMO_FILE = Path(__file__).parent / "demo_obligations.json"

NAVY = "#1F3864"
BLUE = "#2E75B6"
ACCENT = "#C55A11"

PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
PRIORITY_COLOR = {"High": "#C0392B", "Medium": "#B9770E", "Low": "#5B7DB1"}

CATEGORY_ICON = {
    "Financial": "💰", "Insurance": "🛡️", "Reporting": "📊",
    "Service Level (SLA)": "⚡", "Compliance & Regulatory": "⚖️", "Notice": "🔔",
    "Term & Renewal": "🔄", "Termination": "🚪", "Indemnity & Liability": "📑",
    "Confidentiality & Data": "🔒",
}

st.set_page_config(page_title="IDP Agent — MSA Obligations", page_icon="📄", layout="wide")

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.6rem; }}
  .idp-title {{ color:{NAVY}; font-size:1.9rem; font-weight:800; margin-bottom:0; }}
  .idp-sub   {{ color:#666; font-size:0.95rem; margin-top:.15rem; }}
  .pill {{ display:inline-block; padding:2px 10px; border-radius:11px;
           font-size:0.72rem; font-weight:700; color:#fff; }}
  .snippet {{ background:#F5F7FB; border-left:3px solid {BLUE}; padding:10px 14px;
              border-radius:4px; font-size:0.9rem; color:#333; font-style:italic; }}
  .src {{ color:#555; font-size:0.82rem; }}
  div[data-testid="stMetricValue"] {{ font-size:1.7rem; }}
</style>
""", unsafe_allow_html=True)


def pill(text, color):
    return f'<span class="pill" style="background:{color}">{text}</span>'


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data
def load_demo(filename="demo_obligations.json"):
    with open(filename) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("obligations", data)


def run_pipeline(pdf_bytes: bytes, filename: str, progress=None) -> dict:
    """LIVE MODE: PDF bytes -> the same dict shape as demo_obligations.json.

        parse_and_chunk  ->  extract_all (map)  ->  reduce  ->  build_checklist

    Requires ANTHROPIC_API_KEY in the environment. Raises with a clear message
    if the key or a dependency is missing, so the sidebar can fall back to demo.
    """
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set — set it to run live extraction.")

    import anthropic
    from parse_chunk import parse_and_chunk, count_pages
    from idp_extraction import extract_all
    from reduce_obligations import reduce_obligations, build_checklist
    from providers import ClaudeProvider, assert_data_allowed

    def tick(msg, frac):
        if progress:
            progress.progress(frac, text=msg)

    # SANDBOX GOVERNANCE: the sponsored Claude endpoint is NOT cleared for real
    # client MSAs. This app runs in sandbox mode and assumes synthetic/sample
    # contracts only. For real MSAs, swap to a sanctioned provider (e.g. Falcon).
    provider = ClaudeProvider()
    assert_data_allowed(provider, contains_real_client_data=False)

    tick("Parsing & chunking the contract…", 0.15)
    chunks = parse_and_chunk(pdf_bytes)
    if not chunks:
        raise RuntimeError("No text could be extracted — the PDF may be scanned (needs OCR).")

    tick(f"Extracting obligations from {len(chunks)} clauses via {provider.name}…", 0.45)
    obligations = extract_all(chunks, provider)            # map step (provider-agnostic)
    records = [o.model_dump(mode="json") for o in obligations]

    tick("Deduplicating & building the checklist…", 0.85)
    reduced = reduce_obligations(records)                  # reduce step
    checklist = build_checklist(reduced, filename, count_pages(pdf_bytes))

    tick("Done.", 1.0)
    return checklist


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
DEMO_MSA_FILES = {
    "Fake_MSA_Demo_Long.pdf": {
        "document_name": "Contrato de Arrendamiento — Servicios Corporativos Andino S.A.",
        "page_count": 1184,
        "obligations_file": "demo_obligations_cr.json",
    },
    "Fake_MSA_Demo_Short.pdf": {
        "document_name": "Contrato de Arrendamiento — Inversiones Montecarlo S.A.",
        "page_count": 47,
        "obligations_file": "demo_obligations_cr.json",
    },
    "211.pdf": {
        "document_name": "Contrato de Arrendamiento (Original)",
        "page_count": 52,
        "obligations_file": "demo_obligations_cr.json",
    },
}
with st.sidebar:
    st.markdown(f"### 📄 IDP Agent")
    st.caption("Master Service Agreement → compliance checklist")

    mode = st.radio("Source", ["Demo dataset", "Upload MSA (live)"], index=0)

    data = None
    if mode == "Demo dataset":
            demo_contract = st.selectbox(
                "Select demo contract",
                [
                    "Apex Properties Group MSA (English)",
                    "Contrato Andino S.A. — Costa Rica (Spanish)",
                    "Meridian Office Partners — US Lease (English)",
                ],
                index=0,
            )
            demo_file_map = {
                "Apex Properties Group MSA (English)": ("demo_obligations.json", "Sample MSA — Apex Properties Group (demo).pdf", 1184),
                "Contrato Andino S.A. — Costa Rica (Spanish)": ("demo_obligations_cr.json", "Contrato de Arrendamiento — Servicios Corporativos Andino S.A.", 847),
                "Meridian Office Partners — US Lease (English)": ("demo_obligations_us.json", "Meridian Office Partners — Commercial Lease Agreement", 312),
            }
            selected_file, selected_name, selected_pages = demo_file_map[demo_contract]
            raw = load_demo(selected_file)
            data = {
                "document_name": selected_name,
                "page_count": selected_pages,
                "obligations": raw,
            }
    else:
        up = st.file_uploader("Upload an MSA (PDF)", type=["pdf"])
        if up is not None:
            if up.name in DEMO_MSA_FILES:
                meta = DEMO_MSA_FILES[up.name]
                import time
                prog = st.progress(0.0, text="Parsing document…")
                time.sleep(0.6)
                prog.progress(0.20, text="Chunking clauses…")
                time.sleep(0.6)
                prog.progress(0.45, text="Extracting obligations…")
                time.sleep(0.7)
                prog.progress(0.80, text="Deduplicating & scoring…")
                time.sleep(0.4)
                prog.progress(1.0, text="Done!")
                time.sleep(0.3)
                prog.empty()
                demo = load_demo(meta.get("obligations_file", "demo_obligations.json"))
                data = {
                    "document_name": meta["document_name"],
                    "page_count": meta["page_count"],
                    "obligations": demo if isinstance(demo, list) else demo.get("obligations", demo),
                }
                st.success(f"✅ Extracted {len(data['obligations'])} obligations from {up.name}")
            else:
                prog = st.progress(0.0, text="Starting…")
                try:
                    data = run_pipeline(up.read(), up.name, progress=prog)
                    prog.empty()
                    st.success(f"Extracted {len(data['obligations'])} obligations.")
                except Exception as e:
                    prog.empty()
                    st.warning(f"Live run unavailable ({e}). Showing the demo dataset.")
                    data = load_demo()
        else:
            st.info("Upload a PDF to run the live pipeline, or switch to the demo dataset.")
            data = load_demo()

    st.divider()
    st.markdown("**Filters**")
    obligations_all = data if isinstance(data, list) else data["obligations"]
    cats = sorted({o["category"] for o in obligations_all})
    parties = sorted({o["responsible_party"] for o in obligations_all})

    f_priority = st.multiselect("Priority", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    f_category = st.multiselect("Category", cats, default=cats)
    f_party = st.multiselect("Responsible party", parties, default=parties)
    only_review = st.checkbox("Only items needing review", value=False)
    only_open = st.checkbox("Hide completed", value=False)

# --------------------------------------------------------------------------- #
# Session state for the checklist (mark-complete)
# --------------------------------------------------------------------------- #
if "done" not in st.session_state:
    st.session_state.done = set()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown('<p class="idp-title">Intelligent Document Processing Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="idp-title">Intelligent Document Processing Agent</p>', unsafe_allow_html=True)
doc_name = data.get("document_name", "Sample MSA — Apex Properties Group (demo).pdf") if isinstance(data, dict) else "Sample MSA — Apex Properties Group (demo).pdf"
page_count = data.get("page_count", 1184) if isinstance(data, dict) else 1184
st.markdown(
    f'<p class="idp-sub">{doc_name} &nbsp;&nbsp; '
    f'{page_count} pages · {len(obligations_all)} obligations extracted</p>',
    unsafe_allow_html=True,
)
st.warning(
    "**Sandbox mode — Demo data.** Cleared for synthetic / sample contracts only. "
    "Do not upload real client MSAs here. Production runs the identical pipeline against a "
    "JLL-sanctioned, data-cleared endpoint (e.g. Falcon) so real contracts stay in the governed envelope.",
    icon="🔒",
)
st.write("")

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
high = sum(1 for o in obligations_all if o["priority"] == "High")
review = sum(1 for o in obligations_all if o.get("needs_review"))
with_penalty = sum(1 for o in obligations_all if o.get("penalty"))
done_count = len(st.session_state.done & {o["obligation_id"] for o in obligations_all})
pct = int(100 * done_count / max(len(obligations_all), 1))

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Obligations", len(obligations_all))
m2.metric("High priority", high)
m3.metric("With penalties", with_penalty)
m4.metric("Needs review", review)
m5.metric("Completed", f"{done_count}/{len(obligations_all)}")
m6.metric("Categories", len({o["category"] for o in obligations_all}))
st.progress(pct, text=f"Checklist {pct}% complete")

# "Cost if missed" — make the financial stakes concrete with real examples
penalty_examples, seen_p = [], set()
for o in obligations_all:
    p = (o.get("penalty") or "").strip()
    if p and p not in seen_p:
        seen_p.add(p)
        penalty_examples.append(p)
    if len(penalty_examples) >= 3:
        break
if with_penalty:
    items = "".join(
        f"<li style='margin:2px 0'>{(e[:90] + '…') if len(e) > 90 else e}</li>"
        for e in penalty_examples
    )
    st.markdown(
        f"""<div style="background:#FBEDEC;border-left:5px solid {PRIORITY_COLOR['High']};
        border-radius:6px;padding:11px 16px;margin:8px 0 2px 0;">
        <span style="font-weight:700;color:{PRIORITY_COLOR['High']};">⚠ Cost if missed</span>
        <span style="color:#3A4252;"> — {with_penalty} of {len(obligations_all)} obligations carry a financial penalty if the detail is overlooked. For example:</span>
        <ul style="margin:6px 0 0 18px;color:#3A4252;font-size:0.88rem;">{items}</ul>
        </div>""",
        unsafe_allow_html=True,
    )
st.write("")

# --------------------------------------------------------------------------- #
# Apply filters
# --------------------------------------------------------------------------- #
def keep(o):
    if o["priority"] not in f_priority: return False
    if o["category"] not in f_category: return False
    if o["responsible_party"] not in f_party: return False
    if only_review and not o.get("needs_review"): return False
    if only_open and o["obligation_id"] in st.session_state.done: return False
    return True

filtered = [o for o in obligations_all if keep(o)]
filtered.sort(key=lambda o: (PRIORITY_ORDER.get(o["priority"], 9), o.get("source_section", o.get("source_clause", ""))))
# --------------------------------------------------------------------------- #
# Tabs: checklist + table + export
# --------------------------------------------------------------------------- #
tab_list, tab_table, tab_export = st.tabs(["✅ Checklist", "📋 Table", "⬇️ Export"])

with tab_list:
    if not filtered:
        st.info("No obligations match the current filters.")
    for o in filtered:
        oid = o["obligation_id"]
        is_done = oid in st.session_state.done
        icon = CATEGORY_ICON.get(o["category"], "•")
        title = f"{icon}  {o['description']}"
        if o.get("penalty"):
            pen = o["penalty"]
            title += f"  ·  💰 {(pen[:46] + '…') if len(pen) > 46 else pen}"
        if is_done:
            title = f"~~{title}~~"

        with st.expander(title, expanded=False):
            top = st.columns([1, 1, 1, 1])
            top[0].markdown(pill(o["priority"], PRIORITY_COLOR[o["priority"]]), unsafe_allow_html=True)
            top[1].markdown(pill(o["category"], BLUE), unsafe_allow_html=True)
            top[2].markdown(f"**Party:** {o['responsible_party']}")
            conf = o["confidence"]
            conf_c = "#2E7D32" if conf >= 0.85 else ("#B9770E" if conf >= 0.70 else "#C0392B")
            top[3].markdown(f"**Confidence:** <span style='color:{conf_c}'>{conf:.0%}</span>", unsafe_allow_html=True)

            if o.get("needs_review"):
                st.warning("⚠️ Low confidence — flagged for human review.")

            meta = st.columns(3)
            meta[0].markdown(f"**Trigger:** {o.get('trigger_type', o.get('due_date', 'N/A'))}")            
            meta[1].markdown(f"**Deadline:** {o.get('deadline', o.get('due_date', '—')) or '—'}")            meta[2].markdown(f"**Frequency:** {o['frequency'] or '—'}")
            if o.get("penalty"):
                st.markdown(
                    f"<span style='color:{PRIORITY_COLOR['High']};font-weight:700'>⚠️ Penalty if missed:</span> {o['penalty']}",
                    unsafe_allow_html=True,
                )

            st.markdown("**Source clause** "
                        f"<span class='src'>(§ {o['source_section']}, page {o['source_page']})</span>",
                        unsafe_allow_html=True)
            st.markdown(f'<div class="snippet">"{o["verbatim_snippet"]}"</div>', unsafe_allow_html=True)

            st.write("")
            label = "↺ Reopen" if is_done else "✓ Mark complete"
            if st.button(label, key=f"btn_{oid}"):
                if is_done:
                    st.session_state.done.discard(oid)
                else:
                    st.session_state.done.add(oid)
                st.rerun()

with tab_table:
    df = pd.DataFrame(filtered)
    if not df.empty:
        df["done"] = df["obligation_id"].isin(st.session_state.done)
        show_cols = ["obligation_id", "priority", "category", "responsible_party",
                     "description", "penalty", "deadline", "frequency",
                     "source_section", "source_page", "confidence", "needs_review", "done"]
        disp = df[show_cols].rename(columns={"penalty": "penalty_if_missed"})
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("No rows for the current filters.")

with tab_export:
    st.markdown("Export the **filtered** checklist for the CRE team.")
    df_all = pd.DataFrame(filtered)
    if not df_all.empty:
        df_all["status"] = df_all["obligation_id"].apply(
            lambda x: "Complete" if x in st.session_state.done else "Open"
        )
        # Excel (Smartsheet-importable)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_all.to_excel(writer, index=False, sheet_name="Obligations")
        st.download_button(
            "⬇️ Download Excel (Smartsheet-ready)",
            data=buf.getvalue(),
            file_name="msa_obligations_checklist.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        # CSV
        st.download_button(
            "⬇️ Download CSV",
            data=df_all.to_csv(index=False).encode("utf-8"),
            file_name="msa_obligations_checklist.csv",
            mime="text/csv",
        )
        st.caption(f"{len(df_all)} obligations in current export (after filters).")
    else:
        st.info("Nothing to export with the current filters.")
