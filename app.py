"""
Streamlit UI for the CV screening pipeline.

Run with: streamlit run app.py
Needs: pip install streamlit
"""

import tempfile
from pathlib import Path

import streamlit as st

from config import DEFAULT_REVIEW_MARGIN, DEFAULT_THRESHOLD
from graph import build_graph
from schemas import ScreeningState

st.set_page_config(page_title="CV Screening", layout="wide")
st.title("CV Screening")

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

st.subheader("1. Job Description")
jd_input_mode = st.radio("JD input", ["Paste text", "Upload file"], horizontal=True)

jd_raw_text = None
jd_file_path = None

if jd_input_mode == "Paste text":
    jd_raw_text = st.text_area("Paste the job description", height=200)
else:
    jd_file = st.file_uploader("Upload JD (PDF or DOCX)", type=["pdf", "docx"])
    if jd_file is not None:
        suffix = Path(jd_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(jd_file.getvalue())
            jd_file_path = tmp.name

st.subheader("2. Candidate CVs")
cv_files = st.file_uploader(
    "Upload candidate CVs (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True
)

st.subheader("3. Scoring settings")
col1, col2 = st.columns(2)
with col1:
    threshold = st.slider("Shortlist threshold", 0, 100, int(DEFAULT_THRESHOLD))
with col2:
    review_margin = st.slider("Review margin (+/- around threshold)", 0, 20, int(DEFAULT_REVIEW_MARGIN))

run = st.button("Run screening", type="primary")

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if run:
    if not jd_raw_text and not jd_file_path:
        st.error("Provide a JD — paste text or upload a file.")
        st.stop()
    if not cv_files:
        st.error("Upload at least one candidate CV.")
        st.stop()

    candidates = []
    for i, cv_file in enumerate(cv_files):
        suffix = Path(cv_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(cv_file.getvalue())
            candidate_id = Path(cv_file.name).stem or f"candidate_{i}"
            candidates.append({"candidate_id": candidate_id, "file_path": tmp.name})

    initial_state: ScreeningState = {
        "candidates": candidates,
        "threshold": float(threshold),
        "review_margin": float(review_margin),
    }
    if jd_file_path:
        initial_state["jd_file_path"] = jd_file_path
    else:
        initial_state["jd_raw_text"] = jd_raw_text

    with st.spinner("Screening candidates — this calls the LLM once per candidate, may take a moment..."):
        pipeline = build_graph()
        final_state = pipeline.invoke(initial_state)

    st.session_state["final_state"] = final_state

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if "final_state" in st.session_state:
    final_state = st.session_state["final_state"]
    jd_parsed = final_state["jd_parsed"]

    st.subheader("Parsed JD requirements")
    with st.expander(f"{jd_parsed.job_title or 'Job'} — {len(jd_parsed.requirements)} requirements"):
        for req in jd_parsed.requirements:
            st.markdown(f"- **[{req.importance}]** ({req.category}) {req.description}")

    st.subheader("Results")

    status_order = {"Shortlisted": 0, "Needs Review": 1, "Not shortlisted": 2}
    status_icon = {"Shortlisted": "🟢", "Needs Review": "🟡", "Not shortlisted": "🔴"}
    verdict_icon = {"matched": "✅", "partial": "🟡", "missing": "❌"}

    report = sorted(
        final_state["report"],
        key=lambda r: (status_order.get(r["status"], 3), -r["score"]),
    )

    for row in report:
        icon = status_icon.get(row["status"], "")
        header = f"{icon}  {row['candidate_id']} — {row['score']} — {row['status']}"
        with st.expander(header):
            if row.get("review_reason"):
                st.info(row["review_reason"])
            for req in row["requirements"]:
                v_icon = verdict_icon.get(req["verdict"], "")
                st.markdown(f"{v_icon} **[{req['importance']}] {req['category']}** — {req['description']}")
                if req.get("evidence"):
                    st.caption(f"Evidence: {req['evidence']}")
