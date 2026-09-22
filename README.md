# CV Screening Pipeline

An AI-powered first-pass CV screening system for technical hiring. Give it a job description and a batch of candidate CVs, and it produces an explainable, per-requirement match breakdown with a weighted score and a shortlist recommendation — while leaving the final hiring decision to HR.

## What it does

- Accepts a JD as pasted text or a PDF/DOCX file — nothing about the JD's structure is hardcoded. The LLM reads the JD itself and derives whatever requirement categories it actually contains, tagging each as `required` or `preferred` based on the JD's own language.
- Converts candidate CVs (PDF/DOCX) to clean text via [MarkItDown](https://github.com/microsoft/markitdown).
- Matches each candidate against the JD's requirements one at a time — no cross-candidate or cross-job matching. For each requirement, the matcher returns `matched` / `partial` / `missing` plus a short evidence string, so every verdict is traceable back to something in the CV.
- Produces a weighted score per candidate: required requirements count double a preferred one, and `matched`/`partial`/`missing` earn full/half/no credit respectively. The score is `earned / possible weight × 100`, not an arbitrary single similarity number.
- Applies a **three-tier outcome** instead of a hard cutoff: `Shortlisted`, `Needs Review` (score within a configurable band around the threshold, or the CV/matching process itself failed), or `Not shortlisted`. Borderline candidates and processing failures both get flagged for a human to look at rather than silently defaulting to a rejection.
- Ships as both a CLI script (`main.py`) and a Streamlit app (`app.py`).

## Architecture

```
START -> load_jd -> parse_jd -> convert_cvs -> match_candidates -> score_candidates -> build_report -> END
```

A linear [LangGraph](https://github.com/langchain-ai/langgraph) pipeline. No RAG/embeddings layer yet — at small-to-moderate candidate counts, the full JD requirement list and full CV text fit comfortably in one prompt per candidate. Retrieval is a planned addition for when that stops being true (see Limitations).

| Node | Does |
|---|---|
| `load_jd` | Resolves the JD into raw text, converting a PDF/DOCX via MarkItDown if `jd_file_path` is given |
| `parse_jd` | LLM extracts atomic, dynamically-categorized requirements from the JD |
| `convert_cvs` | Converts each candidate's CV file to text via MarkItDown (or uses `raw_text` directly if supplied) |
| `match_candidates` | LLM judges each candidate against every requirement, with evidence |
| `score_candidates` | Pure Python — computes the weighted score and assigns the three-tier status |
| `build_report` | Flattens everything into a per-candidate, per-requirement report |

## Project structure

```
config.py     — model names and tunable defaults (threshold, review margin, etc.)
schemas.py    — Pydantic models and the LangGraph state definition
nodes.py      — all pipeline node implementations
graph.py      — wires the nodes into the LangGraph pipeline
main.py       — CLI entry point / example run
app.py        — Streamlit UI
compare_gemini.py — standalone script for A/B testing a different matcher model against the same JD/CVs
sample_jd_*.pdf, sample_cv_*.pdf — synthetic test fixtures (see below)
```

## Setup

```bash
git clone <your-repo-url>
cd cv-screening
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env      # then fill in your own API keys
```

Requires a [Groq](https://console.groq.com/keys) API key (used for JD parsing) and a [Google AI Studio](https://aistudio.google.com/apikey) key (used for CV matching).

## Usage

**CLI:**
```bash
python main.py
```
Edit the `initial_state` dict in `main.py` to point at your own JD and CV files.

**Streamlit app:**
```bash
streamlit run app.py
```
Upload a JD (paste text or PDF/DOCX) and candidate CVs, set the shortlist threshold and review margin, and run.

## Configuration

Model choices and defaults live in `config.py`:
- `JD_MODEL_NAME` / `JD_MODEL_MAX_TOKENS` — JD parsing model (Groq)
- `MATCH_MODEL_NAME` — CV matching model (Gemini)
- `DEFAULT_THRESHOLD` / `DEFAULT_REVIEW_MARGIN` — shortlist cutoff and the band around it that triggers `Needs Review`

## Sample test data

The `sample_jd_*.pdf` and `sample_cv_*.pdf` files are synthetic (LLM-generated) fixtures for testing — not real people. They cover a deliberately mismatched JD (for stress-testing ambiguous judgment calls), a well-matched JD, and CVs spanning strong fit, weak fit, and a completely off-domain candidate. Do not add real candidate CVs to this repo — keep production data out of version control.

## Known limitations

- **Single JD at a time.** No per-job vector store or cross-job isolation yet — deliberately deferred until multi-job support is needed.
- **No RAG/embeddings.** Full CV text is passed to the matcher directly. Fine at small candidate counts; will need retrieval once CV volume or length makes full-context prompting impractical.
- **LLM non-determinism.** Identical inputs can produce slightly different verdicts across runs (observed on both the Groq and Gemini matchers). The three-tier scoring system exists partly to absorb this — treat scores near the threshold band as provisional, not final.
- **Batch processing is a simple sequential loop**, not built for hundreds of CVs yet.
- **Exception handling is minimal.** A failed CV conversion or a failed matching call is caught and flagged as `Needs Review` rather than crashing the batch, but there's no retry logic or detailed failure categorization yet.

## License

Add a license of your choice (MIT is a common default for a project like this) before making the repo public.
