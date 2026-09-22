"""All LangGraph node implementations for the CV screening pipeline.

Design decisions baked into this module (see the project write-up for the
full rationale):
- One JD at a time (no cross-job matching, no per-job vector store yet).
- JD requirements are NOT a fixed hardcoded schema — the LLM derives whatever
  categories the JD actually has.
- No RAG/embeddings yet. At small candidate counts, full JD + full CV text
  fits in one prompt per candidate. Retrieval gets added later only when CV
  volume/length makes full-context stuffing impractical.
- Scoring weight comes from the JD's own required/preferred split, not an
  abstract category weighting scheme.
"""

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from config import (
    DEFAULT_REVIEW_MARGIN,
    DEFAULT_THRESHOLD,
    JD_MODEL_MAX_TOKENS,
    JD_MODEL_NAME,
    MATCH_MODEL_NAME,
)

from schemas import (
    CandidateMatchResult,
    CandidateScore,
    JDParsed,
    MatchOutput,
    RequirementVerdict,
    ScreeningState,
)


# ---------------------------------------------------------------------------
# load_jd
# ---------------------------------------------------------------------------

def load_jd_node(state: ScreeningState) -> ScreeningState:
    """
    Resolves the JD into raw text. If jd_file_path is given, converts it via
    MarkItDown (same tool used for CVs) and that becomes jd_raw_text;
    otherwise jd_raw_text is assumed to already be provided directly (pasted
    text). Runs first, before parse_jd needs the text.
    """
    if state.get("jd_file_path"):
        from markitdown import MarkItDown
        converter = MarkItDown()
        result = converter.convert(state["jd_file_path"])
        return {"jd_raw_text": result.text_content}
    return {}


# ---------------------------------------------------------------------------
# parse_jd
# ---------------------------------------------------------------------------

JD_PARSE_SYSTEM_PROMPT = """You are analyzing a job description (JD) for an HR screening system.

Read the JD closely and break it down into individual, atomic requirements.
Do NOT force it into a fixed template — decide the categories yourself, based
on what this specific JD actually contains. Typical categories include things
like technical skills, experience, education, responsibilities, certifications,
soft skills, etc., but only include categories the JD actually has, and feel
free to use a different label if that fits the JD better.

For each requirement:
- category: a short free-text label you choose (e.g. "Technical Skills", "Experience")
- description: the requirement itself, close to the JD's own wording
- importance: "required" if the JD frames it as a must-have (e.g. "must have",
  "required", listed under core/minimum qualifications), or "preferred" if the
  JD frames it as a bonus/nice-to-have (e.g. "preferred", "a plus", "bonus")

Split compound requirements into separate atomic ones — e.g. "5+ years Python
and AWS experience" should become two requirements, not one, so each can be
checked against a CV independently.

If the JD states a job title, extract it; otherwise leave it null."""


def parse_jd_node(state: ScreeningState) -> ScreeningState:
    """
    Calls the LLM with the raw JD text and asks it to derive its own
    requirement categories (no fixed schema), returning a JDParsed object
    via structured output.
    """
    # This is a reasoning model on Groq — it burns hidden "thinking" tokens
    # before producing the structured output, which is what hit the
    # output-tokens-per-minute rate limit on straightforward extraction tasks
    # like this one. reasoning_effort="none" disables that for a plain
    # extraction call; bump it back up only if extraction quality suffers.
    llm = ChatGroq(
        model=JD_MODEL_NAME,
        temperature=0,
        max_tokens=JD_MODEL_MAX_TOKENS,  # keep well under the on-demand tier's 1000 OTPM cap
        reasoning_effort="none",
    )
    # function_calling is the workaround for Groq's native json-mode/tool-calling
    # conflict noted in the earlier project.
    structured_llm = llm.with_structured_output(JDParsed, method="function_calling")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JD_PARSE_SYSTEM_PROMPT),
            ("human", "Job description:\n\n{jd_text}"),
        ]
    )

    chain = prompt | structured_llm
    jd_parsed: JDParsed = chain.invoke({"jd_text": state["jd_raw_text"]})

    return {"jd_parsed": jd_parsed}


# ---------------------------------------------------------------------------
# convert_cvs
# ---------------------------------------------------------------------------

def convert_cvs_node(state: ScreeningState) -> ScreeningState:
    """
    For each candidate, convert the CV file (PDF/DOCX/etc.) to clean
    Markdown text using MarkItDown. No LLM call here — pure conversion.
    If a candidate provides raw_text directly, conversion is skipped for
    them (used for stress-testing without a real file).

    Note: this is a per-file conversion failure point (scanned/image-only
    PDFs, weird DOCX formatting, etc.). For now a failed conversion raises;
    proper exception handling (flag-for-review instead of crashing the whole
    batch) is a deferred item.
    """
    from markitdown import MarkItDown

    converter = MarkItDown()
    cv_texts = {}
    for candidate in state["candidates"]:
        cid = candidate["candidate_id"]
        if candidate.get("raw_text"):
            cv_texts[cid] = candidate["raw_text"]
            continue
        try:
            result = converter.convert(candidate["file_path"])
            cv_texts[cid] = result.text_content
        except Exception as exc:
            print(f"[convert_cvs] {cid} conversion failed, flagging for review: {exc}")
            cv_texts[cid] = None
    return {"cv_texts": cv_texts}


# ---------------------------------------------------------------------------
# match_candidates
# ---------------------------------------------------------------------------

MATCH_SYSTEM_PROMPT = """You are screening a candidate's CV against a numbered list of job requirements.

For each requirement number, decide:
- "matched": the CV clearly demonstrates this requirement is met
- "partial": there's genuine adjacent or transferable evidence, but it doesn't fully meet the requirement (e.g. fewer years than asked, a closely related tool in the same category)
- "missing": no evidence in the CV

Be strict about what counts as "partial" — it must be evidence of genuinely related work, not just a shared word or vague thematic overlap. For example, a CV mentioning "real-time" in the context of model inference speed is NOT evidence for a requirement about real-time streaming systems like Kafka/Flink — those are unrelated concepts that happen to share a word. Likewise, general cloud experience (e.g. AWS S3/EC2) is NOT evidence for a specific cloud data warehouse requirement (BigQuery/Redshift/Snowflake) unless the CV actually names a data warehouse. If unsure whether evidence genuinely supports the requirement, mark it "missing" rather than stretching to "partial". Your verdict must follow logically from your own evidence string — if your evidence says something is absent, not named, or not mentioned, the verdict must be "missing", not "partial" or "matched". Re-read your evidence against your verdict before finalizing each judgement.

For "matched" or "partial", you MUST include a short evidence string: a plain paraphrase, under 10 words, in your own words — do NOT use quotation marks or copy exact CV text. For "missing", leave evidence null.

Return a judgement for every requirement number listed, even if missing."""


def match_candidates_node(state: ScreeningState) -> ScreeningState:
    """
    For each candidate, sends the full JD requirement list + full CV text in
    one call and asks for a per-requirement verdict (matched/partial/missing)
    with brief evidence. No RAG yet — full CV text fits in context fine at
    this scale.

    Moved from Groq (gpt-oss-120b) to Gemini after gpt-oss-120b showed
    persistent non-determinism across repeated runs (self-contradictory
    verdict/evidence pairs, wording-driven verdict flips) — see project notes
    for the comparison. Gemini hasn't shown the contradiction pattern in
    testing so far, but only light repeat-run testing has been done; keep an
    eye on it rather than treating it as fully resolved.
    """
    # temperature isn't passed here — gemini-3.5-flash-lite ignores it and
    # logs a warning (fixed sampling defaults on this model).
    llm = ChatGoogleGenerativeAI(model=MATCH_MODEL_NAME)
    structured_llm = llm.with_structured_output(MatchOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MATCH_SYSTEM_PROMPT),
            ("human", "Requirements:\n{requirements}\n\nCandidate CV:\n{cv_text}"),
        ]
    )
    chain = prompt | structured_llm

    requirements = state["jd_parsed"].requirements
    numbered_requirements = "\n".join(
        f"{i}. [{req.importance}] {req.description}" for i, req in enumerate(requirements)
    )

    match_results = {}
    for candidate in state["candidates"]:
        cid = candidate["candidate_id"]
        cv_text = state["cv_texts"].get(cid)

        if cv_text is None:
            print(f"[match_candidates] {cid} has no CV text, flagging for review")
            match_results[cid] = CandidateMatchResult(candidate_id=cid, verdicts=[])
            continue

        try:
            result: MatchOutput = chain.invoke({"requirements": numbered_requirements, "cv_text": cv_text})
            judgement_by_index = {j.requirement_index: j for j in result.judgements}
        except Exception as exc:
            print(f"[match_candidates] {cid} failed, flagging for review: {exc}")
            match_results[cid] = CandidateMatchResult(candidate_id=cid, verdicts=[])
            continue

        verdicts = []
        for i, req in enumerate(requirements):
            judgement = judgement_by_index.get(i)
            verdicts.append(
                RequirementVerdict(
                    category=req.category,
                    description=req.description,
                    importance=req.importance,
                    verdict=judgement.verdict if judgement else "missing",
                    evidence=judgement.evidence if judgement else None,
                )
            )
        match_results[cid] = CandidateMatchResult(candidate_id=cid, verdicts=verdicts)

    return {"match_results": match_results}


# ---------------------------------------------------------------------------
# score_candidates
# ---------------------------------------------------------------------------

def score_candidates_node(state: ScreeningState) -> ScreeningState:
    """
    Pure Python, no LLM. Weight = required/preferred split from the JD itself:
      - required:  matched = 1.0, partial = 0.5, missing = 0.0
      - preferred: matched = 0.5, partial = 0.25, missing = 0.0
    Score = (sum of earned weight / sum of possible weight) * 100.

    TODO: tune the exact weight values as more real JDs get tested; this is
    a reasonable starting point, not a final rubric.
    """
    verdict_weights = {"matched": 1.0, "partial": 0.5, "missing": 0.0}
    importance_multiplier = {"required": 1.0, "preferred": 0.5}

    threshold = state.get("threshold", DEFAULT_THRESHOLD)
    margin = state.get("review_margin", DEFAULT_REVIEW_MARGIN)
    scores = {}
    for cid, result in state["match_results"].items():
        if not result.verdicts:
            scores[cid] = CandidateScore(
                candidate_id=cid,
                score=0.0,
                status="Needs Review",
                review_reason="Could not be automatically screened (CV processing or matching failed) — needs manual review.",
            )
            continue

        earned = 0.0
        possible = 0.0
        for v in result.verdicts:
            mult = importance_multiplier[v.importance]
            possible += mult
            earned += mult * verdict_weights[v.verdict]
        pct = round((earned / possible * 100) if possible > 0 else 0.0, 1)

        if pct >= threshold + margin:
            status, review_reason = "Shortlisted", None
        elif pct >= threshold - margin:
            status, review_reason = "Needs Review", "Borderline score — recommend manual review."
        else:
            status, review_reason = "Not shortlisted", None

        scores[cid] = CandidateScore(candidate_id=cid, score=pct, status=status, review_reason=review_reason)
    return {"scores": scores}


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

def build_report_node(state: ScreeningState) -> ScreeningState:
    """
    Flatten everything into rows HR can look at: score, status, and the
    per-requirement matched/missing breakdown with evidence.
    """
    report = []
    for cid, score in state["scores"].items():
        match_result = state["match_results"][cid]
        report.append(
            {
                "candidate_id": cid,
                "score": score.score,
                "status": score.status,
                "review_reason": score.review_reason,
                "requirements": [v.model_dump() for v in match_result.verdicts],
            }
        )
    return {"report": report}


def build_dataframes(report: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Turns the report into two DataFrames for actual display:
      - summary_df: one row per candidate (score, status, review_reason)
      - detail_df: one row per candidate per requirement (the full breakdown)
    """
    summary_rows = [
        {
            "candidate_id": row["candidate_id"],
            "score": row["score"],
            "status": row["status"],
            "review_reason": row.get("review_reason"),
        }
        for row in report
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values("score", ascending=False).reset_index(drop=True)

    detail_rows = []
    for row in report:
        for req in row["requirements"]:
            detail_rows.append({"candidate_id": row["candidate_id"], **req})
    detail_df = pd.DataFrame(detail_rows)

    return summary_df, detail_df
