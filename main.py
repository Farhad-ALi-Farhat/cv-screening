"""
Entry point for running the CV screening pipeline.

To point at a JD file instead of pasted text, use jd_file_path instead of
jd_raw_text in initial_state, e.g.:

    initial_state = {
        "jd_file_path": "path/to/job_description.pdf",
        "candidates": [...],
        "threshold": 70.0,
    }

(jd_raw_text and jd_file_path are mutually exclusive — jd_file_path wins if
both are set, since load_jd_node overwrites jd_raw_text when a file is given.)
"""

from graph import build_graph
from nodes import build_dataframes
from schemas import ScreeningState


if __name__ == "__main__":
    app = build_graph()

    initial_state: ScreeningState = {
        # Stress-test JD: compound/ambiguous requirements (an open-ended "or
        # related field", a vague "modern data stack" phrase, a soft skill
        # that's hard to verify from text, and a preferred bullet naming two
        # alternative technologies).
        "jd_file_path": "./sample_jd_senior_data_engineer.pdf",
        "candidates": [
            # Point these at your 4 actual CV files (mix of PDF/DOCX is fine).
            {"candidate_id": "cand_1", "file_path": "./ahsan_malik_cv.pdf"},
            {"candidate_id": "cand_2", "file_path": "./bilal_khan_cv.pdf"},
            {"candidate_id": "cand_3", "file_path": "./hamza_raza_cv.pdf"},
            {"candidate_id": "cand_4", "file_path": "./sara_ahmed_cv.pdf"},
            # Stress-test candidate: a strong generalist full-stack engineer
            # with adjacent-but-not-quite data engineering experience,
            # supplied as raw_text so no file is needed.
            {
                "candidate_id": "cand_5_generalist",
                "raw_text": (
                    "Usman Tariq\n"
                    "Full-Stack Software Engineer\n\n"
                    "usman.tariq@example.com | +92 305 555 0198 | Islamabad, Pakistan\n\n"
                    "PROFESSIONAL SUMMARY\n"
                    "Full-stack software engineer with 6 years of experience building web "
                    "applications and internal tooling. Proficient in Python, JavaScript, and "
                    "SQL. Built and maintained several REST APIs and background job pipelines "
                    "using Celery and PostgreSQL. Some exposure to AWS (S3, EC2, Lambda) for "
                    "internal automation tools. Enjoys mentoring interns and collaborating "
                    "across teams on cross-functional projects.\n\n"
                    "EXPERIENCE\n"
                    "Senior Software Engineer, TechNova (2020-Present)\n"
                    "- Led development of internal reporting dashboards consuming data from "
                    "PostgreSQL and periodic CSV exports\n"
                    "- Built scheduled batch jobs (Celery + cron) to aggregate transactional "
                    "data nightly for the analytics team\n"
                    "- Mentored two junior engineers over the past year\n"
                    "- Collaborated closely with the data and product teams to define "
                    "reporting requirements\n\n"
                    "Software Engineer, Bright Labs (2018-2020)\n"
                    "- Built and maintained backend services in Python/Django\n"
                    "- Migrated legacy scripts to AWS Lambda for scheduled automation\n\n"
                    "EDUCATION\n"
                    "BS Computer Science — COMSATS University, 2018\n"
                ),
            },
        ],
        "threshold": 70.0,
    }

    final_state = app.invoke(initial_state)

    print("JD parsed:", final_state["jd_parsed"])
    print()

    summary_df, detail_df = build_dataframes(final_state["report"])
    print("Summary:")
    print(summary_df.to_string(index=False))
    print()
    print("Detail:")
    print(detail_df.to_string(index=False))
