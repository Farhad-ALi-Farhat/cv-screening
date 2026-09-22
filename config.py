from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ — needed for scripts; notebooks often get this some other way



"""Central place for model choices and other tunables."""

# JD parsing model — a reasoning model, but we disable "thinking" for this
# straightforward extraction task to avoid burning hidden reasoning tokens
# against Groq's on-demand OTPM rate limit.
JD_MODEL_NAME = "qwen/qwen3.8-27b"
JD_MODEL_MAX_TOKENS = 800

# Matching model — moved to Gemini after a side-by-side comparison showed
# Groq's gpt-oss-120b producing persistent non-determinism (self-
# contradictory verdicts, wording-driven verdict flips) across repeated runs,
# while Gemini was more stable. No max_tokens cap needed: that was
# specifically a workaround for Groq's on-demand-tier output-tokens-per-
# minute limit, which doesn't apply here — our usage is nowhere near
# gemini-3.5-flash-lite's 250K TPM cap. Its free-tier RPM/RPD (15/500) is
# also far more forgiving than Groq's was for this node.
MATCH_MODEL_NAME = "gemini-3.5-flash-lite"

DEFAULT_THRESHOLD = 70.0
DEFAULT_REVIEW_MARGIN = 5.0  # score within threshold +/- this -> "Needs Review" instead of a hard cutoff
