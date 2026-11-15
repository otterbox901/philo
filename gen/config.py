"""Every setting for the generation pipeline, in one place."""

from pathlib import Path

# --- Model -------------------------------------------------------------------
MODEL = "claude-haiku-5-5"
EFFORT = "medium"  # how hard the model thinks: "low", "medium", "high", ... ("medium" is Haiku's default)
MAX_TOKENS = 16000  # room for thinking plus the JSON answer for 8 quotes
CACHE_TTL = "1h"  # how long the cached system prompt lives; batches can run longer than 5 minutes

# --- Rounds ------------------------------------------------------------------
# Each generation round samples quotes that no earlier round used and keeps its own sample,
# requests, batches and responses. Round 1 (the original run) lives directly in data/gen/;
# later rounds live in data/gen/round<N>/. build_requests, pilot, submit and collect work on
# ROUND; filter combines every round from 1 to ROUND.
ROUND = 2
N_QUOTES_BY_ROUND = {1: 4000, 2: 4400}  # round 2: ~$2.45 at round 1's measured token use

# --- Sampling ----------------------------------------------------------------
SEED = 42
N_QUOTES = N_QUOTES_BY_ROUND[ROUND]
MIN_WORDS = 5  # drop quotes shorter than this
MAX_WORDS = 60  # drop quotes longer than this
QUOTES_PER_REQUEST = 8
PILOT_QUOTES = 50  # the pilot uses the first 50 sampled quotes
NONSENSE_THEMES_PER_REQUEST = 4  # random themes per nonsense request, for variety (see prompts.py)

# --- Filtering ---------------------------------------------------------------
MAX_INPUT_CHARS = 1000
MAX_OVERLAP = 0.40  # drop an input that shares more than 40% of the quote's content words
VAL_FRACTION = 0.10

# --- Collecting --------------------------------------------------------------
POLL_SECONDS = 60  # how often collect.py checks whether the batch has finished

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # the project folder
QUOTES_PATH = ROOT / "data" / "processed" / "quotes.parquet"
EVAL_DIR = ROOT / "data" / "eval"

GEN_DIR = ROOT / "data" / "gen"


def round_dir(n):
    """Where generation round n keeps its files."""
    return GEN_DIR if n == 1 else GEN_DIR / f"round{n}"


ROUND_DIR = round_dir(ROUND)
SAMPLE_IDS_PATH = ROUND_DIR / "sample_ids.txt"
REQUESTS_PATH = ROUND_DIR / "requests.jsonl"
BATCHES_PATH = ROUND_DIR / "batches.json"  # which batches were submitted, and what happened to them
RAW_DIR = ROUND_DIR / "raw"  # one file per successful request
FAILED_RAW_DIR = RAW_DIR / "failed"  # raw responses of failed attempts, for debugging
FAILED_LOG_PATH = ROUND_DIR / "failed.jsonl"  # requests that failed even after the retry
PILOT_RAW_DIR = ROUND_DIR / "pilot" / "raw"

PAIRS_PATH = ROOT / "data" / "processed" / "pairs.parquet"
TRAIN_PATH = ROOT / "data" / "processed" / "pairs_train.parquet"
VAL_PATH = ROOT / "data" / "processed" / "pairs_val.parquet"
