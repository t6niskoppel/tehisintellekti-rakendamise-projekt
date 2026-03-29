import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

API_KEY = os.getenv("OPENAI_API_KEY")
APP_ENV = os.getenv("APP_ENV", "user").lower()
DEFAULT_MODEL = os.getenv("MODEL_SELECTION", "google/gemma-3-27b-it")
MODEL_OPTIONS = ["google/gemma-3-27b-it"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
COURSES_PATH = DATA_DIR / "courses.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.pkl"
TEST_CASES_PATH = DATA_DIR / "test_cases.csv"

USAGE_DIR = BASE_DIR / ".usage"
USAGE_PATH = USAGE_DIR / "token_usage.json"
LAST_TEST_RESULTS_PATH = USAGE_DIR / "last_test_results.csv"
FEEDBACK_LOG_PATH = BASE_DIR / "feedback_log.csv"
