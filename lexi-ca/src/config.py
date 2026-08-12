import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PLAYBOOKS_DIR = DATA_DIR / "playbooks"
CONTRACTS_DIR = DATA_DIR / "contracts"
EVALS_DIR = ROOT_DIR / "evals"
GROUND_TRUTH_DIR = EVALS_DIR / "ground_truth"
MODEL_SEGMENTATION = "claude-haiku-4-5-20251001"  # cheap, fast — extraction only
MODEL_REASONING = "claude-sonnet-4-6"  
CACHE_DIR = ROOT_DIR / "data" / "cache" / "segmentation"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-6"

if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
    )