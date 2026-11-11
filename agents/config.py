"""Agent settings, read from .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen2.5:3b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:3b")
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "2m")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "600"))
# Number of model layers to put on the GPU. Empty = let Ollama decide.
# Set 0 to force CPU when a small GPU runs out of memory.
OLLAMA_NUM_GPU = os.getenv("OLLAMA_NUM_GPU", "").strip()
CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", ROOT / "checkpoints" / "graph.sqlite"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", ROOT / "uploads"))
