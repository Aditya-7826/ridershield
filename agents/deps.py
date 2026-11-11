"""Dependencies shared by the agent nodes (database, LLM client, ML models)."""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine

from agents.llm import JsonLLM
from ml.claim_scoring import Models


@dataclass
class Deps:
    engine: Engine
    llm: JsonLLM
    models: Models
    use_vision: bool = True
    # Evaluation mode: run every agent but write nothing to the database.
    dry_run: bool = False
    # Use the LLM for explanations (the decision itself never depends on it).
    llm_explanations: bool = True
