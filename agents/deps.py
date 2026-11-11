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
