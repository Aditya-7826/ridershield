"""Minimal Ollama client with Pydantic-validated JSON output.

* Uses Ollama's structured outputs: the Pydantic model's JSON schema is sent
  as `format`, so the model is constrained to produce matching JSON.
* The reply is still validated with Pydantic. On invalid JSON or a schema
  error we retry ONCE, telling the model what was wrong. A second failure
  raises LLMError and the calling agent falls back to deterministic code.
* `keep_alive` is short so models unload when idle, and `unload()` lets us
  free the text model before loading the vision model (low-RAM laptops).
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from agents.config import KEEP_ALIVE, LLM_TIMEOUT_SECONDS, OLLAMA_HOST, OLLAMA_NUM_GPU

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """The model could not produce valid output after one retry."""


class JsonLLM(Protocol):
    """Anything that can return a validated Pydantic object (real or fake)."""

    def chat_json(self, model: str, system: str, user: str, schema: type[T], images: list[Path] | None = None) -> T: ...


class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST, keep_alive: str = KEEP_ALIVE, timeout: float = LLM_TIMEOUT_SECONDS):
        self.host = host
        self.keep_alive = keep_alive
        self.http = httpx.Client(timeout=timeout)
        self.calls = 0      # simple counters, handy for evaluation reports
        self.retries = 0

    def _chat(self, model: str, messages: list[dict], schema: dict) -> str:
        options = {"temperature": 0, "num_predict": 400, "num_ctx": 4096}
        if OLLAMA_NUM_GPU:
            options["num_gpu"] = int(OLLAMA_NUM_GPU)
        resp = self.http.post(f"{self.host}/api/chat", json={
            "model": model,
            "messages": messages,
            "format": schema,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        })
        resp.raise_for_status()
        self.calls += 1
        return resp.json()["message"]["content"]

    def chat_json(self, model: str, system: str, user: str, schema: type[T], images: list[Path] | None = None) -> T:
        user_msg: dict = {"role": "user", "content": user}
        if images:
            user_msg["images"] = [base64.b64encode(Path(p).read_bytes()).decode() for p in images]
        messages = [{"role": "system", "content": system}, user_msg]
        json_schema = schema.model_json_schema()
        last_error = ""
        for attempt in range(2):
            try:
                raw = self._chat(model, messages, json_schema)
            except httpx.HTTPError as exc:
                raise LLMError(f"Ollama request failed: {exc}") from exc
            try:
                return schema.model_validate_json(raw)
            except ValidationError as exc:
                last_error = str(exc).splitlines()[0]
                log.warning("invalid JSON from %s (attempt %d): %s", model, attempt + 1, last_error)
                if attempt == 0:
                    self.retries += 1
                    messages += [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": f"That JSON was invalid ({last_error}). "
                                                    f"Reply again with ONLY valid JSON matching the schema."},
                    ]
        raise LLMError(f"{model} returned invalid JSON twice: {last_error}")

    def unload(self, model: str) -> None:
        """Ask Ollama to free a model's memory now (ignore errors)."""
        try:
            self.http.post(f"{self.host}/api/generate", json={"model": model, "keep_alive": 0})
        except httpx.HTTPError:
            pass


class OfflineLLM:
    """Always fails, so every agent uses its deterministic fallback (no Ollama needed)."""

    def chat_json(self, *args, **kwargs):
        raise LLMError("LLM disabled")

    def unload(self, model: str) -> None:
        pass


def to_json(obj) -> str:
    """Compact JSON for prompts."""
    return json.dumps(obj, default=str, separators=(",", ":"))
