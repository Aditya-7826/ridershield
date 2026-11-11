# RiderShield

AI micro-insurance for gig delivery riders: ETA regression, anomaly detection, local LLM agents and human-in-the-loop claim review.

> Work in progress. Everything runs locally and for free (PostgreSQL in Docker, models through Ollama).

## Quick start (so far)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
ollama pull qwen2.5:3b
ollama pull qwen2.5vl:3b
python scripts/health_check.py
```

A full README with architecture, results and run instructions will follow.
