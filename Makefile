# Common commands (Mac/Linux, or Windows with make installed).
# On Windows without make, run the commands shown in the README.

PY ?= python

.PHONY: help install up down health migrate seed train report api ui test eval eval-quick fixtures setup

help:
	@echo "setup      install deps, start Postgres, migrate, seed, train (one-time)"
	@echo "api / ui   start the FastAPI backend / the Streamlit UI"
	@echo "test       run pytest"
	@echo "eval       run the evaluation (includes a slow LLM sample)"
	@echo "eval-quick run the evaluation without the LLM sample"

install:
	$(PY) -m pip install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down

health:
	$(PY) scripts/health_check.py

migrate:
	alembic upgrade head

seed:
	$(PY) scripts/seed_db.py

train:
	$(PY) scripts/train.py --model all

report:
	$(PY) scripts/risk_report.py

api:
	uvicorn app.main:create_app --factory --port 8000

ui:
	streamlit run ui/app.py

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m evals.run

eval-quick:
	$(PY) -m evals.run --no-llm

fixtures:
	$(PY) scripts/make_fixtures.py

setup: install up migrate seed train
