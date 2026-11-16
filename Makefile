PY ?= python
DATA_DIR ?= data/oulad

.PHONY: install install-research data research research-quick test lint web-install web-build dev serve docker

install:            ## API + test dependencies
	$(PY) -m pip install -r requirements-dev.txt

install-research:   ## + LightGBM/XGBoost/CatBoost/Optuna for the research pipeline
	$(PY) -m pip install -r requirements-research.txt

data:               ## download + verify OULAD from the UCI repository
	$(PY) -m riskml.oulad.download --dest $(DATA_DIR)

research: data      ## full research run (~1 h on 4 cores); rewrites artifacts/ and web/src/generated/
	$(PY) -m riskml.research.pipeline --data-dir $(DATA_DIR)

research-quick: data
	$(PY) -m riskml.research.pipeline --data-dir $(DATA_DIR) --quick --out-root build/quick

test:
	$(PY) -m pytest --cov=riskml --cov-report=term-missing

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

web-install:
	cd web && npm ci

web-build:
	cd web && npm run build

dev:                ## API on :8000 + Next.js dev server on :3000
	( $(PY) -m uvicorn riskml.server.main:app --reload --port 8000 & ) ; cd web && npm run dev

serve: web-build    ## production-like: one process serves UI + API on :8000
	SERVE_STATIC=1 $(PY) -m uvicorn riskml.server.main:app --port 8000

docker:
	docker build -t agrim . && docker run --rm -p 7860:7860 agrim
