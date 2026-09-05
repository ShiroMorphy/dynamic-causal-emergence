.PHONY: setup test lint format data experiments figures clean help

PYTHON ?= python3
PIP ?= pip

help:
	@echo "Dynamic Causal Emergence (DCE) - Build System"
	@echo "============================================="
	@echo "make setup        : Install package in editable mode with dev dependencies"
	@echo "make test         : Run pytest with coverage report"
	@echo "make lint         : Run ruff and mypy static analysis"
	@echo "make format       : Format codebase with ruff"
	@echo "make data         : Download and curate EIA-930 power grid dataset"
	@echo "make experiments  : Run synthetic and empirical experimental pipelines"
	@echo "make figures      : Generate all publication-ready vector figures for paper"
	@echo "make clean        : Remove build and cache artifacts"

setup:
	$(PIP) install -e ".[dev,docs]"
	pre-commit install

test:
	pytest tests/

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

data:
	$(PYTHON) -m dce.datasets.eia930.client --download-all
	$(PYTHON) -m dce.datasets.eia930.balance --process

experiments:
	$(PYTHON) -m dce.experiments.run_all --config configs/experiment_eia930.yaml

figures:
	$(PYTHON) -m dce.visualization.generate_paper_figures

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/ .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
