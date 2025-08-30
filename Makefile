# Makefile for Freqtrade - Ubuntu 24.04 / macOS
# Safe, fast-fail, and sensible defaults

SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

# Use a real executable name; override with `make PYTHON=python3.12`
PYTHON ?= python3.12
PIP    := $(PYTHON) -m pip

export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
export PYTHONWARNINGS=ignore::DeprecationWarning
export CUDA_VISIBLE_DEVICES=1

REQ_MAIN := requirements.txt
REQ_DEV  := requirements-dev.txt

.DEFAULT_GOAL := help

.PHONY: help all quick install dev-install lint format test test-backtesting test-hyperopt \
        check-version extract-schema extract-docs docs-build docs-check build clean

help:
	@echo "Targets:"
	@echo "  quick             : install + lint + test (no docs/build/hyperopt)"
	@echo "  all               : clean + install + dev-install + format + lint + test + docs + build"
	@echo "  test-backtesting  : run sample backtest"
	@echo "  test-hyperopt     : run a small hyperopt (slow)"
	@echo "  clean             : remove build, caches, user_data, docs site"
	@echo "Override PYTHON: make PYTHON=python3.12 quick"

# A reasonable default pipeline without very heavy steps
quick: install dev-install format lint test

# Full pipeline (keeps build artifacts; does NOT clean at the end)
all: clean install dev-install format lint test docs-build docs-check build

dev: clean build clean

install:
	$(PIP) install -U pip wheel --ignore-installed urllib3 wheel cryptography jsonschema
	@test -f $(REQ_MAIN) && $(PIP) install -r $(REQ_MAIN) || true
	[ -d "ft_client" ] && $(PIP) install -e ft_client/ || true
	$(PIP) install -e .

dev-install:
	@test -f $(REQ_DEV) && $(PIP) install -r $(REQ_DEV) || \
		$(PIP) install ruff isort mypy mkdocs mkdocs-material

lint:
	isort --check .
	ruff check --output-format=github
	ruff format --check
	mypy freqtrade scripts tests

format:
	isort .
	ruff check --fix
	ruff format

test:
	pytest --random-order --durations 20 -n auto

test-backtesting:
	cp tests/testdata/config.tests.json config.json
	freqtrade create-userdir --userdir user_data
	freqtrade new-strategy -s AwesomeStrategy
	freqtrade new-strategy -s AwesomeStrategyMin --template minimal
	freqtrade backtesting --datadir tests/testdata --strategy-list AwesomeStrategy AwesomeStrategyMin -i 5m

test-hyperopt:
	cp tests/testdata/config.tests.json config.json
	freqtrade create-userdir --userdir user_data
	freqtrade hyperopt --datadir tests/testdata -e 6 --strategy SampleStrategy --hyperopt-loss SharpeHyperOptLossDaily --print-all

check-version:
	$(PYTHON) build_helpers/freqtrade_client_version_align.py

extract-schema:
	$(PYTHON) build_helpers/extract_config_json_schema.py

extract-docs:
	$(PYTHON) build_helpers/create_command_partials.py

docs-build:
	mkdocs build

docs-check:
	./tests/test_docs.sh

build:
	$(PIP) install -U build
	$(PYTHON) -m build --sdist --wheel
	[ -d "ft_client" ] && (cd ft_client && $(PYTHON) -m build --sdist --wheel) || true

clean:
	rm -rf dist/ build/ *.egg-info ft_client/dist/ ft_client/build/ ft_client/*.egg-info
	rm -rf user_data/ config.json site/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
