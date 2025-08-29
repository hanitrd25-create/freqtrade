# Makefile for Freqtrade - Ubuntu 24.04
PYTHON = python3.12.3
PIP = pip
export PIP_BREAK_SYSTEM_PACKAGES=1
export CUDA_VISIBLE_DEVICES=1
export PYTHONWARNINGS=ignore::DeprecationWarning
export PIP_ROOT_USER_ACTION=ignore

all: clean install lint test test-backtesting test-hyperopt check-version extract-schema extract-docs lint format docs-build docs-check build clean

install:
	$(PIP) install -r requirements.txt --ignore-installed urllib3 wheel
	[ -d "ft_client" ] && $(PIP) install -e ft_client/ || true
	$(PIP) install -e .

test:
    pytest --random-order --durations 20 -n auto
	# pytest -n auto --random-order --pyargs .

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
	python build_helpers/freqtrade_client_version_align.py

extract-schema:
	python build_helpers/extract_config_json_schema.py

extract-docs:
	python build_helpers/create_command_partials.py

lint:
	isort --check .
	ruff check --output-format=github
	ruff format --check
	mypy freqtrade scripts tests

format:
	isort .
	ruff check --fix
	ruff format

docs-build:
	mkdocs build

docs-check:
	./tests/test_docs.sh

build:
	$(PIP) install -U build
	$(PYTHON) -m build --sdist --wheel
	[ -d "ft_client" ] && $(PYTHON) -m build --sdist --wheel ft_client || true

clean:
	rm -rf dist/ build/ *.egg-info ft_client/dist/ ft_client/build/ ft_client/*.egg-info
	rm -rf user_data/ config.json site/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
