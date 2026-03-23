# Makefile for ai-cli

.PHONY: run format check test build dist clean

PYTHON := ./env/bin/python
RUFF   := ./env/bin/ruff
MYPY   := ./env/bin/mypy

# Run the application
run:
	$(PYTHON) -m ai_cli

# Apply formatting and auto-fix safe lint issues
format:
	echo "Perform formatting"
	$(RUFF) format ai_cli tests
	$(RUFF) check --fix ai_cli tests

# Check only — no changes written (suitable for CI / pre-commit)
check:
	echo "Run formatting and type checks"
	$(RUFF) format --check ai_cli tests
	$(RUFF) check ai_cli tests
	$(MYPY) ai_cli

# Run the test suite
test:
	echo "Run pytest"
	$(PYTHON) -m pytest tests

# Build a wheel (and sdist) into dist/
build:
	$(PYTHON) -m build

# Alias
dist: build

# Remove build artifacts
clean:
	rm -rf dist/ build/ ai_cli.egg-info/

.SILENT:
