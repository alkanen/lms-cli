# Makefile for ai-cli

.PHONY: run lint test typecheck

# Use the Python binary from the virtual environment
PYTHON := ./env/bin/python

run:
	$(PYTHON) -m ai_cli

lint:
	$(PYTHON) -m black ai_cli tests
	$(PYTHON) -m flake8 ai_cli tests

test:
	$(PYTHON) -m pytest tests

typecheck:
	$(PYTHON) -m mypy ai_cli
