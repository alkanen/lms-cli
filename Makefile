# Makefile for LMS CLI

.PHONY: run lint test typecheck

# Use the Python binary from the virtual environment
PYTHON := ./env/bin/python

run:
	$(PYTHON) main.py rich

lint:
	$(PYTHON) -m black lms_cli tests
	$(PYTHON) -m flake8 lms_cli tests

test:
	$(PYTHON) -m pytest tests

typecheck:
	$(PYTHON) -m mypy lms_cli
