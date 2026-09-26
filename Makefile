# ExplainForge Makefile — 一键复现
PY ?= python

.PHONY: install test lint demo ci bench ci-local clean

install:
	$(PY) -m pip install -r requirements.txt

lint:
	$(PY) -m compileall -q explainforge tests

test:
	$(PY) -m pytest -q -W ignore::UserWarning

demo bench:
	$(PY) explainforge/examples/run_demo.py

ci-local: lint test demo

ci: ci-local

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -f benchmark.json
