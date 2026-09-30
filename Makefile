.PHONY: install test lint demo bench

install:
	pip install -e ".[sota,dev]"

test:
	python -m pytest -q -W ignore::UserWarning

lint:
	ruff check . && ruff format --check .

demo:
	python examples/run_demo.py

bench:
	python -m causaldisc.cli benchmark --out benchmark.json
