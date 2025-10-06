PY=python
VENV=. .venv/bin/activate;

.PHONY: dev test lint type demo fmt ci
dev:
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

fmt:
	ruff format .

lint:
	ruff check .

type:
	mypy

test:
	pytest -q

demo:
	$(PY) -m benchmarks.run_cartpole --controller scfd --vector runs/cartpole_cma/best_vector.json --steps 5000 --episodes 3 --viz scfd --video-format gif --outdir cartpole_outputs

ci: fmt lint type test