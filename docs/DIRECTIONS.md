# Directions

This guide collects the practical steps for setting up SCFD Research, reproducing benchmarks, and exporting controllers.

## 1. System Requirements
- Python 3.10+ (tested on CPython 3.11)
- NumPy/SciPy stack with FFT support
- Optional: CUDA-capable GPU for accelerated linear algebra (SCFD runs on CPU by default)
- Git (for cloning/pulling updates)

## 2. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .[dev]
```

The editable install exposes the `run/` entry points as Python modules.

## 3. Smoke Tests
Run the focused test suites to validate the installation:
```bash
pytest tests/test_heat_arc.py tests/test_train_cma_heat_arc.py
pytest tests/test_heat_routing.py tests/test_train_cma_heat_routing.py
pytest tests/test_heat_front.py tests/test_heat_param_id.py
```

## 4. CMA Search Workflows
Each benchmark has a CMA-ES driver (see `run/train_cma_*.py`). Example (heat routing):
```bash
python -m run.train_cma_heat_routing   --generations 30 --population 10 --elite 3   --steps 1200 --record-interval 120 --outdir runs/heat_routing_cma
```
Outputs:
- `runs/<tag>/best_vector.json` ? controller vector + rich metadata.
- `runs/<tag>/history.csv` ? optimisation trace.
- `runs/<tag>/best_artifact/` ? metrics array and visualisations.

## 5. Replaying Controllers
Use the paired `run/run_*.py` scripts:
```bash
python run/run_heat_routing.py   --vector runs/heat_routing_cma/best_vector.json   --outdir outputs/heat_routing_demo
```
Override metadata with CLI flags (e.g., `--initial-centers`, `--front-radius`).

## 6. Robustness Battery
Evaluate perturbations across domains:
```bash
python -m run.robustness_battery --domains heat flow wave --steps 1600 --out robustness_report.json
```
Summaries include baseline errors and scenario ratios for heat, flow, and wave benchmarks.

## 7. Latency Profiling
Benchmark inference and integration latency:
```bash
python run/latency_profile.py --vector runs/heat_arc_cma/best_vector.json --steps 200
```
The profiler reports per-step timings and budget utilisation tags.

## 8. Orchestrator Usage
`orchestrator/pipeline.py` provides utilities to probe simulations and pick suitable vectors:
```python
from orchestrator.pipeline import plan_for_environment
from benchmarks.heat_diffusion_arc import HeatDiffusionArcSimulator, HeatDiffusionArcParams
from benchmarks.heat_diffusion import HeatDiffusionController, HeatDiffusionControlConfig

controller = HeatDiffusionController(HeatDiffusionControlConfig(), (96, 96))
sim = lambda: HeatDiffusionArcSimulator(HeatDiffusionArcParams(), controller)
plan = plan_for_environment(sim)
print(plan.summary())
```

## 9. Data & Media
- Cart-pole demonstration assets: `cartpole_demo/scfd/`
- Robustness reports: `runs/robustness_*`
- Evolution traces: `runs/*/history.csv`

## 10. Contribution Notes
- Default license is AGPL-3.0. Submit AGPL-compatible patches or contact us for contributor agreements.
- Include tests for new benchmarks (`tests/test_*`).
- Use metadata helpers when persisting vectors.

Questions? Open an issue or email licensing@looptronics.ai.

