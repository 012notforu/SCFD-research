# SCFD Research Benchmarks

Curated snapshot of the SCFD (Spatial Control via Flow Dynamics) benchmarks built during our work session.

## Quickstart

```powershell
python -m venv scfd_vnext/.venv
scfd_vnext/.venv/Scripts/pip install -U pip
scfd_vnext/.venv/Scripts/pip install -e scfd_vnext
```

## Run Tests

```powershell
Push-Location scfd_vnext
./.venv/Scripts/pytest.exe tests -q
Pop-Location
```

## Smoke All Tasks

```powershell
./scfd_vnext/.venv/Scripts/python.exe scfd_vnext/scripts/meta_smoke.py --tasks-dir meta/tasks --output comparisons/meta_smoke_latest.json
```

## Optimise Controllers (examples)

```powershell
./scfd_vnext/.venv/Scripts/python.exe -m run.train_cma_heat_periodic --outdir scfd_vnext/runs/heat_periodic_cma_stripe --target stripe
./scfd_vnext/.venv/Scripts/python.exe -m run.train_cma_heat_anisotropic --outdir scfd_vnext/runs/heat_anisotropic_cma --target tilted --angle-deg 22.5
./scfd_vnext/.venv/Scripts/python.exe -m run.train_cma_flow_constriction --outdir scfd_vnext/runs/flow_constriction_cma
```

## Notes
- `scfd_vnext/` contains controllers, benchmarks, CMA trainers, and tests.
- `meta/` stores task descriptors and tuned vectors.
- `SCFD_STATUS.md` & `TODO_SPATIAL_BENCHMARKS.md` track roadmap progress.
