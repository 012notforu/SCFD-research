# SCFD vs EM: Build Spec for Codex

## Context and History

We maintain two parallel symbol-generating systems that share a diagnostics harness.

1. **SCFD/CFT engine**: hyperbolic, variational, inverse-gradient coherence energy that repels flatness, symplectic leapfrog stepping, near-critical tuning, probabilistic free-energy gate, asynchronous updates, quenched heterogeneity, tiny noise, and strictly read-only symbolization.
2. **EM baseline**: reference implementation of the Emergent Models encode-evolve-decode paradigm (Lenia or CA style convolutional transition, halting predicate, black-box state optimization) with no energy or Noether constraints.

Neural nets stay in the observation loop only; past attempts to couple them into the physics introduced hidden friction, spectral bias toward smoothing, and re-synchronised updates that destroyed expressiveness.

## Module Guide

- `cfg/`: declarative configuration (defaults, overrides) documenting near-critical parameters and gate settings.
- `engine/`: variational PDE core, energies, integrators, heterogeneity scheduler, and read-only symbolization.
- `observe/`: measurement stack (trackers, AE predictor, controller) that nudges only `T`, `alpha`, `gamma` with tiny, EMA-filtered deltas.
- `utils/`: shared numerics, structured logging, and plotting so diagnostics stay consistent across SCFD and EM runs.
- `run/`: CLI entry points for conservative SCFD, hybrid (gate plus symbols), ML loop, and SCFD versus EM comparisons.
- `em_baseline/`: standalone Emergent Models reference implementation that optimises the initial state only.
- `references/`: extracted context (`CODEX CONTEXT.txt`, Emergent Models PDF) to prevent regressions.
- `scripts/`: helper utilities (for example PDF extraction) kept separate from production modules.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
pytest                        # fast suite
pytest -m slow -n auto         # full regression
python run/main_conservative.py --cfg cfg/defaults.yaml
python run/main_hybrid.py --cfg cfg/defaults.yaml
python run/main_ml_loop.py --cfg cfg/defaults.yaml
python run/compare_scfd_vs_em.py --scfd-cfg cfg/defaults.yaml --em-cfg em_baseline/cfg_em.yaml
```

## Operational Guardrails

- Physics stays heuristic-free: derive PDE terms from the coherence energy; keep leapfrog or Verlet stepping; print `c^2`, margin, and CFL at startup.
- No neural network writes into field state; symbolization is read-only with checksums before-and-after.
- Maintain asynchronous scheduler and quenched heterogeneity (non-zero but tiny) by default.
- Metropolis gate must remain probabilistic (monitor acceptance histogram for spikes at 0 or 1).
- Controller can nudge only `T`, `alpha`, `gamma`, and every change is small, EMA-filtered, and tile-localised.
- Never swap in global synchronised updates, explicit Euler stepping, or remove the inverse-gradient coherence penalty.

## Output Conventions

Run scripts write to `runs/<timestamp>_<tag>/` containing the effective config (`cfg.yaml`), `logs.jsonl`, CSV files for energy, spectrum, symbol, and controller metrics, and a `plots/` directory with standard figures. Override via `--outdir` when necessary.

## References

- `CODEX CONTEXT.txt` (instructions for preserving expressiveness).
- `Emergent_Models v14-04-25-1.pdf` (encode-evolve-decode baseline design).
