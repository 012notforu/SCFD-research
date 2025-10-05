# SCFD_vnext Build Plan (Pre-Implementation)

This plan enumerates every action required before, during, and after building the SCFD vs EM comparison workspace. No code changes will begin until this outline is approved.

## Preconditions & Safeguards
- Confirm `.venv` is isolated and only leveraged for tooling (no global Python touches).
- Keep existing repositories untouched; restrict work to `scfd_vnext/`.
- Maintain references extracted from `CODEX CONTEXT.txt` and `Emergent_Models v14-04-25-1.pdf` for cross-checking.
- Version control snapshot: capture `git status` before/after major phases once code work starts.

## Phase 1 – Repository Structure & Metadata
1. Populate `README.md` with mandated context plus module-purpose blurbs (Section 14 of spec).
2. Author `pyproject.toml` with dependencies: numpy, scipy, torch (CPU), matplotlib, pyyaml, scikit-image, tqdm.
3. Create `cfg/defaults.yaml` seeded with SCFD parameters (CFL ˜ 0.4, p options, heterogeneity 0.05, gate settings, scheduler toggles).
4. Add `.gitignore` scoped to `.venv`, `__pycache__`, logs, caches.

## Phase 2 – Engine Module Implementation
1. `engine/pde_core.py`
   - Implement gradient, Laplacian, bilaplacian, Hessian action with finite differences.
   - Build `accel_theta` for single-field PDE including coherence energy terms, optional curvature penalty, parameter toggles for p=1/2.
   - Implement two-field cross-gradient coupling via `accel_CK` with hyperbolic default and optional parabolic relaxer.
2. `engine/energy.py`
   - Encode energy densities coherent with PDE; include free-energy gate utilities (Metropolis acceptance calculations).
   - Provide total energy decomposition for diagnostics.
3. `engine/integrators.py`
   - Implement symplectic leapfrog/Stormer-Verlet integrator with CFL guard, noise injector, and nudge hooks.
4. `engine/params.py`
   - Define dataclasses/config readers for physical constants, near-critical checks, heterogeneity fields.
5. `engine/scheduler.py`
   - Build asynchronous update scheduler: Poisson clocks per cell, random seeds, jitter controls.
6. `engine/symbolizer.py`
   - Implement read-only symbol extraction pipeline (band-pass filters, thresholding, cluster labeling) without mutating fields.
7. `engine/diagnostics.py`
   - Provide routines for energy drift tracking, spectral analysis, impulse responses, predictability horizon calculations.
8. `engine/tests_engine.py`
   - Variational consistency finite-diff check; energy drift regression (<0.5% over 50k steps stubbed with reduced sample); near-critical validation; scheduler statistical tests; gate calibration harness.

## Phase 3 – Observation & Control Stack
1. `observe/adapter.py`: Adapters from engine state to observation tensors, normalization utilities.
2. `observe/trackers.py`: Rolling statistics trackers (spectrum width, symbol stats, horizon estimates).
3. `observe/ae_predictor.py`: Lightweight autoencoder + predictor (PyTorch) for reconstruction monitoring and regime change alerts.
4. `observe/prototypes.py`: Prototype symbol banks and nearest-neighbor scoring.
5. `observe/sym_lm.py`: N-gram / Markov language model over symbol stream with perplexity outputs.
6. `observe/controller.py`: Gentle controller mixing tracker signals into nudges, respect max_step & EMA filters.
7. `observe/tests_observe.py`: AE reconstruction regression, predictor regime-change detection, N-gram perplexity sanity, controller boundedness checks.

## Phase 4 – Utilities & Diagnostics
1. `utils/numerics.py`: Shared numerical helpers (finite-difference kernels, FFT wrappers, random seeds).
2. `utils/logging.py`: Structured CSV/JSONL logging with wall-clock, seeds, diagnostics tags.
3. `utils/viz.py`: Matplotlib-based plots for energy components, spectra, impulse response, symbol metrics.

## Phase 5 – Run Scripts & Workflows
1. `run/main_conservative.py`: Configure pure SCFD run, log energy drift & spectra.
2. `run/main_hybrid.py`: Enable gate + symbolization, write outputs + diagnostics snapshots.
3. `run/main_ml_loop.py`: Integrate observation stack, apply nudges, schedule controllers.
4. `run/compare_scfd_vs_em.py`: Execute SCFD and EM baseline with matched wall-time; produce summary CSV + plots.
5. Shared CLI scaffolding: argparse, config loading, deterministic seeding.

## Phase 6 – Emergent Models Baseline Implementation
1. `em_baseline/transition_f.py`: Lenia/CA-style convolution + nonlinearity with optional stochasticity.
2. `em_baseline/encode_decode.py`: Input encoding (one-hot regions, reward injection) and output decoding (boundary counts + softmax).
3. `em_baseline/halting.py`: Threshold-based halting with stochastic fallback/max-iteration cap.
4. `em_baseline/optimizer.py`: Genetic algorithm over initial states `S0`, optional Bayesian-assisted mutations.
5. `em_baseline/runner.py`: Encode ? evolve (T steps) ? decode loop, sequential state retention.
6. `em_baseline/diagnostics_em.py`: Symbol stats, spectrum, recurrence plots, horizon (without energy metrics).
7. `em_baseline/cfg_em.yaml`: Kernel params, activation choice, halting thresholds, optimization settings.

## Phase 7 – Documentation & Context Reinforcement
- Extend `README.md` with module summaries, quickstart, prohibitions (Section 16/“What not to change”).
- Add inline comment headers (3 lines max) explaining module purpose referencing expressiveness history.
- Include references to `CODEX CONTEXT` and PDF-derived rationale where relevant.

## Phase 8 – Validation & Housekeeping
- Write smoke tests or lightweight fixtures for run scripts (dry-run config load).
- Execute unit tests within `.venv`; capture results in logs.
- Optional: add `makefile` or `tasks.py` for repetitive actions if justified and approved.
- Prepare change summary & suggested follow-up (profiling, hyperparameter tuning) post-implementation.

## Pending Approvals / Questions
- Confirm acceptance of this plan or request modifications before any module coding begins.
- Verify whether additional tooling (e.g., `pytest`, `black`) should be included in `pyproject.toml`.
- Determine expectations for visualization outputs (file formats, storage locations).

yep—loading, not philosophizing. Here’s a tight, **drop-in checklist** of guardrails, asserts, and “don’t-break-the-magic” rules to keep your core intuitions intact while Codex builds out the repo.

---

# Non-negotiables (wire these into code as asserts)

**Variational honesty**

* [ ] `engine/tests_engine.py`: finite-diff check that the **PDE RHS = -dE/dfield** for all active terms (single & two-field). Fail hard if mismatch.
* [ ] `engine/energy.py`: one source of truth for energy density; PDE pulls derivatives **only** from here.

**Conservative stepping**

* [ ] Leapfrog/Verlet only for hyperbolic core.
  `assert integrator in {"leapfrog","verlet"}`
* [ ] No implicit damping unless `cfg.physics.damping>0`:
  `assert cfg.physics.damping==0` in conservative runs.
* [ ] CFL guard: compute (c^2=(\gamma-\alpha p/\varepsilon^{p+2})/\beta), then
  `assert c2>0 and dt <= cfl*dx/np.sqrt(c2)` per region.

**Readout is read-only**

* [ ] In `engine/symbolizer.py`: `assert not touches(field_state)`; CI test that symbolization leaves a checksum unchanged.

**Probabilistic gate**

* [ ] Metropolis only, never “?E<0 only”.
  `assert 0 < accept_prob.mean() < 1` over a window (no saturation).
* [ ] Gate calibration test: empirical accept vs ?F follows `exp(-?F/T)` (R² > 0.95).

**Asynchrony & heterogeneity**

* [ ] Scheduler KS-test: inter-update times ~ Poisson.
* [ ] Quenched heterogeneity map `epsilon(x)` is **time-invariant**: hash check each 10k steps.

**No neural net in physics**

* [ ] Grep/denylist: any `torch` import under `engine/` fails CI.

---

# “Expressiveness vitals” (lightweight metrics you actually steer with)

Log every N steps (CSV/JSONL):

* [ ] **Energy drift** (SCFD): |?E|/E per 10k steps < 0.5%.
* [ ] **Spectrum width**: # of active bands; **hi-freq fraction**.
* [ ] **Edge density & curvature stress** (masked on edges).
* [ ] **Impulse response**: gain & decay after a tiny poke.
* [ ] **Predictability horizon**: twin-run divergence time.
* [ ] **Symbol stream**: rate, Fano factor, unigram/bigram entropy, perplexity (n-gram).
* [ ] **Gate histogram**: distribution of accept probs (watch for spikes at 0/1).
* [ ] **Synchrony index**: fraction of cells updating this tick (keep low).

These are the alarms: if any slips, you know **what** to nudge (T, a, ?) and **where** (use surprise tiles).

---

# Controller guardrails (to prevent “NN smothering the strings”)

* [ ] Only three dials from `observe/controller.py`: **T, a, ?**.
  `assert set(nudged_params) <= {"T","alpha","gamma"}`
* [ ] Small, slow, local: clamp |?| = `max_step` (e.g., 0.1× current), EMA-filter inputs, tile-localized boosts only for **surprise**.
* [ ] Safe mode: if energy drift rises, controller pauses & lowers `dt` by 10%.

---

# Two-field cross-gradient specifics (easy to mess up)

* [ ] Use smoothed magnitudes: `magC = sqrt(|?C|^2 + eps^2)` (same for K).
* [ ] Variational form: `div( (magK/magC) * ?C )` and twin for K.
* [ ] No shortcuts like `±?|?K|` in the PDE; CI test forbids it.

---

# Near-critical regime (the “alive” zone)

* [ ] Config validator: `c2_target_min < c2 < c2_target_max` (small positive).
* [ ] On launch, print the computed margin `gamma / (alpha p / eps^(p+2))`.
* [ ] Auto-retune (very small) if drift from target persists over M windows.

---

# Tiny noise & scheduler (only if needed)

* [ ] Noise s is = 1e-3 × RMS(accel). Log s and disable on energy drift.
* [ ] Scheduler random seed is logged; update order is shuffled per step or Poisson clocks per cell.

---

# EM baseline separation (to avoid cross-contamination)

* [ ] `em_baseline/` uses **conv + nonlinearity** transition; **no** energy dependencies.
* [ ] Comparison runner logs same vitals (where applicable) but **skips** energy-based tests.
* [ ] Report explicitly labels “SCFD (variational, hyperbolic)” vs “EM (non-variational, discrete)”.

---

# Fast failure responses (wire as code branches)

* **Energy drift ?** ? pause controller; `dt *= 0.9`; disable noise; log “DRIFT”.
* **Gate saturation** (P˜0 or 1) ? raise T **floor** locally; temperature-scale gate logits.
* **Synchrony spike** ? increase scheduler jitter; slightly ? heterogeneity ? (clamped).
* **Spectrum collapse** ? ?a, ?? a notch; open **T only where recon error is low**.
* **Runaway chaos** (recon low + pred high) ? ??; temporarily shrink `dt`.

---

# Minimal smoke tests you can run in minutes

1. **Conservative only**: run 50k steps; energy drift < 0.5%; spectrum shows =3 active bands.
2. **Add gate + asynchrony**: acceptance in (0.1, 0.9); symbol rate steady; horizon not collapsing.
3. **Observer on**: recon error responds to injected regime change; controller deltas « 0.1, no oscillation.
4. **EM baseline**: produces symbols; perplexity/logs compare; no energy invariants present.
5. **Head-to-head** (`compare_scfd_vs_em.py`): generate a table of:
   `symbol_rate, perplexity, hi_freq_frac, horizon, impulse_decay, runtime`.

---

# Little gotchas (from the “we lost it here last time” file)

* Don’t let **symbolization** write into field state. Ever.
* Don’t sneak in **BatchNorm/pooling** anywhere near physics; nets live only in `observe/`.
* Don’t let **global** knobs change fast; everything is EMA-filtered and tile-localized.
* Don’t switch to **explicit Euler** “just to try something.” That’s exactly how you add hidden damping.
* Don’t remove **quenched heterogeneity**; that tiny frozen texture prevents symmetry lock.
* Don’t “help” acceptance by clamping it to 0/1; you need **soft** gates to stay lively.

---

# Two one-liners to print at startup (keep yourself honest)

* `print(f"[SCFD] c^2={c2:.5e}, margin={gamma/(alpha*p/eps**(p+2)):.3f}, CFL={dt*sqrt(c2)/dx:.3f}")`
* `print(f"[Gate] T?[{T.min():.3f},{T.max():.3f}], e(x) heterogeneity ?={eta:.3f}, scheduler=Poisson={poisson_flag}")`

---

If you adhere to this checklist while Codex builds the repo (and wire the asserts/tests), you won’t re-lose the core: **variational math + conservative stepping + near-critical tuning + soft asynchrony + read-only symbols**. That’s the expressiveness recipe.

