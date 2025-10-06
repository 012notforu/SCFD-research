# Energy Drift Audit Notes

This document enumerates every code path and configuration item that can influence total energy when running the “conservative” SCFD loop. Use it as the review guide for diagnosing the runaway energy drift we are observing.

## 1. Configuration Surfaces (`cfg/defaults.yaml`)
- **Run tag / steps** (`run.tag`, `run.steps`): defines output directories and default duration. Drift scales with steps, so all tests reuse this value.
- **Grid spec** (`grid.shape`, `grid.spacing`): resolution sets the finite-difference stencil scale (`dx`). Changing shape affects CFL, Laplacian strength, and FFT diagnostics.
- **Physics block** controls every term in the PDE and derived energy:
  - `beta`, `gamma`, `alpha`, `coherence_p`, `epsilon`: feed directly into `PhysicsParams` and therefore `accel_theta` plus coherence energy density.
  - `potential`: currently set to `kind: quadratic` with `stiffness: 0.05`; derivative and energy functions live in `engine/params.PotentialConfig`.
  - `cross_gradient.enabled`, `curvature_penalty.enabled`: switches that add extra operators in `accel_theta`. They’re off in the latest run but still worth auditing.
  - `heterogeneity`: generates the quenched random field via `HeterogeneityConfig.generate_field`; this acts as a static offset added to the state at init time.
- **Integration block**:
  - `dt`, `cfl_limit`: `dt` is the primary knob we reduced to 5e-4. CFL is only advisory (not enforced anywhere else).
  - `nudges.max_step`: clamps the field delta inside `leapfrog_step`; currently `0.1`.
  - `noise`: entire noise term is disabled (set to false) but the code path in `leapfrog_step` still exists.
- **Free energy gate**: presently enabled although the conservative runner never calls `apply_gentle_gate`; still shows up in `startup_summary`.
- **Rest of the file** (symbolizer, observation, logging, EM baseline) is unused by the conservative script but included for completeness.

## 2. Simulation Setup (`run/common.py`)
- `load_simulation_config` wraps `engine.params.load_config`, so any YAML edits flow directly into `PhysicsParams`, `IntegrationParams`, etc.
- `initialize_state` seeds the run:
  - `theta` initialised from `rng.normal(scale=0.05, size=grid.shape)` - large amplitude noise kicks off big potential energy.
  - `theta_dot` zeros.
  - `heterogeneity` = random frozen field returned by `HeterogeneityConfig.generate_field` and stored alongside the state (although not used inside `run/main_conservative.py`).
- `setup_logger` uses `utils.logging.create_run_directory` so energy CSVs can be audited post-run.
- `summarize_energy` delegates to `engine.compute_energy_report`, so any mismatch there shows up in the CSV.
- `store_spectrum` writes FFT magnitudes for diagnostics; no effect on dynamics.

## 3. Conservative Runner (`run/main_conservative.py`)
- Rebuilds `noise_cfg` straight from integration settings; we already confirmed noise is off in the new config.
- Defines `accel(field)` closure that just calls `engine.accel_theta` with the shared `PhysicsParams` and constant `dx` (from `grid.spacing`). Any bug in `accel_theta` directly drives drift.
- Leapfrog update loop (`leapfrog_step`) is the only evolution path; there is no scheduling mask or gate applied here.
- After each step, `summarize_energy` logs per-component energies but does not feed back into the dynamics.

## 4. Physics Implementation

### 4.1 `engine/pde_core.py`
- `grad`, `laplace`, `bilaplacian`, and `_div` implement centered finite differences with periodic boundaries via `np.roll`. Error or scaling issues here affect both the PDE and the computed energies (because energy also uses `grad`).
- `hessian_action`: computes `grad(delta)^T H(delta) grad(delta)` using finite differences. The algebra is sensitive; mis-scaling could inject energy. It assumes uniform `dx` and uses a four-point stencil for mixed partial derivatives.
- `accel_theta` constructs the full PDE:
  - Computes `q = |grad(theta)|^2 + epsilon^2` then `f = a * p * q^{-m}` where `m = (p + 2) / 2`. This matches the stated coherence energy derivative. The `coherence_extras` term is `2 * a * p * m * q^{-(m + 1)} * hessian_action`.
  - Core term: `(theta - f) * lap` - note the sign; if `f` overwhelms `theta` we can get negative diffusion or blow-up.
  - Subtracts `potential.derivative(theta)` and optional curvature penalty (`-2 * d * laplacian^2(theta)`).
  - Divides by `ß` at the end.
- `accel_CK` handles cross-gradient coupling but is currently disabled in the config. Still good to note signs: returns `-d_c / ß`, etc.

### 4.2 `engine/params.PotentialConfig`
- `derivative`:
  - `double_well`: `2 * stiffness * (phi - phi0) * (phi - phi1) * (2 * phi - phi0 - phi1)`.
  - `quadratic`: `2 * stiffness * (phi - center)`.
- `energy` uses matching formulas: quartic for double-well, quadratic for harmonic.
- Any mismatch between derivative and energy terms directly breaks conservation.

### 4.3 `engine/energy.py`
- `kinetic_energy_density` = `0.5 * rho * |theta_dot|^2` - uses the same mass density `rho` as the PDE.
- `coherence_energy_density`: `a * q^{1 - m}` with `q = |grad(theta)|^2 + epsilon^2` and `m = coherence_exponent`. Needs to align with the derivative used in `accel_theta`. If the exponent should be `m - 1` or similar, any mismatch shows up as drift.
- `potential_energy_density` defers to `PotentialConfig.energy`.
- `total_energy_density` sums kinetic + coherence + potential (+ cross term). `compute_total_energy` averages over the grid.

### 4.4 `engine/diagnostics.compute_energy_report`
- Wraps the same densities but averages each component separately before logging. Confirms the CSV uses exactly the energy formulas above.

## 5. Time Integrator (`engine/integrators.py`)
- `leapfrog_step` implements velocity-Verlet with symmetric half-steps.
  - `acc0 = accel_fn(state)` - uses the current theta field.
  - `v_half = v + 0.5 dt acc0`.
  - Optional noise injection (off in current tests).
  - `delta = dt * v_half` then optional clipping by `max_step` (currently 0.1). If `dt` is tiny and `max_step` large, this clamp isn’t active.
  - `new_state = state + delta`.
  - Recompute `acc1` at new state, then `new_velocity = v_half + 0.5 dt acc1`.
- There’s no symplectic correction or boundary handling beyond the periodic assumption; if `accel_theta` is wrong the integrator can’t save it.

## 6. State & Heterogeneity (`engine/params.HeterogeneityConfig`)
- `generate_field`: builds Gaussian-smoothed random field via FFT convolution, scaled by `amplitude` (0.05). Added to the state only at initialization (not reused later in the conservative run) but influences initial gradients and potential energy.

## 7. Logging & Diagnostics
- `utils/logging.RunLogger`: writes `energy.csv` entries invoked from `run/main_conservative.py`. No back-reaction.
- `scripts/check_energy.py`: quick helper we wrote to compute relative drift from the CSV. Useful sanity check but not part of the simulation itself.

## 8. Tests Touching Energy (`engine/tests_engine.py`)
- `test_total_energy_positive`: ensures mean energy is non-negative for zero fields. Doesn’t prevent drift.
- `test_energy_drift_small_series` (marked slow) simply checks the helper’s arithmetic. There is **no** regression test that runs the PDE and asserts low drift yet.

## 9. Known Factors to Revisit
- **Coherence term alignment**: `coherence_energy_density` uses exponent `1 - m`; verify against the derivation used in `accel_theta` where the derivative involves `q^{-m}` and `q^{-(m+1)}`.
- **Potential derivative vs energy**: for the quadratic case, the derivative is `2 * k * (phi - c)` but energy is `k * (phi - c)^2`. Consistent, but double-check scaling versus the density used in the kinetic term.
- **Initial amplitude**: `rng.normal(scale=0.05)` may still be too large given the very soft quadratic potential (`0.05`). Consider smaller initialization after verifying PDE math.
- **Gate/Controller**: disabled in the conservative loop, so they aren’t producing drift—but when re-enabled they’ll add new feedback paths to audit.

## 10. Files Not Yet Touched but Potentially Relevant
- `observe/` modules: idle during conservative runs.
- `utils/viz.py`: pure reporting.
- `run/common.apply_gentle_gate`: unused here, but if activated it would modify energy indirectly by scaling controller nudges.

Review the sections above to pinpoint mismatches between the PDE force calculation and the energy definition—those are the prime suspects for the runaway energy increase.
