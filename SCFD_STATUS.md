# SCFD Benchmark Status (As of 2025-09-30)

Below are the tuned controller vectors currently archived in `meta/vectors/` together with the most recent smoke-test metrics (averaged over seeds 0,1,2 unless stated otherwise).

## Cart-Pole Control (SCFD Blend)

- **Baseline vector:** `meta/vectors/cartpole/2025-09-29_half.json`
  - Source run: `runs/scfd_cma_half`
  - Performance: 2,500-step balance (min/max = 2,500/2,500; std = 0)
  - Task descriptor: `meta/tasks/cartpole_scfd_blend.json`
- **High-horizon vector:** `meta/vectors/cartpole/2025-09-29_highhorizon.json`
  - Source run: `runs/scfd_cma_highhorizon`
  - Performance: 4,000-step balance (min/max = 4,000/4,000; std = 0)
  - Task descriptor: `meta/tasks/cartpole_scfd_blend_highhorizon.json`
- **Heavy mass vector:** `meta/vectors/cartpole/2025-09-29_heavy.json`
  - Source run: `runs/scfd_cma_heavy`
  - Performance: 2,500-step balance (min/max = 2,500/2,500; std = 0)
  - Task descriptor: `meta/tasks/cartpole_scfd_blend_heavy.json`

## Reaction-Diffusion (Gray-Scott)

- **Spots vector:** `meta/vectors/gray_scott/2025-09-29_half.json`
  - Run: `runs/gray_scott_cma_half`
  - Mean MSE: 3.686e-03 (std 3.6e-07)
  - Task descriptor: `meta/tasks/gray_scott_spots.json`
- **Stripes vector:** `meta/vectors/gray_scott/2025-09-29_stripes.json`
  - Run: `runs/gray_scott_cma_stripes`
  - Mean MSE: 7.706e-02 (std 1.9e-07)
  - Task descriptor: `meta/tasks/gray_scott_stripes.json`
- **Checker vector:** `meta/vectors/gray_scott/2025-09-29_checker.json`
  - Run: `runs/gray_scott_cma_checker`
  - Mean MSE: 2.022e-01 (std 8.6e-09)
  - Task descriptor: `meta/tasks/gray_scott_checker.json`

## Heat Diffusion

- **Gradient vector:** `meta/vectors/heat_diffusion/2025-09-29_large.json`
  - Run: `runs/heat_diffusion_cma_large`
  - mean MSE 2.629e-03; std 3.9e-06
  - Task descriptor: `meta/tasks/heat_diffusion_gradient.json`
- **Hot corner vector:** `meta/vectors/heat_diffusion/2025-09-29_hotcorner.json`
  - Run: `runs/heat_diffusion_cma_hotcorner`
  - mean MSE 7.705e-05; std 5.5e-07
  - Task descriptor: `meta/tasks/heat_diffusion_hotcorner.json`
- **Cool spot vector:** `meta/vectors/heat_diffusion/2025-09-29_coolspot.json`
  - Run: `runs/heat_diffusion_cma_coolspot`
  - mean MSE 1.124e-04; std 3.4e-07
  - Task descriptor: `meta/tasks/heat_diffusion_coolspot.json`
- **Periodic wrap vector:** `meta/vectors/heat_diffusion/2025-10-01_periodic_stripe.json`
  - Run: `runs/heat_periodic_cma_stripe`
  - mean MSE 4.419e-02; std 2.7e-05; wrap MSE 1.131e-04; latency 1.40 ms
  - Budget utilisation 1.00; control norm 0.137
  - Task descriptor: `meta/tasks/heat_diffusion_periodic.json`

## Flow Control (Cylinder Jets)

- **Baseline vector:** `meta/vectors/flow_cylinder/2025-09-29_large.json`
  - Run: `runs/flow_cylinder_cma_large`
  - Mean wake MSE: 4.402e-02 (std 3.7e-09)
  - Task descriptor: `meta/tasks/flow_cylinder_karman.json`
- **High inflow vector:** `meta/vectors/flow_cylinder/2025-09-30_highinflow.json`
  - Run: `runs/flow_cylinder_cma_highinflow`
  - Mean wake MSE: 1.517e-01 (std 0.0e+00)
  - Task descriptor: `meta/tasks/flow_cylinder_highinflow.json`
- **Constriction slit (pending):** `meta/vectors/flow_constriction/2025-10-01_slit_channel.json`
  - Run: `runs/flow_constriction_cma`
  - Status: placeholder vector; CMA sweep still pending
  - Task descriptor: `meta/tasks/flow_constriction.json`

## Wave-Field Shaping

- **Focus vector:** `meta/vectors/wave_field/2025-09-29_large.json`
  - Run: `runs/wave_field_cma_large`
  - Mean MSE: 1.540e-02 (std 1.0e-04)
  - Task descriptor: `meta/tasks/wave_field_focus.json`
- **Defocus vector:** `meta/vectors/wave_field/2025-09-30_defocus.json`
  - Run: `runs/wave_field_cma_defocus`
  - Mean MSE: 1.537e-02 (std 6.8e-05)
  - Task descriptor: `meta/tasks/wave_field_defocus.json`

### Meta-Learning Snapshot

- All tuned controllers are tracked in `meta/tasks/` and `meta/vectors/`.
- `scripts/meta_smoke.py` verifies each vector (baseline seeds 0,1,2).
- Latest smoke metrics: see `comparisons/meta_smoke_latest.json`.

## Pending 2D Extensions

- **Heat anisotropic:** `meta/vectors/heat_diffusion/2025-10-01_anisotropic_tilted.json` - scaffolding + tests landed; optimisation pending.
- **Heat obstacle:** `meta/vectors/heat_diffusion/2025-10-01_obstacle_hotcorner.json` - obstacle geometry + budget logging ready for tuning.
- **Flow constriction:** `meta/vectors/flow_constriction/2025-10-01_slit_channel.json` - channel slit geometry + CMA script staged.
