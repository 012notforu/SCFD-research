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
- **Anisotropic tensor vector:** `meta/vectors/heat_diffusion/2025-10-01_anisotropic_tilted.json`
  - Run: `runs/heat_anisotropic_cma`
  - mean MSE 8.713e-03; std 5.1e-05
  - Principal ratio 5.29
  - Task descriptor: `meta/tasks/heat_diffusion_anisotropic.json`
- **Obstacle hot corner vector:** `meta/vectors/heat_diffusion/2025-10-01_obstacle_hotcorner.json`
  - Run: `runs/heat_obstacle_cma`
  - mean MSE 2.030e-02; std 3.2e-05
  - Corner MSE 6.895e-02
  - Task descriptor: `meta/tasks/heat_diffusion_obstacle.json`

## Flow Control (Cylinder Jets)

- **Baseline vector:** `meta/vectors/flow_cylinder/2025-09-29_large.json`
  - Run: `runs/flow_cylinder_cma_large`
  - Mean wake MSE: 4.402e-02 (std 3.7e-09)
  - Task descriptor: `meta/tasks/flow_cylinder_karman.json`
- **High inflow vector:** `meta/vectors/flow_cylinder/2025-09-30_highinflow.json`
  - Run: `runs/flow_cylinder_cma_highinflow`
  - Mean wake MSE: 1.517e-01 (std 0.0e+00)
  - Task descriptor: `meta/tasks/flow_cylinder_highinflow.json`
- **Regime sweep vector:** `meta/vectors/flow_regime/2025-10-04_multi.json`
  - Run: `runs/flow_regime_cma`
  - Mean wake MSE: 3.638e-02 (std 3.2e-02)
  - Regime wake MSEs: 0.6: 3.914e-03, 1.0: 2.542e-02, 1.4: 7.981e-02
  - Task descriptor: `meta/tasks/flow_regime_sweep.json`
- **Constriction slit vector:** `meta/vectors/flow_constriction/2025-10-01_slit_channel.json`
  - Run: `runs/flow_constriction_cma`
  - Throughput: 0.539; backflow: 0.000
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
- **Standing cavity vector:** `meta/vectors/wave_field/2025-10-04_cavity_m2n3.json`
  - Run: `runs/wave_cavity_cma`
  - Mean MSE: 5.669e-02 (std 3.9e-05)
  - Standing mode m=2, n=3
  - Task descriptor: `meta/tasks/wave_field_cavity_mode23.json`
- **Partial sensors vector:** `meta/vectors/wave_field/2025-10-04_partial_focus.json`
  - Run: `runs/wave_partial_cma`
  - Mean MSE: 1.404e-02 (std 4.0e-05)
  - Sensor coverage 0.20; action delay 3
  - Task descriptor: `meta/tasks/wave_field_partial_focus.json`

### Meta-Learning Snapshot

- All tuned controllers are tracked in `meta/tasks/` and `meta/vectors/`.
- `scripts/meta_smoke.py` verifies each vector (baseline seeds 0,1,2).
- Latest smoke metrics: see `comparisons/meta_smoke_latest.json`.

## Pending 2D Extensions\n\n- **Gray-Scott near-Turing hover:** planned multi-parameter sweep to stabilise bifurcation edge.\n- **Mobile actuator demo:** moveable heater/actuator benchmark with shifting control point.\n- **Heat inverse problem:** hidden source localisation tasks for diffusion.\n- **Flow redundant actuators:** body-force budget with multiple actuators.\n- **Wave mode switch:** focus→standing transition mid-run with delay compensation.\n
