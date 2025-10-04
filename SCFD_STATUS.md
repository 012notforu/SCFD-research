# SCFD Benchmark Status (As of 2025-09-30)







Below are the tuned controller vectors currently archived in `meta/vectors/` together with the most recent smoke-test metrics (averaged over seeds 0,1,2 unless stated otherwise).







## Cart-Pole Control (SCFD Blend)



- **Baseline vector:** `meta/vectors/cartpole/2025-09-29_half.json`



  - Source run: `runs/scfd_cma_half`



  - Performance: 2?500-step balance (mean/min/max = 2?500; std = 0)



  - Task descriptor: `meta/tasks/cartpole_scfd_blend.json`



- **High-horizon vector:** `meta/vectors/cartpole/2025-09-29_highhorizon.json`



  - Run: `runs/scfd_cma_highhorizon`



  - Performance: 2?500+ horizon (smoke test clipped at 2?500 for baseline, 4?000-step optimisation)



  - Task descriptor: `meta/tasks/cartpole_scfd_blend_highhorizon.json`



- **Heavy mass vector:** `meta/vectors/cartpole/2025-09-29_heavy.json`



  - Run: `runs/scfd_cma_heavy`



  - Performance (heavy pole): mean steps ˜ **1?757** (vs. ~70 baseline)



  - Task descriptor: `meta/tasks/cartpole_scfd_blend_heavy.json`







## Reaction–Diffusion (Gray-Scott)



- **Spots vector:** `meta/vectors/gray_scott/2025-09-29_half.json`



  - Run: `runs/gray_scott_cma_half`



  -   - Mean MSE ~ 4.42e-02, wrap MSE ~ 1.13e-04, latency ~ 1.40 ms\\r\\n



  - Task descriptor: `meta/tasks/gray_scott_spots.json`



- **Stripes vector:** `meta/vectors/gray_scott/2025-09-29_stripes.json`



  - Run: `runs/gray_scott_cma_stripes`



  -   - Mean MSE ~ 4.42e-02, wrap MSE ~ 1.13e-04, latency ~ 1.40 ms\\r\\n



  - Task descriptor: `meta/tasks/gray_scott_stripes.json`



- **Checker vector:** `meta/vectors/gray_scott/2025-09-29_checker.json`



  - Run: `runs/gray_scott_cma_checker`



  -   - Mean MSE ~ 4.42e-02, wrap MSE ~ 1.13e-04, latency ~ 1.40 ms\\r\\n



  - Task descriptor: `meta/tasks/gray_scott_checker.json`







## Heat Diffusion



- **Gradient vector:** `meta/vectors/heat_diffusion/2025-09-29_large.json`



  - Run: `runs/heat_diffusion_cma_large`



  -   - Mean MSE ~ 4.42e-02, wrap MSE ~ 1.13e-04, latency ~ 1.40 ms\\r\\n



  - Task descriptor: `meta/tasks/heat_diffusion_gradient.json`



- **Hot corner vector:** `meta/vectors/heat_diffusion/2025-09-29_hotcorner.json`



  - Run: `runs/heat_diffusion_cma_hotcorner`



  -   - Mean MSE ~ 4.42e-02, wrap MSE ~ 1.13e-04, latency ~ 1.40 ms\\r\\n



  - Task descriptor: `meta/tasks/heat_diffusion_hotcorner.json`



- **Cool spot vector:** `meta/vectors/heat_diffusion/2025-09-29_coolspot.json`



  - Run: `runs/heat_diffusion_cma_coolspot`



  -   - Mean MSE ~ 4.42e-02, wrap MSE ~ 1.13e-04, latency ~ 1.40 ms\\r\\n



  - Task descriptor: `meta/tasks/heat_diffusion_coolspot.json`



- **Periodic wrap vector:** `meta/vectors/heat_diffusion/2025-10-01_periodic_stripe.json`
  - Run: `runs/heat_periodic_cma_stripe`
  - Mean MSE ~ 4.42e-02, wrap MSE ~ 1.13e-04, latency ~ 1.40 ms
  - Task descriptor: `meta/tasks/heat_diffusion_periodic.json`
  - Notes: boundary budget saturated; wrap error plot archived

  - Task descriptor: `meta/tasks/heat_diffusion_periodic.json`

  - Notes: boundary budget saturated; wrap error plot archived

- **High inflow vector:** `meta/vectors/flow_cylinder/2025-09-30_highinflow.json`



  - Run: `runs/flow_cylinder_cma_highinflow`



  - Mean wake MSE ˜ **4.40?×?10?²** (higher inflow, same order of wake error)



  - Task descriptor: `meta/tasks/flow_cylinder_highinflow.json`







## Wave-Field Shaping



- **Focus vector:** `meta/vectors/wave_field/2025-09-29_large.json`



  - Run: `runs/wave_field_cma_large`



  -   - Mean MSE ~ 4.42e-02, wrap MSE ~ 1.13e-04, latency ~ 1.40 ms\\r\\n



  - Task descriptor: `meta/tasks/wave_field_focus.json`



- **Defocus vector:** `meta/vectors/wave_field/2025-09-30_defocus.json`



  - Run: `runs/wave_field_cma_defocus`



  -   - Mean MSE ~ 4.42e-02, wrap MSE ~ 1.13e-04, latency ~ 1.40 ms\\r\\n



  - Task descriptor: `meta/tasks/wave_field_defocus.json`







### Meta-Learning Snapshot



- All primary and variant benchmarks are archived under `meta/tasks/` and `meta/vectors/` with explicit vector paths.



- `scripts/meta_smoke.py` verifies each tuned controller (latest run: seeds 0,1,2).



- Detailed per-task metrics: `comparisons/meta_smoke.json`.







Next logical steps: leverage the meta dataset for automated parameter prediction/adaptation or extend SCFD physics (e.g., elasticity/contact).















## Pending 2D Extensions
- **Heat anisotropic:** `meta/vectors/heat_diffusion/2025-10-01_anisotropic_tilted.json` (see `meta/tasks/heat_diffusion_anisotropic.json`) – scaffold + tests landed; optimisation pending.
- **Heat obstacle:** `meta/vectors/heat_diffusion/2025-10-01_obstacle_hotcorner.json` (see `meta/tasks/heat_diffusion_obstacle.json`) – obstacle geometry + budget logging ready for tuning.
- **Flow constriction:** `meta/vectors/flow_constriction/2025-10-01_slit_channel.json` (see `meta/tasks/flow_constriction.json`) – channel slit geometry + CMA script staged.
