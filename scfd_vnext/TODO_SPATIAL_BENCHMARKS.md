# Spatial Control Benchmark To-Do

This tracker lives at the root so we can coordinate new SCFD benchmarks. We only check a box after:
1. Implementation lands in the repo.
2. User has run/verified the benchmark locally.
3. Learned vectors (or policies) are dropped into the meta-learning corpus.

## Legend
- [ ] Planned
- [~] In progress
- [x] Completed (code merged + verified + logged for meta-learning)

---

## Flow Control: Karman Cylinder / PDEBench
- [x] Spec out SCFD boundary-actuation interface for cylinder jets (document control knobs).
- [x] Implement benchmark runner (PDEBench or PDE Control Gym hook) with SCFD controller.
- [x] Tune baseline (hand-set weights) and log reference metrics (drag, Strouhal).
- [x] Run evolutionary search and save best vector plus artifacts.
- [x] Add benchmark and vectors to meta-learning dataset (tag with `flow_karman`).

## Reaction-Diffusion: Gray-Scott Patterns
- [x] Define multi-field coupling in SCFD config (u/v fields, reaction coefficients).
- [x] Build benchmark script to reach/maintain target pattern (e.g., spots/stripes).
- [x] Baseline run and metrics (pattern correlation, energy drift).
- [x] Optimise controller, archive best vector and visuals.
- [x] Register dataset entry (`reaction_gray_scott`) for meta-learning.

## Heat Steering / Diffusion Control
- [x] Choose canonical domain (unit square plus obstacles) and boundary control slots.
- [x] Implement heat benchmark: hold temperature profile or move hot spot.
- [x] Baseline evaluation (temperature RMSE, energy cost).
- [x] Optimise controller, store vector and heat maps.
- [x] Add to meta-learning corpus (`diffusion_heat_control`).

## Wave-Field Shaping (Transcranial / WFS)
- [x] Scope feasibility (phase encode in SCFD, target metrics).
- [x] Prototype simplified 2D wave benchmark.
- [x] Decide go/no-go depending on complexity versus impact.

## Future Stretch: Soft Robotics / PlasticineLab
- [ ] Research required SCFD extensions (elasticity, contact handling).
- [ ] Revisit after fluids and diffusion benchmarks are solid.

---

### Meta-Learning Integration Checklist
(Use per benchmark once ready)
- [x] Prepare task descriptor JSON (physics, reward, seeds).
- [x] Serialize best parameter vector(s) plus metadata.
- [x] Update meta-training loader to sample the new task.
- [x] Validate cross-task adaptation run includes the new task.
\n## Variant Benchmarks\n- [x] Cart-pole: high-horizon (steps 4000).\n- [x] Cart-pole: heavy pole parameters.\n- [x] Gray-Scott: stripes target.\n- [x] Gray-Scott: checker target.\n- [x] Heat diffusion: hot corner target.\n- [x] Heat diffusion: cool spot target.\n- [x] Flow control: high-inflow variant.\n- [x] Wave field: defocus target.\n
\n## Extended 2D Coverage\n- [x] Heat: periodic boundary conditions benchmark (vector logged, CMA tuned).\n- [~] Heat: anisotropic diffusion tensor variant (controllers scaffolded).\n- [~] Heat: obstacle + hot corner with control budget (sim + CMA stubs ready).\n- [~] Flow: channel with constriction/slit (new geometry + trainer in repo).\n- [ ] Flow: regime sweep (subcritical / critical / supercritical).\n- [ ] Wave: standing mode cavity target.\n- [ ] Wave: partial sensors + action delay variant.\n- [ ] Gray-Scott: near-Turing threshold / bifurcation hover.\n- [ ] Mobile actuator demo (e.g., moveable heater).\n- [ ] Heat inverse problem (hidden source localization).\n- [ ] Flow: redundant body-force actuators with budget.\n- [ ] Wave: mode switch mid-run (focus?standing).\n- [ ] Robustness battery (disturbances / adversary) per domain.\n- [ ] ARC-style diffusion transform (rotate/reflect motif).\n- [ ] Routing & sorting of multiple blobs without collision.\n- [ ] Front tracking / curvature-bounded propagation.\n- [ ] Parameter ID tasks (diffusivity map, viscosity jump).\n- [ ] Latency profiling & budget tagging for each vector.\n



