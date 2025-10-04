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

## Variant Benchmarks
- [x] Cart-pole: high-horizon (steps 4000).
- [x] Cart-pole: heavy pole parameters.
- [x] Gray-Scott: stripes target.
- [x] Gray-Scott: checker target.
- [x] Heat diffusion: hot corner target.
- [x] Heat diffusion: cool spot target.
- [x] Flow control: high-inflow variant.
- [x] Wave field: defocus target.

## Extended 2D Coverage
- [x] Heat: periodic boundary conditions benchmark (vector logged, CMA tuned).
- [~] Heat: anisotropic diffusion tensor variant (controllers scaffolded).
- [~] Heat: obstacle + hot corner with control budget (sim + CMA stubs ready).
- [~] Flow: channel with constriction/slit (new geometry + trainer in repo).
- [x] Flow: regime sweep (subcritical / critical / supercritical).
- [~] Wave: standing mode cavity target (m=2, n=3 vector logged; smoke pending).
- [~] Wave: partial sensors + action delay variant (20% coverage / delay-3 vector logged).
- [ ] Gray-Scott: near-Turing threshold / bifurcation hover.
- [ ] Mobile actuator demo (e.g., moveable heater).
- [ ] Heat inverse problem (hidden source localization).
- [ ] Flow: redundant body-force actuators with budget.
- [ ] Wave: mode switch mid-run (focus?standing).
- [ ] Robustness battery (disturbances / adversary) per domain.
- [ ] ARC-style diffusion transform (rotate/reflect motif).
- [ ] Routing & sorting of multiple blobs without collision.
- [ ] Front tracking / curvature-bounded propagation.
- [ ] Parameter ID tasks (diffusivity map, viscosity jump).
- [ ] Latency profiling & budget tagging for each vector.


