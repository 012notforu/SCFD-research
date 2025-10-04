# Standard PDE Control Benchmarks for Machine Learning Methods

The field of PDE control has matured significantly in recent years, with several benchmark collections emerging as standard evaluation frameworks for machine learning and optimization algorithms. **The most comprehensive and widely-adopted benchmarks come from fluid dynamics control, with PDEBench and CFDBench leading the way**, while other domains have more specialized but equally rigorous evaluation frameworks. These benchmarks are specifically designed to test reinforcement learning and control algorithms on medium to hard difficulty problems that challenge current methods.

## Fluid flow control benchmarks

### CFDBench and PDEBench lead modern evaluation standards

**CFDBench** (2023) represents the first comprehensive benchmark specifically tailored for evaluating neural operators in CFD problems. Developed by Luo et al., it includes **302K frames of velocity and pressure fields** across 739 cases, with the **periodic Karman vortex street** (flow around cylinder) serving as a core benchmark. The framework provides varying boundary conditions, Reynolds numbers, and domain geometries, making it ideal for testing energy-based optimization methods due to comprehensive state space representation.

**PDEBench** (2022) won the SimTech Best Paper Award and stands as the most comprehensive PDE benchmark collection. It includes 2D incompressible Navier-Stokes equations with multiple Reynolds numbers and boundary conditions, implemented as a full Python framework with pre-trained models. The benchmark's strength lies in its standardized evaluation metrics and energy-based loss functions.

**The Rabault cylinder control benchmark** (2019) remains the most cited landmark study in deep reinforcement learning for flow control. Published in Journal of Fluid Mechanics with over 3000 citations, it established the standard for 2D circular cylinder control at Re=100 with synthetic jets, achieving 8% drag reduction. Multiple extensions have pushed performance to 30-40% drag reduction across Reynolds numbers 60-2000, making it the gold standard for method comparison.

### Implementation frameworks enable practical deployment

**PhiFlow** from TU Munich provides differentiable physics for PyTorch, TensorFlow, and JAX with 2D/3D fluid simulations. **JAX-CFD and JAX-Fluids** from Google and TU Munich offer high-performance differentiable CFD with full automatic differentiation capabilities. These frameworks excel for energy-based optimization due to native gradient-based optimization support.

The **PDE Control Gym** (2024) provides the first dedicated benchmark for learning-based boundary control of PDEs, specifically designed for 2D Navier-Stokes control with RL gym interfaces. Success metrics focus on drag coefficient minimization, Strouhal number changes, and energy-efficient actuation.

## Diffusion and heat steering benchmarks

### Theoretical foundations meet practical implementation challenges

The SIAM Journal on Control and Optimization has established theoretical frameworks for parabolic PDE-constrained optimization, though these focus on specific test cases rather than standardized benchmark suites. **Deal.II Tutorial Programs** provide the most accessible entry point with Step-26 (time-dependent heat equation) and Step-24 (photoacoustic wave equation), offering comprehensive documentation and multiple difficulty levels.

**FEniCS heat equation implementations** use θ-scheme discretization with analytical solutions for validation. Standard test cases include unit square domains, L-shaped domains with corner singularities, and time-dependent boundary conditions. Parameters range from thermal diffusivity values of 10⁻² to 1, with extensive Python tutorials available.

The field's main limitation is the **lack of standardized benchmark suites** comparable to computer vision or NLP. Current benchmarks vary from academic toy examples to industrial-scale simulations, with limited machine learning integration. For energy-based optimization, these benchmarks provide natural energy formulations through temperature/concentration gradients and variational methods.

## Wave-field shaping benchmarks

### Acoustic and optical control across multiple physics domains

**Transcranial ultrasound benchmarks** from IFCN (2022) represent the gold standard with 9 geometric complexity levels and standardized 500 kHz frequency testing. These benchmarks test phase correction algorithms across multiple solvers (k-Wave, mSOUND, Salvus) with established pressure field accuracy metrics.

**Wave Field Synthesis (WFS) benchmarks** use the EMPAC High-Resolution Modular Array with 18 linear arrays and 4-inch drivers at 12cm spacing. These benchmarks test virtual source localization up to 1400 Hz with spatial aliasing avoidance and listener position independence as key metrics.

**Acoustic black hole damping** provides broadband vibration suppression benchmarks with exponential thickness variation and multiple resonance frequencies. Success metrics include broadband resonance peak reduction and energy dissipation through wave focusing.

### Computational frameworks span specialized and general-purpose tools

**PhiFlow and JAX-based frameworks** (j-Wave, JAX-CFD, Exponax) provide differentiable wave equation solvers with GPU acceleration. **FEniCS4EM** specifically targets electromagnetic wave propagation using Nedelec elements for Maxwell equations.

Energy-based optimization methods align naturally with wave control due to **wave amplitude control** (±5% accuracy targets), **phase control** (±π/32 radian precision), and **energy focusing** (>10:1 concentration ratios).

## Reaction-diffusion design benchmarks

### Gray-Scott and Gierer-Meinhardt models dominate evaluation

**The Gray-Scott model** serves as the most widely established and extensively cited benchmark. Standard parameter ranges include diffusion ratio D=2, with a ∈ [0, 0.07] and b ∈ [0.02, 0.068] producing diverse patterns: labyrinthine (a=0.037, b=0.06), spots (a=0.03, b=0.062), and moving gliders (a=0.014, b=0.054). Multiple GitHub repositories and VisualPDE interactive simulations provide accessible implementations.

**Gierer-Meinhardt networked control** benchmarks from the Royal Society (2022) represent cutting-edge optimal control research. These test control algorithms across different network topologies (Erdös-Rényi, Scale-free, Regular lattice) with mean absolute relative error below 5% as success criteria. The framework demonstrates how network structure affects control difficulty, with scale-free networks requiring more control effort.

**The Schnakenberg model** provides established autocatalytic reaction benchmarks from mathematical biology literature. Standard test cases include Hopf bifurcation control, pattern switching between spots and stripes, and fractional-order extensions for memory effects.

### FEniCS and specialized frameworks enable sophisticated modeling

**SMART (Spatial Modeling Algorithms for Reactions and Transport)** provides production-ready implementations with extensive documentation for 3-species reaction systems, Turing pattern formation, and complex cell geometries. **FEniCS tutorial examples** offer accessible entry points with Crank-Nicolson time stepping and comprehensive reaction-diffusion modeling capabilities.

These benchmarks excel for energy-based optimization through **variational formulations** (pattern formation as energy minimization), **optimal control energy functions** (quadratic cost functionals), and **thermodynamic analogies** (free energy minimization in pattern formation).

## Fluid mixing and dye tracking benchmarks

### Mathew et al. benchmark establishes gold standard for mixing control

**The Mathew et al. benchmark** (2007) from Journal of Fluid Mechanics stands as the most widely cited standard in mixing control, referenced in over 50 papers. It tests 2D periodic domains with cellular flow patterns across Péclet numbers Pe ∈ (0,∞], commonly at Pe = ∞, 10³, 10⁴. **Mix-variance minimization** Φ(c) = ||c||²_{H^{-1/2}} serves as the primary objective with energy constraint ∫₀¹ ∫_T² u² dxdt = 1.25.

This benchmark has been validated in PhiFlow, JAX-CFD, and reinforcement learning studies (Nature Scientific Reports 2022), making it ideal for energy-based optimization with established baseline results and comprehensive framework support.

**T-junction mixing benchmarks** represent the microfluidics standard with 51mm diameter pipes, Reynolds numbers 0.1 ≤ Re ≤ 100, and Péclet numbers 10² - 10⁶. Success metrics include mixing index (0-1), segregation coefficients, and pressure drop measurements. Python implementations include MMFT-Simulator and custom microfluidics packages.

### Difficulty levels span fundamental to industrial applications

**Easy benchmarks** include T-junction mixing and basic serpentine mixers, while **medium difficulty** covers Mathew et al. cellular flows and lid-driven cavity mixing. **Hard benchmarks** involve chaotic advection control and cavity oscillation suppression, with **expert level** addressing turbulent mixing control and multiphase systems.

## Active flow around flexible structures benchmarks

### Turek FSI benchmark series provides comprehensive validation

**The Turek FSI benchmark series** (FSI1, FSI2, FSI3) from TU Dortmund represents the standard validation benchmark in the FSI community. FSI2 features large oscillations in a 2D channel with cylinder and flexible flag, creating challenging coupling dynamics for algorithms. With fluid density ρ=1000 kg/m³, Reynolds numbers around 100-200, and strong geometric nonlinearities, this benchmark provides **extensive reference solutions** and mesh convergence studies.

**Kratos Multiphysics** provides the reference implementation, while FEniCS-based solvers offer research flexibility. Control applications focus on flow control via cylinder rotation and downstream actuation, with success metrics including displacement amplitudes, oscillation frequencies, and drag/lift forces.

**The BACT wing benchmark** from NASA Langley represents the aerospace standard for flutter suppression validation. Using NACA 0012 airfoil sections with pitch and plunge apparatus, it tests control across subsonic-transonic ranges with classical flutter, transonic stall flutter, and plunge instability boundaries. Typical improvements show 25-40% increases in critical flutter speed with robust stability across flight envelopes.

### VIV control provides offshore engineering validation

**Vortex-induced vibration benchmarks** test circular cylinders across Reynolds numbers 10²-10⁵ with Strouhal number St = 0.2 and lock-in phenomena. Control methods range from passive (splitter plates, helical strakes) to active (rotating cylinders, closed-loop feedback), achieving up to 99% amplitude suppression. OpenFOAM-Python coupling and FEniCS implementations support both energy-based optimization and VIV energy harvesting applications.

## Electromagnetic and potential-field control benchmarks

### IEEE and EMCC standards ensure community-wide validation

**The IEEE Electromagnetic Code Consortium (EMCC)** has established comprehensive benchmarks including NASA almond targets, cone-sphere configurations, and antenna arrays on finite metal plates. Testing spans 700 MHz to 16 GHz with aluminum targets and precise surface control, validated across FISC, AIM, and XPATCH reference implementations.

**IEEE Standard 1597** provides systematic validation frameworks for thin dipole antennas, loop antennas, rectangular cavities with apertures, and monopoles on finite plates. Success metrics include input impedance accuracy, electromagnetic field values, far-field radiation patterns, and shielding effectiveness measurements.

**Austin Benchmark Suites** from UT Austin CEM Group offer freely available computational bioelectromagnetics problems with hidden reference solutions, error measures, and cost comparisons for performance validation.

### Modern frameworks enable differentiable electromagnetics

**FEniCS4EM** provides Maxwell solvers using Nédélec elements for time-harmonic problems, electromagnetic scattering with PML, and 3D axisymmetric applications. **Jaxwell** combines JAX with Maxwell FDFD solvers for GPU acceleration and automatic differentiation, making it excellent for nanophotonic inverse design and energy-based optimization.

Energy-based optimization proves highly suitable due to **Maxwell energy functionals** (∫(|E|² + |H|²)dV minimization), **Poisson energy** (∫|∇φ|²dV with source constraints), and **focusing objectives** with concentrated energy density targets.

## Soft robotics in continuum media benchmarks

### PlasticineLab and SSMR establish state-of-the-art standards

**PlasticineLab** from MIT/IBM/Stanford represents a well-recognized differentiable soft-body manipulation benchmark using the DiffTaichi system. It tests elastic/plastic deformation with various material properties (Young's modulus, Poisson's ratio, plasticity parameters) across multiple object geometries. Control objectives include shape manipulation tasks (rolling, folding, cutting), goal-conditioned deformation, and multi-tool manipulation with task completion rate and trajectory efficiency as success metrics.

**Spectral Submanifold Reduction (SSMR)** benchmarks from Stanford/ETH Zurich (Nature Robotics, 2025) represent state-of-the-art continuum robot control. The diamond robot with 9768 DOF and 4-cable actuation, along with the trunk robot (4254 DOF, 8-cable actuation), test real-time control with constraints. Success metrics include Integrated Squared Error, constraint violation ratios, and solve times, with high relevance for energy-based methods through dominant dynamics and energy-based model reduction.

**The TDCR Benchmarking Framework** from University of Toronto provides standard methodology for tendon-driven continuum robots with multiple backbone representations and tendon routing configurations. Success metrics focus on position error (% of robot length), orientation error (degrees), and computation time.

### SOFA ecosystem provides comprehensive simulation capabilities

**SOFA Soft Robotics Plugin** serves as the leading open-source platform with real-time FEM solvers, various material models, and actuator models for pneumatic, tendon-driven, and magnetic systems. **SOniCS integration** with FEniCS enables automated finite element tensor generation and error-controlled simulations with very high energy-based optimization relevance through direct energy functional minimization.

## Synthesis and recommendations

The benchmark landscape reveals **fluid dynamics control as the most mature domain** with standardized collections (PDEBench, CFDBench) and widely-adopted individual benchmarks (Rabault cylinder). **Electromagnetic control benefits from IEEE/EMCC industry standards**, while **soft robotics shows rapid advancement** with cutting-edge frameworks like PlasticineLab and SSMR.

**For energy-based optimization research, the most promising starting points include:**
- Rabault cylinder benchmark for fluid control (most validated, multiple Reynolds numbers)  
- Mathew et al. mixing benchmark for scalar transport (gold standard with extensive validation)
- Turek FSI2 for multi-physics coupling (comprehensive reference solutions)
- Gray-Scott model for reaction-diffusion (extensive parameter studies)
- PlasticineLab for soft robotics (differentiable physics with gradient-based optimization)

These benchmarks provide **natural energy formulations, gradient availability through modern frameworks, and established baseline performance** that enables rigorous comparison with existing methods. The convergence toward differentiable physics frameworks (PhiFlow, JAX-CFD, DiffTaichi) particularly supports energy-based optimization approaches that can leverage automatic differentiation for efficient gradient computation.