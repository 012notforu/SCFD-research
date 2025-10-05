# Emergent Models Baseline Requirements

Extracted from Emergent_Models v14-04-25-1.pdf.

## Core Loop
- Maintain large grid state `S` with programmable initial state `S0` representing the policy/program.
- Apply fixed transition function `f` repeatedly: deterministic convolutional CA-style rule with optional stochastic perturbations.
- Continue iterations until halting predicate signals stop (`H` returns true) or maximum steps reached.
- Decode final state at halting step to build output symbols/actions.

## Transition Function (`f`)
- Simple, mostly local operator (e.g., Gaussian kernel convolution + pointwise nonlinearity as in Lenia).
- Support deterministic updates plus optional noise to encourage exploration.
- Configurable kernel shapes and activation function.

## Encoding / Decoding
- Encoding injects external input into designated region/channels of the state via additive or overwrite modes (e.g., one-hot activation of nearby cells).
- Decoding reads out statistics from boundary regions or dedicated channels; example strategy counts activations near right border, smooths with kernel, then softmax for symbol distribution.
- Support reward injection by writing to dedicated reward cells (proportional to reward magnitude).

## Halting Strategies
- Primary: monitor dedicated halting cell; stop when it crosses configurable threshold.
- Secondary safeguards: max iterations cap, optional stochastic halting trigger to avoid infinite loops, optional compute-time penalty logging.

## Optimization
- Treat initial state `S0` as genome; evolve using genetic algorithm (selection, crossover, mutation).
- Optional Bayesian/surrogate adjustments to bias mutations toward promising regions.
- Track reward-based fitness over episodes; maintain population to mitigate divergence.

## Sequential Operation
- Preserve state between inputs to provide memory; propagate halting state as next iteration seed.
- Allow optional reward feedback between steps before next input encoding.

## RL Environment Mapping (dummy 2D car example)
- Inputs: vehicle speed and LIDAR distances; encode into state grid.
- Outputs: acceleration and steering commands derived from decoded distribution.
- Rewards: encode into reward cells; include compute-time penalty for long runs.

## Metrics & Diagnostics
- Monitor symbol statistics, spectrum width, impulse response, predictability horizon (to compare with SCFD engine).
- Log halting times, iteration counts, reward trajectories, genome diversity.
