from __future__ import annotations

import numpy as np

from .ops import grad, norm_sq_grad, laplacian
from .params import PhysicsParams

Array = np.ndarray


def coherence_p_to_m(p: float) -> float:
    return (p + 2.0) / 2.0


def kinetic_energy_density(theta_dot: Array, physics: PhysicsParams) -> Array:
    return 0.5 * physics.beta * theta_dot ** 2


def coherence_energy_density(theta: Array, physics: PhysicsParams, dx: float | tuple[float, float] | None = None) -> Array:
    grad_sq = norm_sq_grad(theta, dx=dx)
    q = grad_sq + physics.epsilon ** 2
    p = float(max(physics.coherence_p, 1))
    return physics.alpha * q ** (-p / 2.0)


def potential_energy_density(theta: Array, physics: PhysicsParams) -> Array:
    return physics.potential.energy(theta)


def curvature_energy_density(theta: Array, physics: PhysicsParams, dx: float | tuple[float, float] | None = None) -> Array:
    if not physics.curvature_penalty.enabled:
        return np.zeros_like(theta)
    lap = laplacian(theta, dx=dx)
    return physics.curvature_penalty.delta * lap ** 2


def cross_gradient_energy_density(
    C: Array,
    K: Array,
    physics: PhysicsParams,
    dx: float | tuple[float, float] | None = None,
) -> Array:
    cfg = physics.cross_gradient
    if not cfg.enabled:
        return np.zeros_like(C)
    gx_c, gy_c = grad(C, dx=dx)
    gx_k, gy_k = grad(K, dx=dx)
    mag_c = np.sqrt(gx_c ** 2 + gy_c ** 2 + cfg.epsilon ** 2)
    mag_k = np.sqrt(gx_k ** 2 + gy_k ** 2 + cfg.epsilon ** 2)
    return cfg.gamma * mag_c * mag_k


def total_energy_density(
    theta: Array,
    theta_dot: Array,
    physics: PhysicsParams,
    dx: float | tuple[float, float] | None = None,
    C: Array | None = None,
    K: Array | None = None,
    C_dot: Array | None = None,
    K_dot: Array | None = None,
) -> Array:
    total = kinetic_energy_density(theta_dot, physics)
    total += coherence_energy_density(theta, physics, dx=dx)
    total += 0.5 * physics.gamma * norm_sq_grad(theta, dx=dx)
    total += potential_energy_density(theta, physics)
    total += curvature_energy_density(theta, physics, dx=dx)
    if C is not None and K is not None:
        total += cross_gradient_energy_density(C, K, physics, dx=dx)
        if C_dot is not None:
            total += 0.5 * physics.cross_gradient.beta * C_dot ** 2
        if K_dot is not None:
            total += 0.5 * physics.cross_gradient.beta * K_dot ** 2
    return total


def metropolis_accept(delta_energy: Array, temperature: float, clip: tuple[float, float]) -> Array:
    logits = -delta_energy / max(temperature, 1e-8)
    prob = 1.0 / (1.0 + np.exp(-logits))
    lo, hi = clip
    return np.clip(prob, lo, hi)


def compute_total_energy(
    theta: Array,
    theta_dot: Array,
    physics: PhysicsParams,
    dx: float | tuple[float, float] | None = None,
    C: Array | None = None,
    K: Array | None = None,
    C_dot: Array | None = None,
    K_dot: Array | None = None,
) -> float:
    density = total_energy_density(
        theta,
        theta_dot,
        physics,
        dx=dx,
        C=C,
        K=K,
        C_dot=C_dot,
        K_dot=K_dot,
    )
    return float(np.mean(density))
