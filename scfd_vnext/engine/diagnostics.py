from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .energy import (
    coherence_energy_density,
    cross_gradient_energy_density,
    curvature_energy_density,
    kinetic_energy_density,
    potential_energy_density,
    total_energy_density,
    coherence_p_to_m,
)
from .ops import norm_sq_grad
from .params import PhysicsParams

Array = np.ndarray


def compute_energy_report(
    theta: Array,
    theta_dot: Array,
    physics: PhysicsParams,
    dx: float | tuple[float, float] | None = None,
    C: Array | None = None,
    K: Array | None = None,
    C_dot: Array | None = None,
    K_dot: Array | None = None,
) -> Dict[str, float]:
    kinetic = float(np.mean(kinetic_energy_density(theta_dot, physics)))
    gradient = float(0.5 * physics.gamma * np.mean(norm_sq_grad(theta, dx=dx)))
    coherence = float(np.mean(coherence_energy_density(theta, physics, dx=dx)))
    potential = float(np.mean(potential_energy_density(theta, physics)))
    curvature = float(np.mean(curvature_energy_density(theta, physics, dx=dx)))
    cross = 0.0
    if C is not None and K is not None:
        cross = float(
            np.mean(
                cross_gradient_energy_density(
                    C,
                    K,
                    physics,
                    dx=dx,
                )
            )
        )
    total = float(
        np.mean(
            total_energy_density(
                theta,
                theta_dot,
                physics,
                dx=dx,
                C=C,
                K=K,
                C_dot=C_dot,
                K_dot=K_dot,
            )
        )
    )
    return {
        "kinetic": kinetic,
        "gradient": gradient,
        "coherence": coherence,
        "potential": potential,
        "curvature": curvature,
        "cross": cross,
        "total": total,
    }


def coherence_metrics(theta: Array, physics: PhysicsParams, dx: float | tuple[float, float] | None = None) -> Dict[str, float]:
    grad_sq = norm_sq_grad(theta, dx=dx)
    q = grad_sq + physics.epsilon ** 2
    p = float(max(physics.coherence_p, 1))
    m = coherence_p_to_m(p)
    f_field = physics.alpha * p * q ** (-m)
    fraction_supercritical = float(np.mean(f_field > physics.gamma))
    return {
        "f_mean": float(np.mean(f_field)),
        "f_max": float(np.max(f_field)),
        "fraction_supercritical": fraction_supercritical,
    }


def radial_spectrum(field: Array, dx: float = 1.0, bins: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    field = np.asarray(field)
    fft = np.fft.fftshift(np.fft.fft2(field))
    power = np.abs(fft) ** 2
    nx, ny = field.shape
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    kx, ky = np.meshgrid(fx, fy, indexing="ij")
    radii = np.sqrt(kx ** 2 + ky ** 2)
    max_radius = radii.max()
    bins_edges = np.linspace(0.0, max_radius, bins + 1)
    spectrum = np.zeros(bins)
    counts = np.zeros(bins)
    for i in range(bins):
        mask = (radii >= bins_edges[i]) & (radii < bins_edges[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            spectrum[i] = power[mask].mean()
    centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
    return centers, spectrum


def energy_drift(energies: Array) -> float:
    energies = np.asarray(energies)
    if energies.size < 2:
        return 0.0
    start = energies[0]
    end = energies[-1]
    denom = max(abs(start), 1e-8)
    return float((end - start) / denom)


def predictability_horizon(
    series_a: Array,
    series_b: Array,
    threshold: float,
) -> int:
    diff = np.abs(np.asarray(series_a) - np.asarray(series_b))
    exceed = np.where(diff > threshold)[0]
    if exceed.size == 0:
        return diff.size
    return int(exceed[0])
