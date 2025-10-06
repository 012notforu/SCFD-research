import numpy as np
import pytest

from engine.energy import compute_total_energy
from engine.params import PhysicsParams, load_config
from engine.integrators import leapfrog_step

def finite_energy(x):
    return np.isfinite(x).all() and not np.isnan(x).any()

def test_energy_drift_small():
    """Test that energy conservation is maintained over short integrations."""
    # Create a simple test configuration
    cfg_path = "cfg/defaults.yaml"  # Use existing default config
    physics = load_config(cfg_path)
    
    # Initialize a simple field
    nx, ny = 32, 32
    theta = np.random.randn(nx, ny) * 0.1
    theta_dot = np.random.randn(nx, ny) * 0.1
    dx = 1.0
    
    E0 = compute_total_energy(theta, theta_dot, physics, dx=dx)
    
    # Run integration for a few steps
    for _ in range(32):
        theta, theta_dot = leapfrog_step((theta, theta_dot), physics, dx=dx, dt=0.01)
    
    E1 = compute_total_energy(theta, theta_dot, physics, dx=dx)
    
    assert finite_energy(np.array([E0, E1]))
    # Symplectic integrator should conserve energy within numerical tolerance
    # Allow 5% drift over 32 steps for this test
    if abs(E0) > 1e-10:  # Avoid division by very small numbers
        assert abs(E1 - E0) / abs(E0) < 0.05