#!/usr/bin/env python3
"""Test that every vector in the registry can be loaded and applied successfully."""
import numpy as np
import json
from pathlib import Path

from orchestrator.pipeline import load_vector_registry, plan_for_environment
from engine import load_config, accel_theta
from engine.integrators import leapfrog_step
from utils.logging import create_run_directory, RunLogger

def test_vector_loading():
    """Test that all vectors can be loaded from files."""
    print("=== Testing Vector Loading ===")
    
    registry = load_vector_registry("runs")
    print(f"Found {len(registry)} vectors in registry")
    
    loaded_vectors = {}
    failed_vectors = []
    
    for i, vector_entry in enumerate(registry):
        try:
            with open(vector_entry.path) as f:
                data = json.load(f)
            vector = np.array(data["vector"], dtype=np.float32)
            loaded_vectors[vector_entry.vector_id] = {
                "vector": vector,
                "physics": vector_entry.physics,
                "length": len(vector),
                "entry": vector_entry
            }
            print(f"  {i+1:2d}. {vector_entry.vector_id} ({vector_entry.physics}) - {len(vector)} dims")
        except Exception as e:
            failed_vectors.append((vector_entry.vector_id, str(e)))
            print(f"  {i+1:2d}. {vector_entry.vector_id} - FAILED: {e}")
    
    print(f"\nLoading results: {len(loaded_vectors)} success, {len(failed_vectors)} failed")
    return loaded_vectors, failed_vectors

def test_vector_application():
    """Test that each vector can be applied to a SCFD simulation."""
    print("\n=== Testing Vector Application ===")
    
    # Load config and vectors
    cfg = load_config("cfg/defaults.yaml")
    loaded_vectors, failed_vectors = test_vector_loading()
    
    # Create test environment
    grid_shape = (16, 16)
    initial_theta = np.random.random(grid_shape).astype(np.float32) * 0.1
    initial_theta_dot = np.zeros_like(initial_theta)
    
    # Test results
    successful_applications = []
    failed_applications = []
    
    print(f"\nTesting application of {len(loaded_vectors)} vectors...")
    
    for vector_id, vector_data in loaded_vectors.items():
        try:
            # Reset to initial state for each test
            theta = initial_theta.copy()
            theta_dot = initial_theta_dot.copy()
            vector = vector_data["vector"]
            
            # Record initial energy
            initial_energy = float(np.mean(theta ** 2))
            
            # Apply vector influence (simple method - can be more sophisticated)
            if len(vector) >= 1:
                # Scale the first component and apply as uniform field influence
                influence = float(vector[0]) * 0.01  # Small scale factor
                theta += influence
            
            # Evolve SCFD for a few steps to see if it's stable
            for step in range(5):
                theta, theta_dot, _, _ = leapfrog_step(
                    theta,
                    theta_dot,
                    lambda f: accel_theta(f, cfg.physics, dx=1.0),
                    cfg.integration.dt
                )
                
                # Check for instability
                if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
                    raise ValueError("Simulation became unstable")
                
                if np.max(np.abs(theta)) > 10.0:
                    raise ValueError("Field values grew too large")
            
            # Record final energy
            final_energy = float(np.mean(theta ** 2))
            energy_change = final_energy - initial_energy
            
            successful_applications.append({
                "vector_id": vector_id,
                "physics": vector_data["physics"],
                "vector_length": vector_data["length"],
                "energy_change": energy_change,
                "stable": True
            })
            
            print(f"  OK {vector_id} ({vector_data['physics']}) - Energy change: {energy_change:.6f}")
            
        except Exception as e:
            failed_applications.append({
                "vector_id": vector_id,
                "physics": vector_data["physics"],
                "error": str(e)
            })
            print(f"  FAIL {vector_id} ({vector_data['physics']}) - {e}")
    
    print(f"\nApplication results: {len(successful_applications)} success, {len(failed_applications)} failed")
    return successful_applications, failed_applications

def test_orchestrator_with_all_vectors():
    """Test that orchestrator can select and use each vector type."""
    print("\n=== Testing Orchestrator with All Vectors ===")
    
    registry = load_vector_registry("runs")
    cfg = load_config("cfg/defaults.yaml")
    
    # Group vectors by physics type
    physics_groups = {}
    for vec in registry:
        physics = vec.physics
        if physics not in physics_groups:
            physics_groups[physics] = []
        physics_groups[physics].append(vec)
    
    print(f"Found {len(physics_groups)} physics types:")
    for physics, vecs in physics_groups.items():
        print(f"  {physics}: {len(vecs)} vectors")
    
    # Test orchestrator with different environment scenarios
    orchestrator_results = []
    
    for physics_type, vectors in physics_groups.items():
        try:
            # Create test environment that might favor this physics type
            test_obs = np.random.random((8, 8)).astype(np.float32)
            
            def env_factory():
                class TestEnvironment:
                    def __init__(self):
                        self.observation = test_obs
                        self.timestep = 0
                    
                    def step(self):
                        return {"observation": self.observation, "done": False}
                
                return TestEnvironment()
            
            # Try orchestrator with just this physics type's vectors
            plan = plan_for_environment(env_factory, registry=vectors, steps=1)
            selected_vector = plan.steps[0].vector
            
            orchestrator_results.append({
                "physics_type": physics_type,
                "selected_vector": selected_vector.vector_id,
                "available_count": len(vectors),
                "success": True
            })
            
            print(f"  OK {physics_type}: selected {selected_vector.vector_id}")
            
        except Exception as e:
            orchestrator_results.append({
                "physics_type": physics_type,
                "error": str(e),
                "available_count": len(vectors),
                "success": False
            })
            print(f"  FAIL {physics_type}: {e}")
    
    successful_orchestrator = sum(1 for r in orchestrator_results if r["success"])
    print(f"\nOrchestrator results: {successful_orchestrator}/{len(physics_groups)} physics types successful")
    
    return orchestrator_results

def generate_vector_compatibility_report():
    """Generate comprehensive report of vector registry compatibility."""
    print("\n=== Generating Vector Compatibility Report ===")
    
    # Run all tests
    loaded_vectors, loading_failures = test_vector_loading()
    application_results, application_failures = test_vector_application()
    orchestrator_results = test_orchestrator_with_all_vectors()
    
    # Create run directory for report
    run_dir = create_run_directory("vector_registry_test")
    logger = RunLogger(run_dir)
    
    # Generate summary report
    report = {
        "total_vectors": len(loaded_vectors) + len(loading_failures),
        "loading_success_rate": len(loaded_vectors) / (len(loaded_vectors) + len(loading_failures)),
        "application_success_rate": len(application_results) / len(loaded_vectors) if loaded_vectors else 0,
        "orchestrator_success_rate": sum(1 for r in orchestrator_results if r["success"]) / len(orchestrator_results) if orchestrator_results else 0,
        "loading_failures": loading_failures,
        "application_failures": application_failures,
        "orchestrator_failures": [r for r in orchestrator_results if not r["success"]],
        "successful_vectors": application_results,
        "physics_type_coverage": len(set(r["physics"] for r in application_results))
    }
    
    # Log detailed report
    logger.log_step(report)
    
    # Save detailed results
    import json
    with open(run_dir / "vector_compatibility_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n=== Vector Registry Compatibility Report ===")
    print(f"Total vectors tested: {report['total_vectors']}")
    print(f"Loading success rate: {report['loading_success_rate']:.1%}")
    print(f"Application success rate: {report['application_success_rate']:.1%}")
    print(f"Orchestrator success rate: {report['orchestrator_success_rate']:.1%}")
    print(f"Physics types covered: {report['physics_type_coverage']}")
    
    if loading_failures:
        print(f"\nLoading failures ({len(loading_failures)}):")
        for vector_id, error in loading_failures:
            print(f"  {vector_id}: {error}")
    
    if application_failures:
        print(f"\nApplication failures ({len(application_failures)}):")
        for failure in application_failures:
            print(f"  {failure['vector_id']}: {failure['error']}")
    
    orchestrator_failures = [r for r in orchestrator_results if not r["success"]]
    if orchestrator_failures:
        print(f"\nOrchestrator failures ({len(orchestrator_failures)}):")
        for failure in orchestrator_failures:
            print(f"  {failure['physics_type']}: {failure['error']}")
    
    print(f"\nDetailed report saved to: {run_dir}")
    
    return report

if __name__ == "__main__":
    report = generate_vector_compatibility_report()
    
    # Final summary
    if report["loading_success_rate"] == 1.0 and report["application_success_rate"] == 1.0:
        print("\nSUCCESS: All vectors are fully compatible and working!")
    else:
        print(f"\nPARTIAL SUCCESS: Some vectors need attention")
    
    print("Vector registry testing complete.")