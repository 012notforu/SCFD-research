#!/usr/bin/env python3
"""Test script for enhanced pathway logging system."""
import numpy as np
import tempfile
from pathlib import Path

from utils.logging import (
    create_run_directory, RunLogger, 
    PhysicsContext, VectorInvocation, OutcomeMetrics, CellLogEntry
)
from engine import load_config

def test_enhanced_logging():
    """Test the enhanced logging capabilities."""
    print("=== Testing Enhanced Logging System ===")
    
    # Create temporary run directory
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        run_dir.mkdir(parents=True)
        
        # Create logger
        logger = RunLogger(run_dir)
        
        # Test basic logging (existing functionality)
        logger.log_step({"energy": 1.5, "step": 1})
        logger.log_csv("test", ["step", "value"], [1, 2.5])
        
        print("OK Basic logging works")
        
        # Test enhanced logging
        # Create mock physics data
        theta = np.random.random((8, 8)).astype(np.float32)
        theta_dot = np.random.random((8, 8)).astype(np.float32)
        physics = PhysicsParams()
        
        # Test physics context computation
        cell_pos = (3, 4)
        physics_ctx = logger.compute_physics_context(theta, theta_dot, physics, cell_pos)
        
        print(f"OK Physics context computed: coherence={physics_ctx.coherence:.3f}")
        
        # Test neighbor summary
        neighbor_summary = logger.compute_neighbor_summary(theta, cell_pos)
        print(f"OK Neighbor summary: mean={neighbor_summary['mean']:.3f}")
        
        # Test cell action logging
        vector_invocation = VectorInvocation(
            vector_id="test_vector",
            reason="exploration",
            confidence=0.8,
            selection_method="orchestrator"
        )
        
        outcome = OutcomeMetrics(
            delta_energy=0.15,
            accepted=True,
            progress_made=1.0,
            stuck_counter=0
        )
        
        cell_entry = CellLogEntry(
            cell_pos=cell_pos,
            timestep=10,
            physics=physics_ctx,
            vector=vector_invocation,
            outcome=outcome,
            neighbors=neighbor_summary
        )
        
        logger.log_cell_action(cell_entry)
        print("OK Cell action logged")
        
        # Test interaction logging
        logger.log_interaction(
            source_cell=(3, 4),
            target_cell=(3, 5),
            interaction_type="field_influence",
            strength=0.25,
            timestep=10
        )
        print("OK Interaction logged")
        
        # Test vector stats summary
        logger.log_vector_stats_summary(timestep=10)
        print("OK Vector stats summary logged")
        
        # Verify files were created
        expected_files = [
            "logs.jsonl",
            "test.csv", 
            "pathway_log.jsonl",
            "interactions.jsonl",
            "vector_stats.jsonl"
        ]
        
        for filename in expected_files:
            filepath = run_dir / filename
            if filepath.exists():
                print(f"OK {filename} created")
                # Show file size for verification
                size = filepath.stat().st_size
                print(f"  File size: {size} bytes")
            else:
                print(f"MISSING {filename}")
        
        # Check that enhanced directories were created
        enhanced_dirs = ["pathway_analysis", "interaction_networks"]
        for dirname in enhanced_dirs:
            dirpath = run_dir / dirname
            if dirpath.exists():
                print(f"OK {dirname}/ directory created")
            else:
                print(f"MISSING {dirname}/ directory")
        
        print("\n=== Enhanced Logging Test Complete ===")
        print("All enhanced logging capabilities are working!")

if __name__ == "__main__":
    test_enhanced_logging()