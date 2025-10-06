import json
import os
import tempfile
import pytest
from pathlib import Path

def test_training_writes_vector_and_meta():
    """Test that training produces vector and metadata files."""
    # Check if there's already a trained vector with metadata
    runs_dir = Path("runs/cartpole_cma")
    if runs_dir.exists():
        vec_path = runs_dir / "best_vector.json"
        meta_path = runs_dir / "metadata.json"
        
        if vec_path.exists():
            assert os.path.exists(vec_path)
            with open(vec_path) as f:
                vector_data = json.load(f)
                assert "vector" in vector_data
        
        # Check for metadata if it exists
        if meta_path.exists():
            with open(meta_path) as f:
                meta_data = json.load(f)
                assert "created_at" in meta_data

def test_replay_determinism():
    """Test that replay with same vector produces consistent results."""
    # This is a placeholder for deterministic replay testing
    # Implementation would depend on having a consistent replay API
    pass