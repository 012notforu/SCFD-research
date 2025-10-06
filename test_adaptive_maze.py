#!/usr/bin/env python3
"""Quick test script for adaptive maze solving meta-learning."""
import numpy as np
from pathlib import Path

from benchmarks.maze_solving import MazeParams, AdaptiveMazeSolver, generate_maze
from orchestrator.pipeline import load_vector_registry

def test_adaptive_maze_solving():
    """Test the adaptive maze solving with real meta-learning."""
    print("=== Testing Adaptive SCFD Maze Solving ===")
    
    # Load vector registry
    print("Loading vector registry...")
    registry = load_vector_registry("runs")
    print(f"Found {len(registry)} trained vectors")
    
    # Show some example vectors
    print("Available vector types:")
    vector_types = {}
    for vec in registry[:10]:
        physics = vec.physics
        vector_types[physics] = vector_types.get(physics, 0) + 1
    for physics, count in vector_types.items():
        print(f"  {physics}: {count} vectors")
    
    # Generate test maze
    print("\nGenerating test maze...")
    maze_size = 16
    maze = generate_maze((maze_size, maze_size), wall_density=0.25, seed=42)
    params = MazeParams(shape=(maze_size, maze_size))
    
    print(f"Maze shape: {maze.shape}")
    print(f"Start: (1, 1), Goal: ({maze_size-2}, {maze_size-2})")
    
    # Create adaptive solver
    solver = AdaptiveMazeSolver(params, maze, registry)
    
    # Run adaptive solving
    print("\nStarting adaptive solving...")
    result = solver.solve_adaptively(max_generations=5, steps_per_generation=50)
    
    # Display results
    print(f"\n=== Results ===")
    print(f"Solved: {result['solved']}")
    print(f"Generations used: {result['generations_used']}")
    print(f"Total steps: {result['total_steps']}")
    print(f"Final path length: {result['final_path_length']}")
    print(f"Vector sequence: {result['vector_sequence']}")
    
    # Show generation details
    print(f"\n=== Generation Details ===")
    for i, gen_result in enumerate(result['generation_results']):
        perf = gen_result['performance']
        print(f"Gen {i}: {perf['vector_id'][:20]}")
        print(f"  Physics: {perf['vector_physics']}")
        print(f"  Reason: {perf['selection_reason']}")
        print(f"  Progress: {perf['progress_made']} new cells, {perf['distance_progress']} distance")
        print(f"  Steps used: {perf['steps_used']}")
    
    # Save visualization
    if result['generations_used'] > 0:
        print(f"\nSaving visualization...")
        viz_dir = Path("adaptive_maze_test")
        viz_files = solver.generate_visualization(viz_dir, result)
        print(f"Visualization saved: {viz_files['visualization']}")
        print(f"History saved: {viz_files['history']}")
    
    # Success analysis
    if result['solved']:
        print(f"\nSUCCESS: Maze solved using adaptive meta-learning!")
        print(f"Required {result['generations_used']} different vectors")
        
        # Check if switching improved performance
        if result['generations_used'] > 1:
            print(f"Adaptive vector switching was beneficial")
        else:
            print(f"First vector was sufficient")
    else:
        print(f"\nPartial success: Explored {len(solver.visited_cells)} cells")
        print(f"Final distance to goal: {abs(solver.current_pos[0] - solver.goal_pos[0]) + abs(solver.current_pos[1] - solver.goal_pos[1])}")
    
    return result

def run_multiple_mazes():
    """Test adaptive solving on multiple mazes."""
    print("\n=== Testing Multiple Mazes ===")
    
    registry = load_vector_registry("runs")
    results = []
    
    for i in range(3):
        print(f"\n--- Maze {i+1} ---")
        
        # Generate different maze
        maze = generate_maze((12, 12), wall_density=0.3, seed=i*42)
        params = MazeParams(shape=(12, 12))
        
        solver = AdaptiveMazeSolver(params, maze, registry)
        result = solver.solve_adaptively(max_generations=3, steps_per_generation=30)
        
        results.append({
            "maze_id": i,
            "solved": result["solved"],
            "generations": result["generations_used"],
            "vectors": result["vector_sequence"],
        })
        
        print(f"Maze {i+1}: {'SOLVED' if result['solved'] else 'PARTIAL'} in {result['generations_used']} generations")
    
    # Summary
    solved_count = sum(1 for r in results if r["solved"])
    print(f"\nSummary: {solved_count}/3 mazes solved")
    
    # Vector usage analysis
    all_vectors = []
    for r in results:
        all_vectors.extend(r["vectors"])
    
    vector_counts = {}
    for vec in all_vectors:
        vector_counts[vec] = vector_counts.get(vec, 0) + 1
    
    print(f"Most used vectors:")
    for vec, count in sorted(vector_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {vec[:25]}: {count} times")

if __name__ == "__main__":
    # Run single maze test
    result = test_adaptive_maze_solving()
    
    # Run multiple maze test
    run_multiple_mazes()
    
    print(f"\n=== Test Complete ===")
    print(f"Adaptive meta-learning maze solving test finished!")
    print(f"Check 'adaptive_maze_test/' directory for visualizations.")