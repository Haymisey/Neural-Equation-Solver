#!/usr/bin/env python3
"""
Quick demo of the Neural Equation Solver
========================================

This script provides a quick demonstration of the PINN solving the ODE:
dy/dx + y = e^(-x) with y(0) = 1
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_equation_solver import PINN, ODESolver

def quick_demo():
    """Run a quick demonstration of the neural equation solver."""
    
    print("NEURAL EQUATION SOLVER - QUICK DEMO")
    print("=" * 50)
    print("Solving: dy/dx + y = e^(-x) with y(0) = 1")
    print("Analytical solution: y(x) = e^(-x) + x*e^(-x)")
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create PINN
    pinn = PINN(input_dim=1, hidden_dim=50, output_dim=1, num_layers=4)
    solver = ODESolver(pinn, device)
    
    print(f"Network parameters: {sum(p.numel() for p in pinn.parameters()):,}")
    print()
    
    # Train the network
    print("Training PINN...")
    losses, p_losses, b_losses = solver.train(
        epochs=5000,
        lr=0.001,
        x_domain=(0, 2),
        n_collocation=1000,
        n_boundary=10,
        save_plots=False
    )
    
    # Solve and compare
    print("\nEvaluating solution...")
    x_test, y_nn, y_analytical, error = solver.plot_solution(
        x_domain=(0, 2),
        n_points=1000,
        save_plot=False
    )
    
    # Print results
    print(f"\nResults:")
    print(f"   Max Error: {np.max(error):.2e}")
    print(f"   Mean Error: {np.mean(error):.2e}")
    print(f"   RMSE: {np.sqrt(np.mean(error**2)):.2e}")
    
    # Check specific points
    test_points = [0.0, 0.5, 1.0, 1.5, 2.0]
    print(f"\nSolution at key points:")
    print(f"   x    | Analytical | Neural Net | Error")
    print(f"   " + "-" * 35)
    
    for x_val in test_points:
        # Find closest point in our test array
        idx = np.argmin(np.abs(x_test - x_val))
        y_anal = y_analytical[idx]
        y_nn_val = y_nn[idx]
        err = abs(y_nn_val - y_anal)
        print(f"   {x_val:4.1f} | {y_anal:10.6f} | {y_nn_val:10.6f} | {err:8.2e}")
    
    print(f"\nDemo completed successfully!")
    print(f"   The neural network learned to solve the differential equation!")
    print(f"   Physics-Informed Neural Networks are working!")
    
    return solver, x_test, y_nn, y_analytical, error

if __name__ == "__main__":
    solver, x_test, y_nn, y_analytical, error = quick_demo()
