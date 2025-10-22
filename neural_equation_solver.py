"""
Neural Equation Solver - Physics-Informed Neural Networks (PINNs)
================================================================

This module implements a Physics-Informed Neural Network to solve 
ordinary differential equations (ODEs) numerically.

Example equation: dy/dx + y = e^(-x)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
import os

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving ODEs.
    
    The network learns to approximate the solution y(x) of an ODE
    by incorporating the physics (differential equation) as a loss term.
    """
    
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1, num_layers=4):
        super(PINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build the neural network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)
    
    def derivative(self, x, order=1):
        """
        Compute derivatives of the network output with respect to input x.
        Uses automatic differentiation to compute exact derivatives.
        """
        x.requires_grad_(True)
        y = self.forward(x)
        
        if order == 1:
            dydx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), 
                                     create_graph=True, retain_graph=True)[0]
            return y, dydx
        else:
            # For higher order derivatives
            y, dydx = self.derivative(x, order-1)
            d2ydx2 = torch.autograd.grad(dydx, x, grad_outputs=torch.ones_like(dydx), 
                                       create_graph=True, retain_graph=True)[0]
            return y, dydx, d2ydx2


class ODESolver:
    """
    Solver for the ODE: dy/dx + y = e^(-x)
    """
    
    def __init__(self, pinn, device='cpu'):
        self.pinn = pinn.to(device)
        self.device = device
        
    def physics_loss(self, x):
        """
        Compute the physics loss for the ODE: dy/dx + y = e^(-x)
        The loss is |dy/dx + y - e^(-x)|^2
        """
        y, dydx = self.pinn.derivative(x)
        
        # Right-hand side of the ODE
        rhs = torch.exp(-x)
        
        # Physics residual: dy/dx + y - e^(-x)
        residual = dydx + y - rhs
        
        return torch.mean(residual**2)
    
    def boundary_loss(self, x_bc, y_bc):
        """
        Compute boundary condition loss.
        For this example, we'll use initial condition y(0) = 1
        """
        y_pred = self.pinn(x_bc)
        return torch.mean((y_pred - y_bc)**2)
    
    def train(self, epochs=30000, lr=0.0001, x_domain=(0, 2), 
              n_collocation=1000, n_boundary=10, save_plots=True):
        """
        Train the PINN to solve the ODE.
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            x_domain: Domain for collocation points (start, end)
            n_collocation: Number of collocation points for physics loss
            n_boundary: Number of boundary points
            save_plots: Whether to save training plots
        """
        
        # Set up optimizer
        optimizer = optim.Adam(self.pinn.parameters(), lr=lr)
        
        # Generate training data
        x_collocation = torch.linspace(x_domain[0], x_domain[1], n_collocation, 
                                     device=self.device).reshape(-1, 1)
        
        # Boundary condition: y(0) = 1 (initial condition)
        x_bc = torch.tensor([[0.0]], device=self.device)
        y_bc = torch.tensor([[1.0]], device=self.device)
        
        # Training history
        losses = []
        physics_losses = []
        boundary_losses = []
        
        print("Starting PINN training...")
        print(f"Device: {self.device}")
        print(f"Collocation points: {n_collocation}")
        print(f"Boundary points: {n_boundary}")
        print("-" * 50)
        
        for epoch in tqdm(range(epochs), desc="Training PINN"):
            optimizer.zero_grad()
            
            # Physics loss
            p_loss = self.physics_loss(x_collocation)
            
            # Boundary loss
            b_loss = self.boundary_loss(x_bc, y_bc)
            
            # Total loss
            total_loss = p_loss + b_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Store losses
            losses.append(total_loss.item())
            physics_losses.append(p_loss.item())
            boundary_losses.append(b_loss.item())
            
            # Print progress
            if epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Total Loss: {total_loss.item():.6f} | "
                      f"Physics Loss: {p_loss.item():.6f} | Boundary Loss: {b_loss.item():.6f}")
        
        print("\nTraining completed!")
        
        # Plot training progress
        if save_plots:
            self.plot_training_progress(losses, physics_losses, boundary_losses)
        
        return losses, physics_losses, boundary_losses
    
    def plot_training_progress(self, losses, physics_losses, boundary_losses):
        """Plot the training progress."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.semilogy(losses)
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.semilogy(physics_losses)
        plt.title('Physics Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.semilogy(boundary_losses)
        plt.title('Boundary Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def solve(self, x_test):
        """Solve the ODE at given points."""
        with torch.no_grad():
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=self.device).reshape(-1, 1)
            y_pred = self.pinn(x_test_tensor)
            return y_pred.cpu().numpy().flatten()
    
    def plot_solution(self, x_domain=(0, 2), n_points=1000, save_plot=True):
        """Plot the neural network solution vs analytical solution."""
        
        # Generate test points
        x_test = np.linspace(x_domain[0], x_domain[1], n_points)
        
        # Neural network solution
        y_nn = self.solve(x_test)
        
        # Analytical solution: y(x) = e^(-x) + x*e^(-x)
        # For dy/dx + y = e^(-x) with y(0) = 1
        y_analytical = np.exp(-x_test) + x_test * np.exp(-x_test)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(x_test, y_analytical, 'b-', label='Analytical Solution', linewidth=2)
        plt.plot(x_test, y_nn, 'r--', label='Neural Network Solution', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.title('PINN Solution vs Analytical Solution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate and display error
        error = np.abs(y_nn - y_analytical)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        plt.text(0.05, 0.95, f'Max Error: {max_error:.2e}\nMean Error: {mean_error:.2e}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if save_plot:
            plt.savefig('pinn_solution.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return x_test, y_nn, y_analytical, error


def main():
    """Main function to run the neural equation solver."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create PINN
    pinn = PINN(input_dim=1, hidden_dim=50, output_dim=1, num_layers=4)
    print(f"PINN created with {sum(p.numel() for p in pinn.parameters())} parameters")
    
    # Create solver
    solver = ODESolver(pinn, device)
    
    # Train the network
    print("\n" + "="*60)
    print("🧮 NEURAL EQUATION SOLVER - PINN Training")
    print("="*60)
    
    losses, p_losses, b_losses = solver.train(
        epochs=10000,
        lr=0.001,
        x_domain=(0, 2),
        n_collocation=1000,
        n_boundary=10
    )
    
    # Plot solution
    print("\n" + "="*60)
    print("📊 SOLUTION VISUALIZATION")
    print("="*60)
    
    x_test, y_nn, y_analytical, error = solver.plot_solution(
        x_domain=(0, 2),
        n_points=1000
    )
    
    print(f"\nSolution Statistics:")
    print(f"Max Error: {np.max(error):.2e}")
    print(f"Mean Error: {np.mean(error):.2e}")
    print(f"RMSE: {np.sqrt(np.mean(error**2)):.2e}")
    
    return solver, x_test, y_nn, y_analytical, error


if __name__ == "__main__":
    solver, x_test, y_nn, y_analytical, error = main()
