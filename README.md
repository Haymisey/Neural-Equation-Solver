# Neural Equation Solver - Physics-Informed Neural Networks (PINNs)

A cutting-edge implementation of Physics-Informed Neural Networks to solve ordinary differential equations (ODEs) numerically. This project demonstrates how deep learning can be used to solve differential equations without explicit integration.

## What We're Solving

**ODE:** `dy/dx + y = e^(-x)` with initial condition `y(0) = 1`

**Analytical Solution:** `y(x) = e^(-x) + x*e^(-x)`

## Key Features

- **Physics-Informed Neural Networks (PINNs)**: Incorporates the differential equation directly into the loss function
- **Automatic Differentiation**: Uses PyTorch's autograd for exact derivative computation
- **Real-time Training Visualization**: Monitor training progress with live plots
- **Solution Validation**: Compare neural network solutions with analytical results
- **Extensible Architecture**: Easy to modify for different ODEs and PDEs

## Installation

```bash
# Clone or download the project
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run the Complete Solver
```python
python neural_equation_solver.py
```

### Interactive Jupyter Notebook
```bash
jupyter notebook neural_equation_solver.ipynb
```

## What You'll See

1. **Training Progress**: Real-time loss curves showing physics loss and boundary loss
2. **Solution Comparison**: Neural network solution vs analytical solution
3. **Error Analysis**: Quantitative metrics (max error, mean error, RMSE)
4. **Visualization**: High-quality plots saved automatically

## How PINNs Work

### Traditional Neural Networks
- Learn from data: `(x, y)` pairs
- Minimize data loss: `|y_pred - y_true|²`

### Physics-Informed Neural Networks
- Learn from physics: the differential equation itself
- Minimize physics loss: `|dy/dx + y - e^(-x)|²`
- Incorporate boundary conditions as additional loss terms

### Key Advantages
- **No labeled data needed**: The physics provides the supervision
- **Generalizable**: Works for complex geometries and boundary conditions
- **Interpretable**: The network learns the actual physics
- **Efficient**: Can solve high-dimensional problems

## Technical Details

### Network Architecture
- **Input**: x (spatial coordinate)
- **Output**: y(x) (solution)
- **Hidden Layers**: 4 layers with 50 neurons each
- **Activation**: Tanh (smooth derivatives)
- **Parameters**: ~10,000 trainable parameters

### Loss Function
```
Total Loss = Physics Loss + Boundary Loss
Physics Loss = |dy/dx + y - e^(-x)|²
Boundary Loss = |y(0) - 1|²
```

### Training Strategy
- **Collocation Points**: 1000 random points in domain [0, 2]
- **Boundary Points**: Initial condition at x = 0
- **Optimizer**: Adam with learning rate 0.001
- **Epochs**: 10,000 (typically converges in 5,000-8,000)

## Expected Results

- **Training Time**: 2-5 minutes on CPU, 30 seconds on GPU
- **Accuracy**: Mean error < 1e-3, Max error < 1e-2
- **Convergence**: Physics loss typically drops to 1e-6

## Learning Outcomes

After working with this project, you'll understand:

1. **Physics-Informed Neural Networks**: How to incorporate physics into deep learning
2. **Automatic Differentiation**: Computing exact derivatives in neural networks
3. **Scientific Computing**: Solving differential equations with machine learning
4. **Loss Function Design**: Balancing multiple objectives in neural network training
5. **Validation Methods**: How to verify neural network solutions

## Next Steps

### Extensions to Try
1. **Different ODEs**: Modify the physics loss for other equations
2. **Partial Differential Equations**: Extend to 2D/3D problems
3. **Time-dependent Problems**: Add temporal dimension
4. **Complex Geometries**: Irregular domains and boundary conditions
5. **Multi-physics**: Coupled systems of equations

### Advanced Topics
- **Adaptive Collocation**: Dynamic point sampling
- **Multi-scale Networks**: Different resolutions for different scales
- **Uncertainty Quantification**: Bayesian neural networks
- **Transfer Learning**: Pre-trained networks for similar problems

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.
- Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning.

## Contributing?

Feel free to extend this project! Some ideas:
- Add support for different ODEs
- Implement PDE solvers
- Create interactive web interface
- Add uncertainty quantification
- Optimize for different hardware

## License

This project is open source and available under the MIT License.
