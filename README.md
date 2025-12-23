# Numerical Verification of Energy Inequality for Navier-Stokes Equations

## Overview

This project provides a numerical implementation for verifying the energy inequality in the incompressible Navier-Stokes equations. The implementation uses a stable projection method with semi-implicit time stepping and includes several benchmark flow configurations for testing. The work is based on the theoretical foundations established by Olga Ladyzhenskaya for the mathematical analysis of Navier-Stokes equations.

## Mathematical Background

### Incompressible Navier-Stokes Equations
The 2D incompressible Navier-Stokes equations are given by:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}$$

$$\nabla \cdot \mathbf{u} = 0$$

where:
- $\mathbf{u} = (u, v)$ is the velocity field
- $p$ is the pressure
- $\nu$ is the kinematic viscosity
- $\mathbf{f}$ is the external force

### Energy Inequality (Ladyzhenskaya, 1969)
For smooth solutions, the energy satisfies the inequality:

$$\frac{d}{dt} \int_{\Omega} \frac{1}{2}|\mathbf{u}|^2 d\mathbf{x} + \nu \int_{\Omega} |\nabla \mathbf{u}|^2 d\mathbf{x} \leq \int_{\Omega} \mathbf{f} \cdot \mathbf{u} d\mathbf{x}$$

This project numerically verifies this inequality using various flow configurations. The inequality represents a fundamental property of viscous flows, showing that the rate of change of kinetic energy plus the viscous dissipation is bounded by the work done by external forces.

## Theoretical Foundation

This implementation is based on Ladyzhenskaya's contributions to:
1. **Existence and uniqueness theorems** for Navier-Stokes equations
2. **Energy estimates** and a priori bounds for solutions
3. **Regularity theory** for weak and strong solutions
4. **Numerical analysis** foundations for finite difference methods

Key references:
- Ladyzhenskaya, O. A. (1969). *The Mathematical Theory of Viscous Incompressible Flow*
- Ladyzhenskaya, O. A. (1963). "Solution of the first boundary value problem for quasilinear parabolic equations"
- Ladyzhenskaya, O. A. (1975). "On the solvability of the boundary value problems for the Navier-Stokes equations in regions with non-smooth boundaries"

## Implementation Details

### Numerical Method
- **Projection method** (Chorin-type) for pressure-velocity coupling
- **Semi-implicit scheme** for time integration
- **Upwind differencing** for convection terms (stability enhancement)
- **Central differencing** for diffusion terms
- **Iterative solver** for pressure Poisson equation

### Stability Features
- Automatic time step selection based on CFL and viscosity conditions
- Boundary condition implementation for both Dirichlet and periodic cases
- Divergence-free velocity field enforcement
- Numerical stability checks at initialization
- Ladyzhenskaya-inspired energy estimates for validation

## Code Structure

### Main Class: `StableNavierStokesSolver`
The solver class provides all functionality for simulating 2D Navier-Stokes flows:

#### Key Methods:
- `__init__()`: Initialize solver with domain parameters
- `set_initial_condition()`: Specify initial velocity field
- `set_force()`: Define external force field
- `time_step()`: Perform single time step using projection method
- `run_simulation()`: Execute complete simulation
- `analyze_results()`: Check energy inequality satisfaction
- `plot_results()`: Visualize simulation results

#### Computational Methods:
- `compute_convection()`: Upwind scheme for nonlinear terms
- `compute_viscous_term()`: Central differencing for diffusion
- `solve_pressure_poisson()`: Iterative solver for pressure
- `apply_boundary_conditions()`: Handle boundary conditions
- `compute_energy()`: Calculate kinetic energy
- `compute_enstrophy()`: Compute dissipation measure
- `compute_divergence()`: Check divergence-free condition

## Numerical Experiments

The project includes four benchmark experiments that test different aspects of the Navier-Stokes solutions:

### 1. Taylor-Green Vortex
- **Domain**: $[0, 2\pi] \times [0, 2\pi]$
- **Boundary conditions**: Periodic
- **Initial condition**: Analytical vortex solution
- **Purpose**: Test periodic boundaries and exact solution decay
- **Ladyzhenskaya connection**: Demonstrates regularity of periodic solutions

### 2. Decaying Gaussian Vortex
- **Domain**: $[0, 2] \times [0, 2]$
- **Boundary conditions**: Dirichlet (no-slip walls)
- **Initial condition**: Gaussian velocity profile
- **Purpose**: Test viscous decay and boundary interactions
- **Ladyzhenskaya connection**: Illustrates boundary layer effects

### 3. Two Interacting Vortices
- **Domain**: $[0, 2] \times [0, 2]$
- **Boundary conditions**: Dirichlet
- **Initial condition**: Two counter-rotating vortices
- **Purpose**: Study vortex interactions and nonlinear effects
- **Ladyzhenskaya connection**: Tests nonlinear stability estimates

### 4. Forced Flow
- **Domain**: $[0, 1] \times [0, 1]$
- **Boundary conditions**: Dirichlet
- **Initial condition**: Zero velocity
- **Force**: Time-periodic external forcing
- **Purpose**: Test energy input from external forces
- **Ladyzhenskaya connection**: Validates energy inequality with forcing

## Installation and Requirements

### Dependencies
```bash
numpy>=1.19.0
matplotlib>=3.3.0
scipy>=1.5.0
```

### Installation
```bash
# Clone repository
git clone https://github.com/karinaurazova/Numerical_Verification_of_Energy_Inequality_for_Navier-Stokes_Equations.git
cd Numerical_Verification_of_Energy_Inequality_for_Navier-Stokes_Equations

# Install dependencies
pip install -r requirements.txt
# OR
pip install numpy matplotlib scipy
```

## Usage

### Running Experiments
```python
# Import the solver
from Numerical_Verification_of_Energy_Inequality_for_Navier_Stokes_Equations import StableNavierStokesSolver

# Choose experiment
experiment = 2  # 1, 2, 3, or 4

if experiment == 1:
    from experiments import experiment_taylor_green
    solver = experiment_taylor_green()
elif experiment == 2:
    from experiments import experiment_gaussian_vortex
    solver = experiment_gaussian_vortex()
# ... etc.
```

### Creating Custom Experiments
```python
def custom_experiment():
    # Define domain and parameters
    Lx, Ly = 1.0, 1.0
    nx, ny = 64, 64
    nu = 0.01
    dt = 0.001
    T = 1.0
    
    # Create solver
    solver = StableNavierStokesSolver(Lx, Ly, nx, ny, nu, dt, T)
    
    # Define initial conditions
    def u0_func(x, y):
        return np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
    
    def v0_func(x, y):
        return -np.cos(2*np.pi*x) * np.sin(2*np.pi*y)
    
    solver.set_initial_condition(u0_func, v0_func)
    
    # Run simulation
    solver.run_simulation()
    solver.plot_results()
    
    return solver
```

## Output and Visualization

The solver provides comprehensive output including:

### Numerical Output
- Time step information
- Energy evolution
- Dissipation rates
- Divergence measurements
- Inequality violation counts
- Ladyzhenskaya-style energy estimates

### Visualizations
1. **Energy inequality plot**: LHS vs RHS of energy inequality
2. **Inequality margin**: Difference between LHS and RHS
3. **Kinetic energy evolution**: Time history of total energy
4. **Cumulative dissipation**: Time-integrated dissipation
5. **Velocity divergence**: Maximum divergence over time
6. **Final velocity field**: Color map of velocity magnitude

### Example Analysis Output
```
RESULTS ANALYSIS
----------------------------------------------------------------------
Total steps: 200
Inequality violations: 0 (0.0%)
Maximum divergence: 2.34e-06
Average divergence: 1.12e-06
Energy inequality holds (Ladyzhenskaya condition satisfied).
```

## Stability Considerations

The implementation includes automatic stability checks inspired by Ladyzhenskaya's energy methods:

### CFL Condition
$$\Delta t \leq \frac{\text{CFL} \cdot \min(\Delta x, \Delta y)}{\max(|\mathbf{u}|)}$$

### Viscosity Condition (Ladyzhenskaya-type)
$$\Delta t \leq \frac{1}{4} \frac{\min(\Delta x^2, \Delta y^2)}{\nu}$$

The solver warns users if the chosen time step violates these conditions, which are derived from Ladyzhenskaya's stability analysis.

## Performance Notes

- The implementation uses explicit time stepping for simplicity
- Pressure Poisson equation solved with iterative relaxation
- Computation scales with $O(N_x N_y)$ per time step
- Memory usage scales with grid size
- Energy estimates computed at each step for validation

## Extending the Code

### Adding New Boundary Conditions
Override the `apply_boundary_conditions()` method to implement custom boundary conditions following Ladyzhenskaya's analysis of different boundary types.

### Implementing Different Time Stepping
Modify the `time_step()` method to implement:
- Fully implicit schemes
- Runge-Kutta methods
- Adaptive time stepping based on energy estimates

### Adding New Diagnostics
Extend the `analyze_results()` method to compute additional quantities of interest, such as higher-order norms and Sobolev space estimates used in Ladyzhenskaya's theory.

## References

### Primary Theoretical Works
1. Ladyzhenskaya, O. A. (1969). *The Mathematical Theory of Viscous Incompressible Flow* (2nd ed.). Gordon and Breach.
2. Ladyzhenskaya, O. A. (1963). "Solution of the first boundary value problem for quasilinear parabolic equations". *Trudy Matematicheskogo Instituta Imeni V. A. Steklova*, 92, 115-158.
3. Ladyzhenskaya, O. A. (1975). "On the solvability of the boundary value problems for the Navier-Stokes equations in regions with non-smooth boundaries". *Zap. Nauchn. Sem. LOMI*, 59, 81-97.

### Numerical Methods
4. Chorin, A. J. (1968). "Numerical Solution of the Navier-Stokes Equations". *Mathematics of Computation*, 22(104), 745-762.
5. Temam, R. (2001). *Navier-Stokes Equations: Theory and Numerical Analysis* (3rd ed.). AMS Chelsea Publishing.
6. Guermond, J. L., Minev, P., & Shen, J. (2006). "An Overview of Projection Methods for Incompressible Flows". *Computer Methods in Applied Mechanics and Engineering*, 195(44-47), 6011-6045.

## License

This project is available for academic and research use. Please cite if used in publications.

## Acknowledgments

This implementation builds upon the fundamental mathematical work of Olga Ladyzhenskaya, whose contributions to the theory of Navier-Stokes equations provide the theoretical foundation for numerical verification of energy inequalities.

## Contact

For questions or contributions, please contact karina_urazova@icloud.com.

---

*This implementation provides a robust framework for studying energy transfer mechanisms in viscous incompressible flows and serves as an educational tool for computational fluid dynamics, grounded in Ladyzhenskaya's rigorous mathematical analysis.*
