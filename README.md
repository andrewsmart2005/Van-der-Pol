# Van der Pol Oscillator

The Van der Pol oscillator is a nonlinear system with nonlinear damping, defined by this second-order equation:

$$\ddot{x} - \mu(1-x^2)\dot{x} + x = 0$$

where $\mu$ controls the strength of the nonlinear damping. This project analyzes the system through mathematical analysis, numerical solvers, and four machine learning approaches.

## Repository Structure

Van-der-Pol
├── MATH.md 
├── README.md
├── compare.py
├── figures
│   ├── comparison.png
│   ├── convergence_order.png
│   ├── energy_drift.png
│   ├── neural_ode_prediction.png
│   ├── pinn_van_der_pol.png
│   ├── resnet_van_der_pol.png
│   ├── solver_phase_comparison.png
│   ├── van_der_pol_comparison.png
│   ├── van_der_pol_initial_conditions.png
│   ├── van_der_pol_mu_analysis.png
│   ├── van_der_pol_nn_comparison.png
│   └── van_der_pol_vector_field.png
├── math
│   └── math_analysis.py
├── models 
│   ├── PINN.py
│   ├── ResNet.py
│   ├── baseline_nn.py
│   └── neural_ode.py
├── numerical
│   └── numerical_solvers.py
└── weights
    ├── baseline_nn.pth
    ├── neural_ode.pth
    ├── pinn_model.pth
    └── resnet.pth

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- SciPy
- torchdiffeq

Install dependencies:
```bash
pip install torch numpy matplotlib scipy torchdiffeq
```

## How to Run

Run scripts:

1. **Mathematical analysis**
   ```bash
   python math_analysis.py
   ```

2. **Numerical solvers**
   ```bash
   python numerical_solvers.py
   ```

3. **Train ML models**
   ```bash
   python models/baseline_nn.py
   python models/ResNet.py
   python models/neural_ode.py
   python models/PINN.py
   ```

## Results Summary
MSE:
Baseline NN: 0.000098
ResNet: 0.036578
Neural ODE: 0.108601
PINN: 0.205964 

The Baseline NN achieved the lowest MSE but memorizes the training trajectory rather than learning the dynamics. The ResNet produced the most accurate phase portrait and learned a physically meaningful state-to-state mapping. The Neural ODE and PINN struggled with long-horizon phase accuracy, a known challenge for these architectures on oscillatory systems.

Generated plots in `figures/`

