# Physics-Informed Neural Network (PINN) for solving ODEs
# This code defines a simple PINN to solve the van der Pol oscillator ODE.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# van der Pol oscillator dynamics
def van_der_pol(t, state, mu):
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return np.array([dxdt, dvdt])

# RK4 step function
def rk4_step(func, t, state, dt, mu):
    k1 = func(t, state, mu)
    k2 = func(t + dt/2, state + dt/2 * k1, mu)
    k3 = func(t + dt/2, state + dt/2 * k2, mu)
    k4 = func(t + dt, state + dt * k3, mu)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# Generate training data using RK4
def rk4(func, initial_state, t_start, t_end, dt, mu):
    t_vals = np.arange(t_start, t_end, dt)
    states = np.zeros((len(t_vals), len(initial_state)))
    states[0] = initial_state
    for i in range(1, len(t_vals)):
        states[i] = rk4_step(func, t_vals[i-1], states[i-1], dt, mu)
    return t_vals, states

# Sample collocation points in time
t_start = 0.0
t_end = 20.0
num_points = 1000
t_collocation = np.linspace(t_start, t_end, num_points)

# PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t):
        return self.net(t)
    
# Define physics loss based on the ODE residuals
def physics_loss(model, t, mu):
    t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1).requires_grad_(True)  # Shape (N, 1)
    pred = model(t_tensor)  # Shape (N, 2)
    x_pred = pred[:, 0]
    v_pred = pred[:, 1]

    # Compute derivatives using autograd
    dxdt_pred = torch.autograd.grad(x_pred.sum(), t_tensor, create_graph=True)[0]
    dvdt_pred = torch.autograd.grad(v_pred.sum(), t_tensor, create_graph=True)[0]

    # Compute ODE residuals
    dxdt_residual = dxdt_pred - v_pred
    dvdt_residual = dvdt_pred - (mu * (1 - x_pred**2) * v_pred - x_pred)

    return (dxdt_residual**2 + dvdt_residual**2).mean() # Should both be close to zero if ODE is satisfied

# Define the data loss
def data_loss(model, t, states):
    t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1)  # Shape (N, 1)
    states_tensor = torch.tensor(states, dtype=torch.float32)  # Shape (N, 2)
    pred = model(t_tensor)  # Shape (N, 2)
    return nn.MSELoss()(pred, states_tensor)

def ic_loss(model, initial_state):
    t0_tensor = torch.tensor([[0.0]], dtype=torch.float32)  # Initial time
    pred_initial = model(t0_tensor)  # Predicted state at t=0
    initial_state_tensor = torch.tensor(initial_state, dtype=torch.float32)  # Actual initial state
    return nn.MSELoss()(pred_initial, initial_state_tensor)

# Train the PINN
mu = 1.0
initial_state = np.array([2.0, 0.0])
t_vals, states = rk4(van_der_pol, initial_state, t_start, t_end, 0.01, mu)
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

iterations = 10000
for iter in range(iterations):
    optimizer.zero_grad()
    loss_physics = physics_loss(model, t_collocation, mu)
    loss_data = data_loss(model, t_vals, states)
    loss = 0.1 * loss_physics + loss_data + ic_loss(model, initial_state)  # weighting physics loss less to allow data to guide training
    loss.backward()
    optimizer.step()
    if iter % 100 == 0:
        print(f"Iteration {iter}, Loss: {loss.item():.6f}")

# Evaluate the trained model
model.eval()
with torch.no_grad():
    #t_test = np.linspace(t_start, t_end, 200)
    t_test_tensor = torch.tensor(t_vals, dtype=torch.float32).unsqueeze(1)
    pred_states = model(t_test_tensor).numpy()

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_vals, states[:, 0], label='True x')
plt.plot(t_vals, pred_states[:, 0], label='PINN Predicted x', linestyle='dashed')
plt.title('PINN: True vs Predicted x(t)')
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(states[:, 0], states[:, 1], label='True Phase Space')
plt.plot(pred_states[:, 0], pred_states[:, 1], label='PINN Predicted Phase Space', linestyle='dashed')
plt.title('PINN: True vs Predicted Phase Space')
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.legend()
plt.tight_layout()
plt.savefig('../figures/pinn_van_der_pol.png')
plt.show()

torch.save(model.state_dict(), '../weights/pinn_model.pth')

mse = np.mean((pred_states - states)**2)
print(f"MSE: {mse:.6f}")