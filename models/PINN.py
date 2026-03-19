# PINN 
# t -> x(t), v(t) and penalty for not satsifying ODE

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "figures"
WEIGHTS_DIR = ROOT_DIR / "weights"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# van der Pol 
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

# solver
def rk4(func, initial_state, t_start, t_end, dt, mu):
    t_vals = np.arange(t_start, t_end, dt)
    states = np.zeros((len(t_vals), len(initial_state)))
    states[0] = initial_state
    for i in range(1, len(t_vals)):
        states[i] = rk4_step(func, t_vals[i-1], states[i-1], dt, mu)
    return t_vals, states

# param and collocation points
t_start = 0.0
t_end = 20.0
num_points = 1000
t_collocation = np.linspace(t_start, t_end, num_points)

# PINN model
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t):
        return self.layers(t)
    
# Define physics loss based on the ODE residuals
def physics_residual(model, t, mu):
    t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1).requires_grad_(True)  # Shape (N, 1)
    pred = model(t_tensor)  # Shape (N, 2)
    x = pred[:, 0]
    v = pred[:, 1]

    # Compute derivatives with autograd
    dxdt_pred = torch.autograd.grad(x.sum(), t_tensor, create_graph=True)[0]
    dvdt_pred = torch.autograd.grad(v.sum(), t_tensor, create_graph=True)[0]

    # Compute ODE residuals
    res_x = dxdt_pred - v
    res_v = dvdt_pred - (mu * (1 - x**2) * v - x)

    return (res_x**2 + res_v**2).mean() # Should both be close to zero if ODE is satisfied

# Define the data loss
def data_loss(model, t, states):
    t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1)  
    states_tensor = torch.tensor(states, dtype=torch.float32)  
    pred = model(t_tensor)  

    return nn.MSELoss()(pred, states_tensor)

def ic_loss(model, initial_state):
    t0 = torch.tensor([[0.0]])  # Initial time
    pred_initial = model(t0)  # pred state at t=0
    ic = torch.tensor(initial_state, dtype=torch.float32)  
    return nn.MSELoss()(pred_initial, ic)

# Train the PINN
mu = 1.0
initial_state = np.array([2.0, 0.0])
t_vals, states = rk4(van_der_pol, initial_state, t_start, t_end, 0.01, mu)
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

iterations = 10000
# 1000 col points



for iter in range(iterations):
    optimizer.zero_grad()
    
    loss_data = data_loss(model, t_vals, states)
    loss_physics = physics_residual(model, t_collocation, mu)
    loss_ic = ic_loss(model, initial_state)

    loss = 0.1 * loss_physics + loss_data + loss_ic  # weighting physics loss less 
    loss.backward()
    optimizer.step()
    
    if iter % 100 == 0:
        print(f"Iteration {iter}, Loss: {loss.item():.6f}")

# Evaluate the trained model
model.eval()
with torch.no_grad():
    #t_test = np.linspace(t_start, t_end, 200)
    t_eval = torch.tensor(t_vals, dtype=torch.float32).unsqueeze(1)
    pred_states = model(t_eval).numpy()

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_vals, states[:, 0], label='True x')
plt.plot(t_vals, pred_states[:, 0], label='PINN Predicted x', linestyle='--')
plt.title('PINN: x(t)')
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(states[:, 0], states[:, 1], label='True Phase Space')
plt.plot(pred_states[:, 0], pred_states[:, 1], label='PINN Predicted Phase Space', linestyle='--')
plt.title('PINN: Phase Space')
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pinn_van_der_pol.png')
plt.show()

torch.save(model.state_dict(), WEIGHTS_DIR / 'pinn_model.pth')

mse = np.mean((pred_states - states)**2)
print(f"MSE: {mse:.6f}")