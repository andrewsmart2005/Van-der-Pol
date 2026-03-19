# ResNet 
# one step update, residual connection
# h_next = h + f(h)
from zipfile import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

def rk4(func, initial_state, t_start, t_end, dt, mu):
    t_vals = np.arange(t_start, t_end, dt)
    states = np.zeros((len(t_vals), len(initial_state)))
    states[0] = initial_state
    for i in range(1, len(t_vals)):
        states[i] = rk4_step(func, t_vals[i-1], states[i-1], dt, mu)
    return t_vals, states

# normalize state data
def normalize(states):
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    return (states - mean) / std, mean, std

# model
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, h):
        return h + self.layers(h)  # Residual connection

# Parameters
t_start = 0.0
t_end = 20.0
dt = 0.01
initial_state = np.array([2.0, 0.0])
mu = 1.0

# Train
t_vals, states = rk4(van_der_pol, initial_state, t_start, t_end, dt, mu)
states_norm, mean, std = normalize(states)


x_in = torch.tensor(states_norm[:-1], dtype=torch.float32) # curr state, input
y_out = torch.tensor(states_norm[1:], dtype=torch.float32) # next state target

# Build and train model
model = ResNet()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

# Training loop
iterations = 10000
for iter in range(iterations):
    #model.train()
    optimizer.zero_grad()
    pred = model(x_in)
    loss = loss_fn(pred, y_out)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients
    optimizer.step()

    if (iter + 1) % 100 == 0:
        print(f'Iteration {iter+1}/{iterations}, Loss: {loss.item():.6f}')

# rollout from init
state = torch.tensor((initial_state - mean) / std, dtype=torch.float32)  # normalize initial state
pred_states = [state.numpy()]

# Iteratively apply the model to predict trajectory
model.eval()
with torch.no_grad():
    for _ in range(len(t_vals) - 1):
        state = model(state)
        pred_states.append(state.numpy())

pred_states = np.array(pred_states)

pred_states = pred_states * std + mean  # Denormalize predictions

# Plot true vs predicted trajectories
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_vals[:-1], states[:-1, 0], label='True x(t)', color='black')
plt.plot(t_vals[:-1], pred_states[:-1, 0], label='Predicted x(t)', linestyle='--')
plt.title('Van der Pol Oscillator: True vs Predicted x(t)')
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)

plt.plot(states[:-1, 0], states[:-1, 1], label='True Trajectory', color='black')
plt.plot(pred_states[:-1, 0], pred_states[:-1, 1], label='Predicted Trajectory', linestyle='--')
plt.title('Van der Pol Oscillator: Phase Space: True vs Predicted')
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(FIGURES_DIR / 'resnet_van_der_pol.png')
plt.show()
torch.save(model.state_dict(), WEIGHTS_DIR / 'resnet.pth')

mse = np.mean((pred_states - states) ** 2)
print(f"MSE: {mse:.6f}")

# testing on different initial conditions
# nvm

mse_test = np.mean((pred_states - states) ** 2)
print(f"Test MSE: {mse_test:.6f}")
