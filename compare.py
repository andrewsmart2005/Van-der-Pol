# Comparison of all methods for the Van der Pol oscillator
# Loads trained models and compares: Baseline NN, ResNet, Neural ODE, PINN

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# ── Shared utilities ──────────────────────────────────────────────────────────

def van_der_pol(t, state, mu):
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return np.array([dxdt, dvdt])

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

# ── Model definitions ─────────────────────────────────────────────────────────

class VanDerPolNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return x + self.net(x)

class NeuralODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 2)
        )
    def forward(self, t, state):
        return self.net(state)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    def forward(self, t):
        return self.net(t)

# ── Parameters & reference data ───────────────────────────────────────────────

t_start, t_end, dt, mu = 0.0, 20.0, 0.01, 1.0
initial_state = np.array([2.0, 0.0])
t_vals, states_true = rk4(van_der_pol, initial_state, t_start, t_end, dt, mu)

t_min, t_max = t_vals.min(), t_vals.max()
t_norm = (t_vals - t_min) / (t_max - t_min)
mean = states_true.mean(axis=0)
std = states_true.std(axis=0)

t_tensor = torch.tensor(t_vals, dtype=torch.float32)
t_norm_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1)

# ── Load models & generate predictions ───────────────────────────────────────

# Baseline NN: t → [x, v]
baseline = VanDerPolNN()
baseline.load_state_dict(torch.load('baseline_nn.pth', weights_only=True))
baseline.eval()
with torch.no_grad():
    pred_baseline = baseline(t_norm_tensor).numpy()

# ResNet: rollout from initial state
resnet = ResNet()
resnet.load_state_dict(torch.load('resnet.pth', weights_only=True))
resnet.eval()
state = torch.tensor((initial_state - mean) / std, dtype=torch.float32)
rollout = [state.numpy()]
with torch.no_grad():
    for _ in range(len(t_vals) - 1):
        state = resnet(state)
        rollout.append(state.numpy())
pred_resnet = np.array(rollout) * std + mean

# Neural ODE: odeint from initial state
node = NeuralODE()
node.load_state_dict(torch.load('neural_ode.pth', weights_only=True))
node.eval()
with torch.no_grad():
    pred_node = odeint(node, torch.tensor(initial_state, dtype=torch.float32), t_tensor).numpy()

# PINN: t → [x, v]
pinn = PINN()
pinn.load_state_dict(torch.load('pinn_model.pth', weights_only=True))
pinn.eval()
with torch.no_grad():
    pred_pinn = pinn(torch.tensor(t_vals, dtype=torch.float32).unsqueeze(1)).numpy()

# ── MSE error table ───────────────────────────────────────────────────────────

methods = {
    'Baseline NN': pred_baseline,
    'ResNet':      pred_resnet,
    'Neural ODE':  pred_node,
    'PINN':        pred_pinn,
}

print(f"\n{'Method':<15} | {'MSE':>12}")
print("-" * 30)
for name, pred in methods.items():
    mse = np.mean((pred - states_true) ** 2)
    print(f"{name:<15} | {mse:>12.6f}")

# ── Phase portrait comparison ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(states_true[:, 0], states_true[:, 1], 'k-', linewidth=2, label='True (RK4)')
for name, pred in methods.items():
    axes[0].plot(pred[:, 0], pred[:, 1], '--', label=name)
axes[0].set_title('Phase Portrait: All Methods')
axes[0].set_xlabel('x')
axes[0].set_ylabel('v')
axes[0].legend()
axes[0].grid()

axes[1].plot(t_vals, states_true[:, 0], 'k-', linewidth=2, label='True (RK4)')
for name, pred in methods.items():
    axes[1].plot(t_vals, pred[:, 0], '--', label=name)
axes[1].set_title('x(t): All Methods')
axes[1].set_xlabel('Time (t)')
axes[1].set_ylabel('x(t)')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.savefig('comparison.png')
plt.show()
