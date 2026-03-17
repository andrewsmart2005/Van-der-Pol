# Neural ODE
# Key idea: instead of learning state → next state (discrete steps like ResNet), a Neural ODE learns the
# vector field f(state) and then uses a real ODE solver to integrate it:                               
                                                                                                        
# d/dt [x, v] = NN(x, v)

# You use torchdiffeq to solve this, which means the solver handles the time integration and you just
# train the NN to approximate the right-hand side of the ODE.

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt

# van der Pol oscillator dynamics
def van_der_pol(t, state, mu):
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return np.array([dxdt, dvdt])

# Training data generation using RK4 solver
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

# Neural ODE model
# The NN takes in the current state and outputs the derivatives (dx/dt, dv/dt)
class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )

    def forward(self, t, state):
        return self.net(state)
    
# Parameters
t_start = 0.0
t_end = 20.0
dt = 0.05
initial_state = np.array([2.0, 0.0])
mu = 1.0

# Generate training data
t_vals, states = rk4(van_der_pol, initial_state, t_start, t_end, dt, mu)
# Convert to PyTorch tensors
t_tensor = torch.tensor(t_vals, dtype=torch.float32)
states_tensor = torch.tensor(states, dtype=torch.float32)
t_tensor_train = t_tensor[::10]  # Use every 10th point for training
states_tensor_train = states_tensor[::10]

# Train the Neural ODE
model = NeuralODE()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

# Training loop
iterations = 2000
for i in range(iterations):
    optimizer.zero_grad()
    pred_states = odeint(model, states_tensor_train[0], t_tensor_train)
    loss = loss_fn(pred_states, states_tensor_train)
    loss.backward()
    optimizer.step()
    scheduler.step() # Decay learning rate every 500 iters, keep?

    if (i+1) % 100 == 0:
        print(f"Iteration {i+1}/{iterations}, Loss: {loss.item():.6f}")


# eval
model.eval()
with torch.no_grad():
    pred_states = odeint(model, states_tensor[0], t_tensor).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(t_vals, states[:, 0], label='True x')
    axes[0].plot(t_vals, pred_states[:, 0], label='Predicted x', linestyle='dashed')
    axes[0].set_title('Neural ODE: True vs Predicted x(t)')
    axes[0].set_xlabel('Time (t)')
    axes[0].set_ylabel('x(t)')
    axes[0].legend()
    axes[1].plot(states[:, 0], states[:, 1], label='True Phase Space')
    axes[1].plot(pred_states[:, 0], pred_states[:, 1], label='Predicted Phase Space', linestyle='dashed')
    axes[1].set_title('Neural ODE: True vs Predicted Phase Space')
    axes[1].set_xlabel('x(t)')
    axes[1].set_ylabel('v(t)')
    axes[1].legend()
    plt.savefig('neural_ode_prediction.png')
    plt.show()

    torch.save(model.state_dict(), 'neural_ode.pth')