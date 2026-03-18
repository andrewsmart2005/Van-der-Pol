# ResNet for learning the Van der Pol oscillator dynamics
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


# van der Pol oscillator dynamics
def van_der_pol(t, state, mu):
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return np.array([dxdt, dvdt])

# generate training data using RK4 solver
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
def normalize_data(states):
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    return (states - mean) / std, mean, std

# ResNet model for learning the dynamics
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return x + self.net(x)  # Residual connection

# Parameters
t_start = 0.0
t_end = 20.0
dt = 0.01
initial_state = np.array([2.0, 0.0])
mu = 1.0

# Train
t_vals, states = rk4(van_der_pol, initial_state, t_start, t_end, dt, mu)
states_norm, mean, std = normalize_data(states)
X_train = states_norm[:-1]  # Current state as input
y_train = states_norm[1:]   # Next state as target

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Build and train model
model = ResNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
iterations = 10000
for iter in range(iterations):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients
    optimizer.step()
    if (iter + 1) % 100 == 0:
        print(f'Iteration {iter+1}/{iterations}, Loss: {loss.item():.6f}')

# rollout
state = torch.tensor((initial_state - mean) / std, dtype=torch.float32)  # Normalize initial state
predicted_states = [state.numpy()]

# Iteratively apply the model to predict trajectory
model.eval()
with torch.no_grad():
    for _ in range(len(t_vals) - 1):
        state = model(state)
        predicted_states.append(state.numpy())
predicted_states = np.array(predicted_states)

predicted_states = predicted_states * std + mean  # Denormalize predictions

# Plot true vs predicted trajectories
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_vals[:-1], states[:-1, 0], label='True x(t)', color='black')
plt.plot(t_vals[:-1], predicted_states[:-1, 0], label='Predicted x(t)', linestyle='--')
plt.title('Van der Pol Oscillator: True vs Predicted x(t)')
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(states[:-1, 0], states[:-1, 1], label='True Trajectory', color='black')
plt.plot(predicted_states[:-1, 0], predicted_states[:-1, 1], label='Predicted Trajectory', linestyle='--')
plt.title('Van der Pol Oscillator: Phase Space: True vs Predicted')
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('resnet_van_der_pol.png')
plt.show()
torch.save(model.state_dict(), 'resnet.pth') 

mse = np.mean((predicted_states - states) ** 2)
print(f"MSE: {mse:.6f}")

# testing on different initial conditions
initial_state_test = np.array([0.5, 0.0])
state = torch.tensor((initial_state_test - mean) / std, dtype=torch.float32)  # Normalize initial state
predicted_states_test = [state.numpy()]
model.eval()
with torch.no_grad():
    for _ in range(len(t_vals) - 1):
        state = model(state)
        predicted_states_test.append(state.numpy())
predicted_states_test = np.array(predicted_states_test)
predicted_states_test = predicted_states_test * std + mean  # Denormalize predictions
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_vals[:-1], states[:-1, 0], label='True x(t)', color='black')
plt.plot(t_vals[:-1], predicted_states_test[:-1, 0], label='Predicted x(t)', linestyle='--')
plt.title('Van der Pol Oscillator: True vs Predicted x(t) - Test Initial Condition')
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(states[:-1, 0], states[:-1, 1], label='True Trajectory', color='black')
plt.plot(predicted_states_test[:-1, 0], predicted_states_test[:-1, 1], label='Predicted Trajectory', linestyle='--')
plt.title('Van der Pol Oscillator: Phase Space: True vs Predicted - Test Initial Condition')        
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.grid()
plt.legend()
plt.tight_layout()
#plt.savefig('resnet_van_der_pol_test.png')
plt.show()

mse_test = np.mean((predicted_states_test - states) ** 2)
print(f"Test MSE: {mse_test:.6f}")
