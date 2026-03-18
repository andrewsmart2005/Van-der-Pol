# Generate training data for a simple feedforward neural network to learn the Van der Pol oscillator dynamics.
# Normalize the data
# Build the model - simple feedforward network
# Train the model - MSE loss, Adam optimizer
# Evaluate the model - compare predictions to true trajectories
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
np.random.seed(42)

# Van der Pol oscillator dynamics
def van_der_pol(t, state, mu):
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return np.array([dxdt, dvdt])

# Generate training data using RK4 solver
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

# Parameters
t_start = 0.0
t_end = 20.0
dt = 0.01
initial_state = np.array([2.0, 0.0])
mu = 1.0

# Generate training data
t_vals, states = rk4(van_der_pol, initial_state, t_start, t_end, dt, mu)

# noisy
#states += 0.1 * np.random.normal(size=states.shape)

# Normalize the data
t_min, t_max = t_vals.min(), t_vals.max()
t_norm = (t_vals - t_min) / (t_max - t_min)

X_train = t_norm[:-1].reshape(-1, 1)  # Time as input
y_train = states[:-1] # Next state as target

# Build the model - simple feedforward network, Tanh activation to capture nonlinearity
class VanDerPolNN(nn.Module):
    def __init__(self):
        super(VanDerPolNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)
    
# Train the model
model = VanDerPolNN()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert training data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
iterations = 10000
for iter in range(iterations):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = loss_fn(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (iter+1) % 100 == 0:
        print(f'Iteration {iter+1}/{iterations}, Loss: {loss.item():.6f}')

# Evaluate the model - compare predictions to true trajectories
model.eval()
with torch.no_grad():
    predictions = model(X_train_tensor).numpy()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_vals[:-1], states[:-1, 0], label='True x(t)', color='black')
plt.plot(t_vals[:-1], predictions[:, 0], label='Predicted x(t)', linestyle='--')
plt.title('Van der Pol Oscillator: True vs Predicted x(t)')
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(states[:-1, 0], states[:-1, 1], label='True Phase Portrait', color='black')
plt.plot(predictions[:, 0], predictions[:, 1], label='Predicted Phase Portrait', linestyle='--')
plt.title('Van der Pol Oscillator: True vs Predicted Phase Portrait')
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('van_der_pol_nn_comparison.png')
plt.show()  
torch.save(model.state_dict(), 'baseline_nn.pth')

mse = np.mean((predictions - states[:-1]) ** 2) 
print(f"Baseline NN MSE: {mse:.6f}")