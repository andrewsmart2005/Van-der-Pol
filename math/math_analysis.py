

# Van der Pol oscillator
# x'' - μ(1 - x^2)x' + x = 0
# rewrite as a first-order system:
# dx/dt = v
# dv/dt = μ(1 - x^2)v - x
import numpy as np
import matplotlib.pyplot as plt

# I want:
# Time series of x(t) and v(t)
# Phase space plot of x vs v
# Multiiple μ(mu) values to see how the nonlinearity affects the dynamics

# func that takes in state [x, v] and returns [dx/dt, dv/dt]
def van_der_pol(t, state, mu):
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return np.array([dxdt, dvdt])

# analyze the equilibrium
mu_vals = [0.5, 1.0, 2.0, 4.0]
for mu in mu_vals:
    J = np.array([[0, 1], [-1, mu]])
    eigenvalues = np.linalg.eigvals(J)
    print(f"Eigenvalues of the Jacobian at the equilibrium point for μ={mu}:", eigenvalues)

# Write RK4 solver so we can generate trajectories
def rk4_step(func, t, state, dt, mu):
    k1 = func(t, state, mu)
    k2 = func(t + dt/2, state + dt/2 * k1, mu)
    k3 = func(t + dt/2, state + dt/2 * k2, mu)
    k4 = func(t + dt, state + dt * k3, mu)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# RK4 integration loop -> take in func, initial state, time range, and mu; return trajectory: times and states
def rk4(func, initial_state, t_start, t_end, dt, mu):
    t_vals = np.arange(t_start, t_end, dt)
    states = np.zeros((len(t_vals), len(initial_state)))
    states[0] = initial_state
    for i in range(1, len(t_vals)):
        states[i] = rk4_step(func, t_vals[i-1], states[i-1], dt, mu)
    return t_vals, states

# Parameters
t_start = 0.0
t_end = 30.0
dt = 0.01
initial_state = np.array([2.0, 0.0])
initial_states = [np.array([0.1, 0.0]), np.array([1.0, -2.0]), np.array([0.0, 2.0])]
mu = 1.0
mu_values = [0.5, 1.0, 2.0]  # Different nonlinearity parameters


# Plot with one initial state, varying mu
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
for mu in mu_values:
    t_values, states = rk4(van_der_pol, initial_state, t_start, t_end, dt, mu)
    plt.plot(t_values, states[:, 0], label=f'μ={mu}')
plt.title('Van der Pol Oscillator: x(t) for Different μ')
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.grid()
plt.legend()
plt.subplot(2, 1, 2)
for mu in mu_values:
    t_values, states = rk4(van_der_pol, initial_state, t_start, t_end, dt, mu)
    plt.plot(states[:, 0], states[:, 1], label=f'μ={mu}')
plt.title('Van der Pol Oscillator: Phase Space (x vs v) for Different μ')
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('../figures/van_der_pol_mu_analysis.png')
plt.show()

# Plot with one mu (=1), varying initial conditions
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
for initial_state in initial_states:
    t_values, states = rk4(van_der_pol, initial_state, t_start, t_end, dt, mu)
    plt.plot(t_values, states[:, 0], label=f'Initial={initial_state}')
plt.title('Van der Pol Oscillator: x(t) for Different Initial Conditions ')
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.grid()
plt.legend()
plt.subplot(2, 1, 2)
for initial_state in initial_states:
    t_values, states = rk4(van_der_pol, initial_state, t_start, t_end, dt, mu)
    plt.plot(states[:, 0], states[:, 1], label=f'Initial={initial_state}')
plt.title('Van der Pol Oscillator: Phase Space (x vs v) for Different Initial Conditions')
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig('../figures/van_der_pol_initial_conditions.png')
plt.show()

# vector field plot for mu = 1.0
mu = 1.0
x_range = np.linspace(-5, 5, 20)
v_range = np.linspace(-5, 5, 20)
X, V = np.meshgrid(x_range, v_range)

DX = V
DV = mu * (1 - X**2) * V - X

plt.figure()
plt.streamplot(X, V, DX, DV)  
plt.title(f'Van der Pol Vector Field (μ={mu})') 
plt.xlabel('x')
plt.ylabel('v')                                                                    
plt.grid()                                                                         
plt.savefig('../figures/van_der_pol_vector_field.png')
plt.show()                                                                         