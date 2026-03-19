# Forward Euler, RK4

# Comparisons:

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def van_der_pol(t, state, mu):
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return np.array([dxdt, dvdt])

# Forward Euler
def euler(func, initial_state, t_start, t_end, dt, mu):
    t_vals = np.arange(t_start, t_end, dt)
    results = np.zeros((len(t_vals), len(initial_state)))
    results[0] = initial_state
    for i in range(1, len(t_vals)):
        derivative = func(t_vals[i-1], results[i-1], mu)
        results[i] = results[i-1] + dt * derivative
    return t_vals, results

# RK4: weighted average of 4 slope estimates
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


from scipy.integrate import solve_ivp 

# reference sol
def reference_solution(func, initial_state, t_start, t_end, mu):
    t_eval = np.linspace(t_start, t_end, 1000)
    sol = solve_ivp(func, (t_start, t_end), initial_state, args=(mu,), t_eval=t_eval)
    return sol.t, sol.y.T

# param
t_start = 0
t_end = 20
dt = 0.01
mu = 1.0
initial_state = [2.0, 0.0]

# Run solvers
t_euler, states_euler = euler(van_der_pol, initial_state, t_start, t_end, dt, mu)
t_rk4, states_rk4 = rk4(van_der_pol, initial_state, t_start, t_end, dt, mu)
t_ref, states_ref = reference_solution(van_der_pol, initial_state, t_start, t_end, mu)

# time series
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_ref, states_ref[:, 0], label='Reference', color='black')
plt.plot(t_euler, states_euler[:, 0], label='Euler', linestyle='--')
plt.plot(t_rk4, states_rk4[:, 0], label='RK4', linestyle='-.')
plt.title('Van der Pol Oscillator: x(t)')
plt.xlabel('Time')
plt.ylabel('x')
plt.legend()
plt.subplot(1, 2, 2)
# phase
plt.plot(states_ref[:, 0], states_ref[:, 1], label='Reference (SciPy)', color='black')
plt.plot(states_euler[:, 0], states_euler[:, 1], label='Euler', linestyle='--')
plt.plot(states_rk4[:, 0], states_rk4[:, 1], label='RK4', linestyle='-.')
plt.title('Phase Portrait')
plt.xlabel('x') 
plt.ylabel('v')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'van_der_pol_comparison.png')
plt.show()


# Convergence order plot:
# Run solvers with different dt values and compute error against reference solution at final time.
dt_values = [0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
errors_euler = []
errors_rk4 = []
t_end_convergence = 0.2

t_ref, states_ref = reference_solution(van_der_pol, initial_state, t_start, t_end_convergence, mu)

for dt in dt_values:
    # how many step
    n_steps = int(round(t_end_convergence / dt))
    t_points = np.linspace(t_start, t_end_convergence, n_steps + 1)
    h = t_points[1] - t_points[0]  

    euler_states = np.zeros((len(t_points), 2))
    rk4_states = np.zeros((len(t_points), 2))
    euler_states[0] = initial_state
    rk4_states[0] = initial_state                                             

    for i in range(1, len(t_points)):
        # derivs
        derivative = van_der_pol(t_points[i-1], euler_states[i-1], mu)
        euler_states[i] = euler_states[i-1] + h * derivative
        rk4_states[i] = rk4_step(van_der_pol, t_points[i-1], rk4_states[i-1], h, mu)
    #dist
    error_euler = np.linalg.norm(euler_states[-1] - states_ref[-1])
    error_rk4 = np.linalg.norm(rk4_states[-1] - states_ref[-1])
    errors_euler.append(error_euler)
    errors_rk4.append(error_rk4)

# error vs dt
plt.figure()
plt.loglog(dt_values, errors_euler, label='Euler', marker='o')
plt.loglog(dt_values, errors_rk4, label='RK4', marker='o')
plt.loglog(dt_values, [dt**1 * errors_euler[0]/dt_values[0] for dt in dt_values], label='O(dt)', linestyle='--')
plt.loglog(dt_values, [dt**4 * errors_rk4[0]/dt_values[0]**4 for dt in dt_values], label='O(dt^4)', linestyle='-.')
plt.title('Convergence Order')
plt.xlabel('Time Step (dt)')
plt.ylabel('Error at Final Time')
plt.legend()
plt.grid()
plt.savefig(FIGURES_DIR / 'convergence_order.png')
plt.show()  

# Energy drift over time:
# energy not conserved, drift should be worse for Eu
dt = 0.3
t_euler, states_euler = euler(van_der_pol, initial_state, t_start, t_end, dt, mu)
t_rk4, states_rk4 = rk4(van_der_pol, initial_state, t_start, t_end, dt, mu)

def energy(state, mu):
    x, v = state
    return 0.5 * v**2 + 0.5 * x**2

energy_euler = [energy(state, mu) for state in states_euler]
energy_rk4 = [energy(state, mu) for state in states_rk4]
plt.figure()
plt.plot(t_euler, energy_euler, label='Euler', linestyle='--')
plt.plot(t_rk4, energy_rk4, label='RK4', linestyle='-.')
plt.title('Energy Drift Over Time')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.grid()
plt.savefig(FIGURES_DIR / 'energy_drift.png')
plt.show()
