# Van der Pol Oscillator: Numerical Solvers
# x'' - μ(1 - x^2)x' + x = 0
# as first-order system:
# dx/dt = v
# dv/dt = μ(1 - x^2)v - x

# Solvers to implement:
# Forward Euler, RK4

# Comparisons:
# Accuracy vs reference (scipy solve_ivp)
# Phase portrait comparison
# Convergence order plot
# Energy drift over time

import numpy as np
import matplotlib.pyplot as plt

def van_der_pol(t, state, mu):
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return np.array([dxdt, dvdt])

# Forward Euler: state_next = state + dt * f(t, state)
def euler(func, initial_state, t_start, t_end, dt, mu):
    t_vals = np.arange(t_start, t_end, dt)
    states = np.zeros((len(t_vals), len(initial_state)))
    states[0] = initial_state
    for i in range(1, len(t_vals)):
        states[i] = states[i-1] + dt * func(t_vals[i-1], states[i-1], mu)
    return t_vals, states

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

# SciPy reference solution (ground truth)
from scipy.integrate import solve_ivp 

def reference_solution(func, initial_state, t_start, t_end, mu):
    t_eval = np.linspace(t_start, t_end, 1000)
    sol = solve_ivp(func, (t_start, t_end), initial_state, args=(mu,), t_eval=t_eval)
    return sol.t, sol.y.T

t_start = 0
t_end = 20
dt = 0.01
mu = 1.0
initial_state = [2.0, 0.0]

# Run solvers
t_euler, states_euler = euler(van_der_pol, initial_state, t_start, t_end, dt, mu)
t_rk4, states_rk4 = rk4(van_der_pol, initial_state, t_start, t_end, dt, mu)
t_ref, states_ref = reference_solution(van_der_pol, initial_state, t_start, t_end, mu)

# Plotting results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_ref, states_ref[:, 0], label='Reference (SciPy)', color='black')
plt.plot(t_euler, states_euler[:, 0], label='Euler', linestyle='--')
plt.plot(t_rk4, states_rk4[:, 0], label='RK4', linestyle='-.')
plt.title('Van der Pol Oscillator: x(t)')
plt.xlabel('Time')
plt.ylabel('x')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(states_ref[:, 0], states_ref[:, 1], label='Reference (SciPy)', color='black')
plt.plot(states_euler[:, 0], states_euler[:, 1], label='Euler', linestyle='--')
plt.plot(states_rk4[:, 0], states_rk4[:, 1], label='RK4', linestyle='-.')
plt.title('Phase Portrait')
plt.xlabel('x') 
plt.ylabel('v')
plt.legend()
plt.tight_layout()
plt.savefig('van_der_pol_comparison.png')
plt.show()


# Convergence order plot:
# Run solvers with different dt values and compute error against reference solution at final time.
dt_values = [0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
errors_euler = []
errors_rk4 = []
t_end_convergence = 0.2

_, states_ref = reference_solution(van_der_pol, initial_state, t_start, t_end_convergence, mu)

for dt in dt_values:
    n_steps = int(round(t_end_convergence / dt))
    t_euler_c = np.linspace(t_start, t_end_convergence, n_steps + 1)               
    t_rk4_c = np.linspace(t_start, t_end_convergence, n_steps + 1)                 
    dt_actual = t_euler_c[1] - t_euler_c[0]                                        
                                                                                    
    states_euler_c = np.zeros((len(t_euler_c), 2))                                 
    states_rk4_c = np.zeros((len(t_rk4_c), 2))
    states_euler_c[0] = initial_state                                              
    states_rk4_c[0] = initial_state                                                

    for i in range(1, len(t_euler_c)):                                             
        states_euler_c[i] = states_euler_c[i-1] + dt_actual * van_der_pol(t_euler_c[i-1], states_euler_c[i-1], mu)                               
        states_rk4_c[i] = rk4_step(van_der_pol, t_rk4_c[i-1], states_rk4_c[i-1], dt_actual, mu)                                                                     
                  
    error_euler = np.linalg.norm(states_euler_c[-1] - states_ref[-1])              
    error_rk4 = np.linalg.norm(states_rk4_c[-1] - states_ref[-1])
    errors_euler.append(error_euler)
    errors_rk4.append(error_rk4)

plt.figure()
plt.loglog(dt_values, errors_euler, label='Euler', marker='o')
plt.loglog(dt_values, errors_rk4, label='RK4', marker='o')
plt.loglog(dt_values, [dt**1 * errors_euler[0]/dt_values[0] for dt in dt_values], label='O(dt)', linestyle='--')
plt.loglog(dt_values, [dt**4 * errors_rk4[0]/dt_values[0]**4 for dt in dt_values], label='O(dt^4)', linestyle='--')
plt.title('Convergence Order')
plt.xlabel('Time Step (dt)')
plt.ylabel('Error at Final Time')
plt.legend()
plt.grid()
plt.savefig('convergence_order.png')
plt.show()  

# Energy drift over time:
# Energy for Van der Pol is not conserved, but we can still track it to see how well the solvers capture the dynamics.
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
plt.savefig('energy_drift.png')
plt.show()