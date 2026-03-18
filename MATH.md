# Mathmatical Analysis of Van Der Pol Equation

## Equation:
$$\ddot{x} - \mu(1 - x^2)\dot{x}+x=0$$

## First-Order System
$v = \dot{x}$

$\frac{dx}{dt} = v$

$\frac{dv}{dt} = \mu(1-x^2)v - x$

## Equilibrium and Stability Analysis:

Setting both derivatives to zero:

$\frac{dx}{dt} = 0$

$\frac{dv}{dt} = 0$

From the first equation, $\frac{dx}{dt} = v$, we get $v = 0$. 
Substituting that into the second equation, we get:

$\mu(1-x^2)v - x = -x = 0$ --> $x = 0$

So the unique equilibrium is at $(0, 0)$

## Jacobian Matrix
$$ J = \begin{pmatrix}
  \frac{\partial{\dot{x}}}{\partial{x}} & \frac{\partial{\dot{x}}}{\partial{v}} \\
  \frac{\partial{\dot{v}}}{\partial{x}} & \frac{\partial{\dot{v}}}{\partial{v}}
\end{pmatrix} $$
$$ J = \begin{pmatrix}
  0 & 1 \\
  2\mu vx-1 & \mu(1-x^2)
\end{pmatrix} $$

Evaluating this at $(0,0)$:

$$ J = \begin{pmatrix}
  0 & 1 \\
  -1 & \mu
\end{pmatrix} $$

## Eigenvalues

$\det(J - \lambda I) = 0$:

$\lambda^2 - \mu \lambda + 1 = 0$

$\frac{\mu}{2} \pm \frac{\sqrt{\mu^2 - 4}}{2}$

When $\mu < 2$, you get complex eigenvalues with a positive real part, which gives an unstable spiral, meaning it spirals away from the origin.

When $\mu \geq 2$, you get two real positive eigenvalues, which gives an unstable node, so the trajectories move away from the origin without spiraling.

The origin is unstable for both these cases, but the trajectories do not go to infinity. Because of the nonlinear damping term, $\mu(1-x^2)\dot{x}$, the trajectories settle into a closed loop, the limit cycle. 

## Lagrangian and Hamiltonian Neural Networks

Energy is not conserved with Van der Pol because of the term $\mu(1-x^2)\dot{x}$, which acts as nonlinear damping.

- When $|x| < 1$, there is negative damping, which means energy is pumped into the system, pushing the oscillations to grow. 

- When $|x| > 1$, there is positive damping, which means energy is removed from the system, pulling the oscillations back down.

Hamiltonian Neural Networks and Lagrangian Neural Networks were considered, however these methods are primarily designed for conservative systems with energy-preserving dynamics.
The Van der Pol oscillator is non-conservative with non-linear dissipation that leads to a limit cycle. Because of this, standard HNN and LNN methods do not directly represent its dynamics.

Instead, the ML methods compared were a baseline feedforward NN, a ResNet, a Neural ODE, and a PINN, which don't require a conservative structure. 

## ML Methods
### Baseline FeedForward NN:
The baseline feedforward NN maps time to state. It is a standard multilayer perceptron that serves as a baseline model without any explicit structural assumptions about the dynamics.

### Residual Network (ResNet):
The ResNet learns the vector field F(x,v) and generates trajectories by iterating $h_{n+1}=h_n+dt*F(h_n)$. It starts from an initial condition and repeatedly applies the Euler update to generate a trajectory.

### Neural ODE:
This is a continuous-time version of the ResNet that uses an ODE solver, which makes it well-suited for learning dynamical systems.

### PINN:
The Physics-informed neural network enforces the physics of the Van der Pol equation by embedding the governing ODE directly into the loss function. The network maps time to x(t) and uses automatic differentiation during training to compute the derivatives x'(t) and x''(t) from the network's output.




