"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 3: Small disturbance theory for airfoils and bodies of revolution
Part 1A: Thickness and lifting problem in 2D
"""

import numpy as np
import matplotlib.pyplot as plt

# Initialization of variables and parameters
uinf = 1.0
a = 1.0
tau = 0.1
b = tau * a

xmin = -a
xmax = a
nx = 100
dx = (xmax - xmin) / (nx - 1)
x = np.linspace(xmin, xmax, nx)

tmin = xmin + dx / 2
tmax = xmax - dx / 2
nt = nx - 1
t = np.linspace(tmin, tmax, nt)

# Generate the profile shapes
y_biconvex = -(0.1 / a) * x ** 2 + 0.1 * a
y_ellipse = np.sqrt(b**2 * (1 - x**2 / a**2))

# Define configurations
configs = [
    {'y': y_biconvex},
    {'y': y_ellipse}
]

plt.figure(figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')

for idx, cfg in enumerate(configs):

    y = cfg['y']

    # Calculate the derivative of y with respect to t
    dydt = np.zeros(nt)
    for j in range(nt):
        dydt[j] = (y[j + 1] - y[j]) / dx

    # Compute the tangential velocity component numerically
    ut_num = np.zeros(nx)
    for i in range(nx):
        ut = np.zeros(nt)
        for j in range(nt):
            ut[j] = 2 * a * (uinf / np.pi) * dydt[j] / (x[i] - t[j])
        utsum = 0
        for k in range(nt - 1):
            temp = 0.5 * (ut[k] + ut[k + 1]) * dx
            utsum += temp
        ut_num[i] = utsum

    # Compute the analytical solution for ut
    ut_ana = np.full(nx, tau * uinf)

    plt.subplot(2, 2, idx + 1)
    plt.plot(x, y, linewidth=2)
    plt.plot(x, -y, linewidth=2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([-a, a, -0.5, 0.5])
    plt.title('Profile Shape')
    plt.grid(True)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')

    plt.subplot(2, 2, idx + 3)
    plt.plot(x, ut_num, '--', label='Numerical Solution')
    plt.plot(x, ut_ana, label='Analytical Solution')
    plt.title('U_T Across Profile Chord')
    plt.grid(True)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'../fig/project3_part1a_{idx}.png', bbox_inches='tight')

# Show all plots
plt.show()
