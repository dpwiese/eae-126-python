"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 3: Small disturbance theory for airfoils and bodies of revolution
Part 1B: Thickness and lifting problem in 2D
"""

import numpy as np
import matplotlib.pyplot as plt

# Initialization of variables and parameters
uinf = 1.0
rhoinf = 1.0
a = 1.0

xmin = -1
xmax = 1
nx = 200
dx = (xmax - xmin) / (nx - 1)
x = np.linspace(xmin, xmax, nx)
tmin = xmin + dx / 2
tmax = xmax - dx / 2
nt = nx - 1
t = np.linspace(tmin, tmax, nt)

# Generate the profile shapes
y_flat_plate = np.zeros(nx)
y_circular_arc = -(0.1 / a) * x ** 2 + 0.1 * a

# Define configurations
configs = [
    { 'alphadeg': 5,    'name': 'flat plate at aoa',    'y': y_flat_plate },
    { 'alphadeg': 0,    'name': 'circular arc',         'y': y_circular_arc }
]

plt.figure(figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')

for idx, cfg in enumerate(configs):

    # Parameters for each transformation
    alphadeg    = cfg['alphadeg']
    y           = cfg['y']

    alpha = np.deg2rad(alphadeg)

    # Calculate the derivative of y with respect to x
    dydx = np.zeros(nx)
    dydx[0] = (y[1] - y[0]) / dx
    dydx[1:nx - 1] = (y[2:] - y[:nx - 2]) / (2 * dx)
    dydx[nx - 1] = (y[nx - 1] - y[nx - 2]) / dx

    n = nx - 1

    # Initialize A and B matrices
    A = np.zeros((nx - 1, nt))
    B = np.zeros(nx - 1)

    # Populate A and B matrices
    for i in range(nx - 1):
        for j in range(nt):
            A[i, j] = dx / (x[i + 1] - t[j])
        B[i] = (alpha - dydx[i]) * 2 * np.pi * uinf

    # Modify A and B for last row
    A[nx - 2, :nt - 2] = 0
    A[nx - 2, nt - 2] = -1 / 2
    A[nx - 2, nt - 1] = 2 / 2
    B[nx - 2] = 0

    # Perform Gaussian Elimination
    for j in range(n - 1):
        for i in range(j, n - 1):
            bar = A[i + 1, j] / A[j, j]
            A[i + 1, :] = A[i + 1, :] - A[j, :] * bar
            B[i + 1] = B[i + 1] - bar * B[j]

    # Perform back substitution
    gamma = np.zeros(n)
    gamma[n - 1] = B[n - 1] / A[n - 1, n - 1]

    for j in range(n - 2, 1, -1):
        gamma[j] = (B[j] - np.dot(A[j, j + 1:n], gamma[j + 1:n])) / A[j, j]

    # Compute gammax
    gammax = np.zeros(nx)

    for i in [0]:
        gammax[i] = gamma[i] - 0.5 * (gamma[i + 1] - gamma[i])

    for i in range(1, nx - 1):
        gammax[i] = (gamma[i] + gamma[i - 1]) / 2

    for i in [nx - 1]:
        gammax[i] = 0

    # Sum gamma
    gamma_total = np.sum(0.5 * (gammax[:-1] + gammax[1:]) * dx)

    numer = np.sum((x[:-1] - xmin) * (0.5 * (gammax[:-1] + gammax[1:]) * dx))
    denom = np.sum(0.5 * (gammax[:-1] + gammax[1:]) * dx)
    xcp = 0.5 * numer / denom

    LIFT = rhoinf * uinf * gamma_total
    CL = LIFT / (0.5 * rhoinf * uinf**2)

    uc_x = 0.5 * gammax

    cp = 2 * uc_x / uinf

    plt.subplot(2, 2, idx + 1)
    plt.plot(x, y, linewidth=2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([-a, a, -0.5, 0.5])
    plt.title('Profile Shape')
    plt.grid(True)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')

    plt.subplot(2, 2, idx + 3)
    plt.plot(x, cp)
    plt.title(f'C_p Distribution Over Profile Surface AOA: {cfg["alphadeg"]}')
    plt.xlabel('Chord Position')
    plt.ylabel('-C_p')
    plt.grid(True)

plt.tight_layout()
plt.savefig(f'../fig/project3_part1b.png', bbox_inches='tight')

# Show all plots
plt.show()
