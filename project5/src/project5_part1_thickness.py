"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 5: Steady, inviscid, adiabatic, compressible, and irrotational flows over airfoils -
    numerical solutions to thickness problem
Part 1: Two Dimensional Symmetric Airfoils at Zero Angle of Attack

Five cases: flat plate, biconvex, cambered plate, NACA, and Joukowski airfoil
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, '../..')

from tools.tridiagscalar import tridiagscalar

# Constants and parameters
uinf = 1
mach_num_array = [0.0, 0.4, 0.8]

# ny must be even
nx = 100
ny = 100

xmin = -10
xmax = 10
ymin = -10
ymax = 10

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

nLE = round(2 * nx / 5)
nTE = round(3 * nx / 5) + 1

maxiterator = 1000
omega = 1.91
resmax = 1e-6

# tau - thickness ratio of parabolic profiles and NACA
# sh - vertical shift for parabolic profiles
# coe - 'A' coefficient for parabolic profiles
# chord - chordlength of airfoil points on grid
tau = 0.1
sh = tau * (x[nTE - 1] + dx / 2)
coe = sh / x[nLE - 1] ** 2
chord = (x[nTE - 1] - x[nLE - 1])

# Joukowski stuff
ntheta = 1000
theta = np.linspace(0, 2 * np.pi, ntheta)
epsilon = 0.1
mu = 0.1

# Adjust size of 'b' so chord length is correct
b = (chord / 4) * 0.99
a = np.sqrt(mu**2 + (b - epsilon)**2)

xj = epsilon + a * np.cos(theta)
yj = mu + a * np.sin(theta)
X = xj * (1 + (b**2) / (xj**2 + yj**2))
Y = yj * (1 - (b**2) / (xj**2 + yj**2))

# Case 1: Flat Plate (default)
yBT = np.zeros(nx)
yBB = np.zeros(nx)

u    = np.zeros((nx, ny))
uold = np.zeros((nx, ny))

A = np.zeros(ny)
B = np.ones(ny)
C = np.zeros(ny)
D = np.zeros(ny)

resid = np.zeros((nx, ny))

# Case 2: Biconvex
for i in range(nLE-1, nTE):
    yBT[i] = -coe * x[i]**2 + sh
    yBB[i] = coe * x[i]**2 - sh

# # Case 3: Cambered plate
# for i in range(nLE - 1, nTE):
#     yBT[i] = -coe * x[i]**2 + sh
#     yBB[i] = -coe * x[i]**2 + sh

# # Case 4: NACA 00xx
# for i in range(nLE - 1, nTE):
#     yBT[i] = 10 * tau * chord * (0.2969 * np.sqrt((x[i] + chord/2) / chord) - \
#                                 0.1260 * ((x[i] + chord/2) / chord) - \
#                                 0.3537 * ((x[i] + chord/2) / chord)**2 + \
#                                 0.2843 * ((x[i] + chord/2) / chord)**3 - \
#                                 0.1015 * ((x[i] + chord/2) / chord)**4)
#     yBB[i] = -yBT[i]

# # Case 5: Joukowski (This is broken)
# # TODO@dpwiese - fix me
# rlen = np.zeros(nx)
# for i in range(nLE - 1, nTE):
#     # Find locations in X
#     r = np.where((X > x[i-1]) & (X < x[i+1]))[0]
#     rlen[i] = len(r)
#     yBB[i] = Y[int(round(0.15 * rlen[i]))]
#     yBT[i] = Y[int(round(0.85 * rlen[i]))]
#     # yBB[i] = Y[r[0]]
#     # yBT[i] = Y[r[rlen[i]]]

# Loop over configurations (only varying Mach number)
for idx, mach_num in enumerate(mach_num_array):

    res = 1
    iterator = 0
    resplot = []

    Beta = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            Beta[i, j] = np.sqrt(1 - mach_num ** 2)

    # Main iterator
    while iterator < maxiterator and res > resmax:

        # Points upstream of airfoil
        for i in range(1, nLE - 1):
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, ny - 1):
                A[j] = 1 / dy**2
                B[j] = -2 / dy**2 - 2 * (Beta[i, j]**2) / dx**2
                C[j] = 1 / dy**2
                D[j] = -(Beta[i, j]**2) * (uold[i - 1, j] + uold[i + 1, j]) / dx**2

            for j in [ny - 1]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            u[i, :] = tridiagscalar(A, B, C, D)
            u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
            uold[i, :] = u[i, :]

        # Points where airfoil is
        for i in range(nLE - 1, nTE):
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            # Region below body
            for j in range(1, ny // 2 - 1):
                A[j] = 1 / dy**2
                B[j] = -2 / dy**2 - 2 * (Beta[i, j]**2) / dx**2
                C[j] = 1 / dy**2
                D[j] = -(Beta[i, j]**2) * (uold[i - 1, j] + uold[i + 1, j]) / dx**2

            # Just before body
            for j in [ny // 2 - 1]:
                A[j] = 1 / dy**2
                B[j] = -1 / dy**2 - 2 * (Beta[i, j]**2) / dx**2
                C[j] = 0
                D[j] = -(Beta[i, j]**2) * (uold[i - 1, j] + uold[i + 1, j]) / dx**2 - \
                            (yBB[i + 1] - 2 * yBB[i] + yBB[i - 1]) / (dx**2 * dy)

            # Region just after body
            for j in [ny // 2]:
                A[j] = 0
                B[j] = -1 / dy**2 - 2 * (Beta[i, j]**2) / dx**2
                C[j] = 1 / dy**2
                D[j] = -(Beta[i, j]**2) * (uold[i - 1, j] + uold[i + 1, j]) / dx**2 + \
                            (yBT[i + 1] - 2 * yBT[i] + yBT[i - 1]) / (dx**2 * dy)

            # Region above body
            for j in range(ny // 2 + 1, ny - 1):
                A[j] = 1 / dy**2
                B[j] = -2 / dy**2 - 2 * (Beta[i, j]**2) / dx**2
                C[j] = 1 / dy**2
                D[j] = -(Beta[i, j]**2) * (uold[i - 1, j] + uold[i + 1, j]) / dx**2

            for j in [ny - 1]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            u[i, :] = tridiagscalar(A, B, C, D)
            u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
            uold[i, :] = u[i, :]

        # Points downstream of airfoil
        for i in range(nTE, nx - 1):
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, ny - 1):
                A[j] = 1 / dy**2
                B[j] = -2 / dy**2 - 2 * (Beta[i, j]**2) / dx**2
                C[j] = 1 / dy**2
                D[j] = -(Beta[i, j]**2) * (uold[i - 1, j] + uold[i + 1, j]) / dx**2

            for j in [ny - 1]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            u[i, :] = tridiagscalar(A, B, C, D)
            u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
            uold[i, :] = u[i, :]

        # Residual
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                resid[i, j] = abs((Beta[i, j]**2) * (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2 + \
                                 (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy**2)

        for i in range(nLE - 2, nTE + 1):
            for j in range(ny // 2 - 2, ny // 2 + 2):
                resid[i, j] = 0

        res = np.max(resid)
        resplot.append(res)
        iterator += 1

    cpbot = -2 * u[:, ny // 2 - 1] / uinf
    cptop = -2 * u[:, ny // 2] / uinf

    # Create subplots for the current loop iteratoration
    plt.figure(figsize=(12, 8), dpi=200, facecolor='w', edgecolor='k')

    # Airfoil profile
    plt.subplot(3, 2, 1)
    plt.plot(x, yBB, '--b')
    plt.plot(x, yBT, '-b')
    plt.title(f'Airfoil Profile (mach_num {mach_num})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([-3, 3, -1, 1])
    plt.grid()

    # Pressure contours around airfoil
    plt.subplot(3, 2, 2)
    plt.contour(x, y, u.T, 100)
    plt.title('Pressure Contours Around Airfoil')
    plt.xlabel('u-direction')
    plt.ylabel('v-direction')
    plt.colorbar()
    plt.grid()

    # -Cp across airfoil
    plt.subplot(3, 2, 3)
    plt.plot(x, -cptop, '-b', label='Top')
    plt.plot(x, -cpbot, '--b', label='Bottom')
    plt.title('-C_P Across Airfoil')
    plt.xlabel('Location Along Chord')
    plt.ylabel('-C_P')
    plt.legend()
    plt.grid()

    # Perturbation velocity
    plt.subplot(3, 2, 4)
    plt.plot(x, u[:, ny//2 - 1], '--b', label='Bottom')
    plt.plot(x, u[:, ny//2], '-b', label='Top')
    plt.title('Perturbation Velocity Above and Below Airfoil')
    plt.xlabel('x')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid()

    # Residuals
    plt.subplot(3, 2, 5)
    plt.semilogy(resplot, label=f'mach_num {mach_num}')
    plt.title('Residual versus Iteration Count')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Residual')
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'../fig/project5_part1_thickness_{idx}.png', bbox_inches='tight')

# Show all plots
plt.show()
