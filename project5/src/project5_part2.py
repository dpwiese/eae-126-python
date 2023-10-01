"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 5: Steady, inviscid, adiabatic, compressible, and irrotational flows over airfoils -
    numerical solutions to thickness problem
Part 2: Body of revolution at zero angle of attack

Show plots for biconvex and ellipse bodies of revolution.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, '../..')

from tools.tridiagscalar import tridiagscalar

uinf = 1
mach_num_array = [0.0]

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
omega = 1.8
resmax = 1e-6

# tau - thickness ratio of parabolic profiles and NACA
# sh - vertical shift for parabolic profiles
# coe - 'A' coefficient for parabolic profiles
# chord - chordlength of airfoil points on grid
tau = 0.1
sh = tau * (x[nTE-1] + dx / 2)
coe = sh / x[nLE-1]**2
chord = (x[nTE-1] - x[nLE-1])

aell = chord / 2
bell = 0.40

yBB = np.zeros(nx)
yBT = np.zeros(nx)

S = np.zeros(nx)

u = np.zeros((nx, ny))
uold = np.zeros((nx, ny))

A = np.zeros(ny)
B = np.ones(ny)
C = np.zeros(ny)
D = np.zeros(ny)

resid = np.zeros((nx, ny))

# Case 1: Biconvex
for i in range(nLE-1, nTE):
    yBT[i] = -coe * x[i] ** 2 + sh
    yBB[i] = -yBT[i]

# # Case 2: Ellipse
# for i in range(nLE-1, nTE):
#     yBT[i] = bell * np.sqrt(1-x[i] ** 2 / aell ** 2)
#     yBB[i] = -yBT[i]

for i in range(nLE-1, nTE):
    S[i] = np.pi * yBT[i] ** 2

for idx, mach_num in enumerate(mach_num_array):

    res = 1
    iterator = 1
    resplot = np.zeros(maxiterator)

    Beta = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            Beta[i, j] = np.sqrt(1 - mach_num ** 2)

    # Main iteratorator
    while iterator < maxiterator and res > resmax:

        # Points upstream of airfoil
        for i in range(1, nLE - 1):
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, ny - 1):
                A[j] = (y[j] - dy / 2) / (y[j] * dy**2)
                B[j] = -(y[j] - dy / 2) / (y[j] * dy**2) - (y[j] + dy / 2) / (y[j] * dy**2) - \
                            2 * (Beta[i, j]**2) / dx**2
                C[j] = (y[j] + dy / 2) / (y[j] * dy**2)
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
                A[j] = (y[j] - dy / 2) / (y[j] * dy**2)
                B[j] = -(y[j] - dy / 2) / (y[j] * dy**2) - (y[j] + dy / 2) / (y[j] * dy**2) - \
                            2 * (Beta[i, j]**2) / dx**2
                C[j] = (y[j] + dy / 2) / (y[j] * dy**2)
                D[j] = -(Beta[i, j]**2) * (uold[i - 1, j] + uold[i + 1, j]) / dx**2

            # Just before body (bottom)
            for j in [ny // 2 - 1]:
                A[j] = (y[j] - dy / 2) / (y[j] * dy**2)
                B[j] = -(y[j] - dy / 2) / (y[j] * dy**2) - 2 * (Beta[i, j]**2) / dx**2
                C[j] = 0
                D[j] = -(Beta[i, j]**2) * (uold[i - 1, j] + uold[i + 1, j]) / dx**2 - \
                            (S[i + 1] - 2 * S[i] + S[i - 1]) / (2 * np.pi * dx**2 * dy * y[j])

            # Just after body (top)
            for j in [ny // 2]:
                A[j] = 0
                B[j] = -(y[j] + dy / 2) / (y[j] * dy**2) - 2 * (Beta[i, j]**2) / dx**2
                C[j] = (y[j] + dy / 2) / (y[j] * dy**2)
                D[j] = -(Beta[i, j]**2) * (uold[i - 1, j] + uold[i + 1, j]) / dx**2 + \
                            (S[i + 1] - 2 * S[i] + S[i - 1]) / (2 * np.pi * dx**2 * dy * y[j])

            # Region above body
            for j in range(ny // 2 + 1, ny - 1):
                A[j] = (y[j] - dy / 2) / (y[j] * dy**2)
                B[j] = -(y[j] - dy / 2) / (y[j] * dy**2) - (y[j] + dy / 2) / (y[j] * dy**2) - \
                            2 * (Beta[i, j]**2) / dx**2
                C[j] = (y[j] + dy / 2) / (y[j] * dy**2)
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
                A[j] = (y[j] - dy / 2) / (y[j] * dy**2)
                B[j] = -(y[j] - dy / 2) / (y[j] * dy**2) - (y[j] + dy / 2) / (y[j] * dy**2) - \
                            2 * (Beta[i, j]**2) / dx**2
                C[j] = (y[j] + dy / 2) / (y[j] * dy**2)
                D[j] = -(Beta[i, j]**2) * (uold[i - 1, j] + uold[i + 1, j]) / dx**2

            for j in [ny - 1]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            u[i, :] = tridiagscalar(A, B, C, D)
            u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
            uold[i, :] = u[i, :]

        # Calculate Residual
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                resid[i, j] = abs(
                    ((y[j] - dy / 2) / (y[j] * dy**2)) * u[i, j - 1] +
                    (-(y[j] - dy / 2) / (y[j] * dy**2) - (y[j] + dy / 2) / (y[j] * dy**2) - 2 * (Beta[i, j]**2) / dx**2) * u[i, j] +
                    ((y[j] + dy / 2) / (y[j] * dy**2)) * u[i, j + 1] -
                    (-(Beta[i, j]**2) * (uold[i - 1, j] + uold[i + 1, j]) / dx**2)
                )

        for i in range(nLE - 1, nTE + 2):
            for j in range(int(ny / 2) - 1, int(ny / 2) + 3):
                resid[i, j] = 0

        res = np.max(resid)
        resplot[iterator - 1] = res
        iterator += 1

    xcp = np.zeros(nTE - nLE + 1)
    cpbot = np.zeros(nTE - nLE + 1)
    cptop = np.zeros(nTE - nLE + 1)

    for i in range(nLE - 1, nTE):
        xcp[i - nLE + 1] = x[i]
        cpbot[i - nLE + 1] = -2 * u[i, ny // 2 - 1] / uinf
        cptop[i - nLE + 1] = -2 * u[i, ny // 2] / uinf

    # Create subplots for the current loop iteratoration
    fig = plt.figure(figsize=(12, 8), dpi=200, facecolor='w', edgecolor='k')

    TITLE_FONT_SIZE = 10
    TICK_FONT_SIZE = 8

    # Plot 4: Perturbation velocity
    plt.subplot(3, 2, 4)
    plt.plot(x, u[:, ny // 2 - 1], '--b')
    plt.plot(x, u[:, ny // 2], '-b')
    plt.title('Perturbation Velocity Above and Below Airfoil', fontsize=TITLE_FONT_SIZE)
    plt.legend(['Bottom', 'Top'])
    plt.grid(True)
    plt.gca().tick_params(labelsize=TICK_FONT_SIZE)
    h4 = plt.gca()

    # Plot 1: Airfoil profile
    plt.subplot(3, 2, 1)
    plt.plot(x, yBB, '--b')
    plt.title('Airfoil Profile', fontsize=TITLE_FONT_SIZE)
    plt.plot(x, yBT, '-b')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([-4, 4, -1, 1])
    plt.legend(['Bottom', 'Top'])
    plt.grid(True)
    plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

    # Plot 2: Pressure contours around airfoil
    plt.subplot(3, 2, 2)
    c2 = plt.contour(x, y, u.T, 100)
    plt.title(f'Pressure Contours Around Airfoil Mach: {mach_num}', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('u-direction')
    plt.ylabel('v-direction')
    plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

    norm2 = matplotlib.colors.Normalize(vmin=c2.cvalues.min(), vmax=c2.cvalues.max())
    sm2 = plt.cm.ScalarMappable(norm=norm2, cmap = c2.cmap)
    sm2.set_array([])

    # Set the colorbar to the second subplot (for spacing only), remove it, and set the colorbar
    # on the intended subplot.
    cb1 = plt.colorbar(c2, ax=h4, orientation='vertical', fraction=0.046, pad=0.04)
    cb1.remove()
    cb2 = plt.colorbar(sm2, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=TICK_FONT_SIZE)

    # Plot 3: -Cp across airfoil
    plt.subplot(3, 2, 3)
    plt.plot(xcp, -cptop, '-b')
    plt.title('-C_P Across Airfoil', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Location Along Chord')
    plt.ylabel('-C_P')
    plt.plot(xcp, -cpbot, '--b')
    plt.legend(['Top', 'Bottom'])
    plt.grid(True)
    plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

    # Plot 5: Residuals
    plt.subplot(3, 2, 5)
    plt.semilogy(resplot[resplot != 0])
    plt.title('Residual versus Iteration Count', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Residual')
    plt.grid(True)
    plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

    plt.tight_layout()
    fig.savefig(f'../fig/project5_part2_{idx}.png', bbox_inches='tight')

# Show all plots
plt.show()
