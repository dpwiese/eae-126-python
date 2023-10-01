"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 6: Steady, inviscid, adiabatic, compressible, and irrotational flows over airfoils -
    numerical solutions to lifting Problem
Part 2: Circular arc at zero angle of attack
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, '../..')

from tools.tridiagscalar import tridiagscalar

aoa_deg = 0
aoa_rad = np.deg2rad(aoa_deg)
uinf = 1

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

maxiter = 1000
omega = 1.91
resmax = 1e-6

# tau - thickness ratio of parabolic profiles and NACA
# sh - vertical shift for parabolic profiles
# coe - 'A' coefficient for parabolic profiles
tau = 0.1
sh = 2 * tau * (x[nTE - 1] + dx / 2)
coe = sh / (x[nLE - 1] - dx / 2) ** 2

yBT = np.zeros(nx)
yBB = np.zeros(nx)

for i in range(nLE - 1, nTE):
    yBT[i] = -coe * x[i] ** 2 + sh
    yBB[i] = -coe * x[i] ** 2 + sh

mach_num = 0

u = np.zeros((nx, ny))
uold = np.zeros((nx, ny))

for i in [0]:
    for j in range(ny):
        u[i, j] = uinf * np.cos(aoa_rad)
        uold[i, j] = uinf * np.cos(aoa_rad)

for i in [nx - 1]:
    for j in range(ny):
        u[i, j] = uinf * np.cos(aoa_rad)
        uold[i, j] = uinf * np.cos(aoa_rad)

for i in range(nx):
    for j in [0]:
        u[i, j] = uinf * np.cos(aoa_rad)
        uold[i, j] = uinf * np.cos(aoa_rad)

    for j in [ny - 1]:
        u[i, j] = uinf * np.cos(aoa_rad)
        uold[i, j] = uinf * np.cos(aoa_rad)

resu = 1
iteru = 1

residu = np.zeros((nx, ny))
resplotu = np.zeros(maxiter)

Au = np.zeros(ny)
Bu = np.zeros(ny)
Cu = np.zeros(ny)
Du = np.zeros(ny)

Beta = np.zeros((nx, ny))

for i in range(nx):
    for j in range(ny):
        Beta[i, j] = np.sqrt(1 - mach_num ** 2)

while iteru < maxiter and resu > resmax:
    # Points upstream of airfoil
    for i in range(1, nLE - 1):
        for j in [0]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = uinf * np.cos(aoa_rad)

        for j in range(1, ny - 1):
            Au[j] = 1 / dy ** 2
            Bu[j] = -2 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 1 / dy ** 2
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2

        for j in [ny - 1]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = uinf * np.cos(aoa_rad)

        u[i, :] = tridiagscalar(Au, Bu, Cu, Du)
        u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
        uold[i, :] = u[i, :]

    # Points on vertical LE line
    for i in [nLE - 1]:
        for j in [0]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = uinf * np.cos(aoa_rad)

        # Region below body
        for j in range(1, ny // 2 - 1):
            Au[j] = 1 / dy ** 2
            Bu[j] = -2 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 1 / dy ** 2
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2

        # Just before body
        for j in [ny // 2 - 1]:
            Au[j] = 1 / dy ** 2
            Bu[j] = -1 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 0
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2 - \
                     (yBB[i + 1] - 2 * yBB[i] + yBB[i - 1]) / (dx ** 2 * dy) + 0 / (2 * dx)

        # Just after body
        for j in [ny // 2]:
            Au[j] = 0
            Bu[j] = -1 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 1 / dy ** 2
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2 + \
                     (yBT[i + 1] - 2 * yBT[i] + yBT[i - 1]) / (dx ** 2 * dy) - 0 / (2 * dx)

        # Region above body
        for j in range(ny // 2 + 1, ny - 1):
            Au[j] = 1 / dy ** 2
            Bu[j] = -2 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 1 / dy ** 2
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2

        for j in [ny - 1]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = uinf * np.cos(aoa_rad)

        u[i, :] = tridiagscalar(Au, Bu, Cu, Du)
        u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
        uold[i, :] = u[i, :]

    # Points where airfoil is
    for i in range(nLE, nTE):
        for j in [0]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = uinf * np.cos(aoa_rad)

        # Region below body
        for j in range(1, ny // 2 - 1):
            Au[j] = 1 / dy ** 2
            Bu[j] = -2 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 1 / dy ** 2
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2

        # Just before body
        for j in [ny // 2 - 1]:
            Au[j] = 1 / dy ** 2
            Bu[j] = -1 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 0
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2 - \
                     (yBB[i + 1] - 2 * yBB[i] + yBB[i - 1]) / (dx ** 2 * dy)

        # Just after body
        for j in [ny // 2]:
            Au[j] = 0
            Bu[j] = -1 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 1 / dy ** 2
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2 + \
                     (yBT[i + 1] - 2 * yBT[i] + yBT[i - 1]) / (dx ** 2 * dy)

        # Region above body
        for j in range(ny // 2 + 1, ny - 1):
            Au[j] = 1 / dy ** 2
            Bu[j] = -2 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 1 / dy ** 2
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2

        for j in [ny - 1]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = uinf * np.cos(aoa_rad)

        u[i, :] = tridiagscalar(Au, Bu, Cu, Du)
        u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
        uold[i, :] = u[i, :]

    # Points downstream of airfoil
    for i in range(nTE, nx - 1):
        for j in [0]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = uinf * np.cos(aoa_rad)

        for j in range(1, ny - 1):
            Au[j] = 1 / dy ** 2
            Bu[j] = -2 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 1 / dy ** 2
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2

        for j in [ny - 1]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = uinf * np.cos(aoa_rad)

        u[i, :] = tridiagscalar(Au, Bu, Cu, Du)
        u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
        uold[i, :] = u[i, :]

    # Residual
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            residu[i, j] = np.abs(
                (Beta[i, j] ** 2) * (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx ** 2 + \
                (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy ** 2
            )

    for i in range(nLE - 2, nTE + 1):
        for j in range(ny // 2 - 2, ny // 2 + 1):
            residu[i, j] = 0

    resu = np.max(residu)
    resplotu[iteru - 1] = resu
    iteru += 1

xcp = []
cpbot = []
cptop = []

for i in range(nLE-1, nTE):
    xcp.append(x[i])
    cpbot.append(-2 * u[i, ny // 2 - 1] / uinf)
    cptop.append(-2 * u[i, ny // 2] / uinf)

TITLE_FONT_SIZE = 10
TICK_FONT_SIZE = 8

plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
plt.plot(x, yBB, '--b')
plt.title('Airfoil Profile', fontsize=TITLE_FONT_SIZE)
plt.plot(x, yBT, '-b')
plt.gca().set_aspect('equal', adjustable='box')
plt.axis([xmin, xmax, -4, 4])
plt.legend(['Bottom', 'Top'])
plt.grid()
plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

plt.savefig('../fig/project6_part2_airfoil.png', bbox_inches='tight')

plt.figure(figsize=(12, 8), dpi=200, facecolor='w', edgecolor='k')
plt.subplot(2, 2, 1)
c1 = plt.contour(x, y, u.T, 100)
plt.title('u Contours Around Airfoil', fontsize=TITLE_FONT_SIZE)
plt.xlabel('u-direction')
plt.ylabel('v-direction')
plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

norm1 = matplotlib.colors.Normalize(vmin=c1.cvalues.min(), vmax=c1.cvalues.max())
sm1 = plt.cm.ScalarMappable(norm=norm1, cmap=c1.cmap)
sm1.set_array([])
cb1 = plt.colorbar(sm1, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
cb1.ax.tick_params(labelsize=TICK_FONT_SIZE)

plt.subplot(2, 2, 2)
plt.plot(x, u[:, ny // 2 - 1], '--b')
plt.plot(x, u[:, ny // 2], '-b')
plt.title('u Perturbation Velocity Above and Below Airfoil', fontsize=TITLE_FONT_SIZE)
plt.legend(['Bottom', 'Top'])
plt.grid()
plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

plt.subplot(2, 2, 3)
plt.plot(xcp, -np.array(cptop), '-b')
plt.title('-C_P Across Airfoil', fontsize=TITLE_FONT_SIZE)
plt.xlabel('Location Along Chord')
plt.ylabel('-C_P')
plt.plot(xcp, -np.array(cpbot), '--b')
plt.legend(['C_P Top', 'C_P Bottom'])
plt.grid()
plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

plt.subplot(2, 2, 4)
plt.semilogy(resplotu[resplotu != 0], '-b')
plt.title('Residual versus Iteration Count', fontsize=TITLE_FONT_SIZE)
plt.xlabel('Number of Iterations')
plt.ylabel('Residual')
plt.grid()
plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

plt.tight_layout()
plt.savefig('../fig/project6_part2.png', bbox_inches='tight')

# Show all plots
plt.show()
