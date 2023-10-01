"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 6: Steady, inviscid, adiabatic, compressible, and irrotational flows over airfoils -
    numerical solutions to lifting Problem
Part 3: Cross flow problem
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, '../..')

from tools.tridiagscalar import tridiagscalar

aoa_deg = 5
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

chord = (x[nTE] - x[nLE])

mach_num = 0

# Flat plate
yBT = np.zeros(nx)
yBB = np.zeros(nx)

Beta = np.sqrt(1 - mach_num ** 2) * np.ones((nx, ny))

u = np.zeros((nx, ny))
uold = np.zeros((nx, ny))
v = np.zeros((nx, ny))
vold = np.zeros((nx, ny))

# Initialize u and v
for i in [0]:
    for j in range(ny):
        v[i, j] = uinf * np.sin(aoa_rad)
        vold[i, j] = uinf * np.sin(aoa_rad)

for i in [nx - 1]:
    for j in range(ny):
        v[i, j] = uinf * np.sin(aoa_rad)
        vold[i, j] = uinf * np.sin(aoa_rad)

for i in range(nx):
    for j in [0]:
        v[i, j] = uinf * np.sin(aoa_rad)
        vold[i, j] = uinf * np.sin(aoa_rad)

    for j in [ny - 1]:
        v[i, j] = uinf * np.sin(aoa_rad)
        vold[i, j] = uinf * np.sin(aoa_rad)

resu = 1
iteru = 1
resv = 1
iterv = 1

residu = np.zeros((nx, ny))
resplotu = np.zeros(maxiter)

residv = np.zeros((nx, ny))
resplotv = np.zeros(maxiter)

Au = np.zeros(ny)
Bu = np.zeros(ny)
Cu = np.zeros(ny)
Du = np.zeros(ny)

Av = np.zeros(ny)
Bv = np.zeros(ny)
Cv = np.zeros(ny)
Dv = np.zeros(ny)

####################################################################################################
# U-solving loop

while iteru < maxiter and resu > resmax:
    # Points upstream of airfoil
    for i in range(1, nLE - 1):
        for j in [0]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = 0

        for j in range(1, ny - 1):
            Au[j] = 1 / dy ** 2
            Bu[j] = -2 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 1 / dy ** 2
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2

        for j in [ny - 1]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = 0

        u[i, :] = tridiagscalar(Au, Bu, Cu, Du)
        u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
        uold[i, :] = u[i, :]

    # Points on vertical LE line
    for i in [nLE - 1]:
        for j in [0]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = 0

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
                        (yBB[i + 1] - 2 * yBB[i] + yBB[i - 1]) / (dx ** 2 * dy) + 0.15 / (2 * dy)

        # Just after body
        for j in [ny // 2]:
            Au[j] = 0
            Bu[j] = -1 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 1 / dy ** 2
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2 + \
                        (yBT[i + 1] - 2 * yBT[i] + yBT[i - 1]) / (dx ** 2 * dy) - 0.15 / (2 * dy)

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
            Du[j] = 0

        u[i, :] = tridiagscalar(Au, Bu, Cu, Du)
        u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
        uold[i, :] = u[i, :]

    # Points where airfoil is
    for i in range(nLE, nTE - 1):
        for j in [0]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = 0

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
            Du[j] = 0

        u[i, :] = tridiagscalar(Au, Bu, Cu, Du)
        u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
        uold[i, :] = u[i, :]

    # Points on vertical TE line
    for i in [nTE - 1]:
        for j in [0]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = 0

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
                        (yBB[i + 1] - 2 * yBB[i] + yBB[i - 1]) / (dx ** 2 * dy) - 0.15 / (2 * dy)

        # Just after body
        for j in [ny // 2]:
            Au[j] = 0
            Bu[j] = -1 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 1 / dy ** 2
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2 + \
                        (yBT[i + 1] - 2 * yBT[i] + yBT[i - 1]) / (dx ** 2 * dy) + 0.15 / (2 * dy)

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
            Du[j] = 0

        u[i, :] = tridiagscalar(Au, Bu, Cu, Du)
        u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
        uold[i, :] = u[i, :]

    # Points downstream of airfoil
    for i in range(nTE, nx - 1):
        for j in [0]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = 0

        for j in range(1, ny - 1):
            Au[j] = 1 / dy ** 2
            Bu[j] = -2 / dy ** 2 - 2 * (Beta[i, j] ** 2) / dx ** 2
            Cu[j] = 1 / dy ** 2
            Du[j] = -(Beta[i, j] ** 2) * (uold[i - 1, j] + uold[i + 1, j]) / dx ** 2

        for j in [ny - 1]:
            Au[j] = 0
            Bu[j] = 1
            Cu[j] = 0
            Du[j] = 0

        u[i, :] = tridiagscalar(Au, Bu, Cu, Du)
        u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
        uold[i, :] = u[i, :]

    # Residual calculation
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            residu[i, j] = abs( \
                (Beta[i, j] ** 2) * (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx ** 2 + \
                (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy ** 2)

    for i in range(nLE-2, nTE + 1):
        for j in range(ny // 2 - 2, ny // 2 + 2):
            residu[i, j] = 0

    # Update residual and iteration counter
    resu = np.max(residu)
    resplotu[iteru] = resu;
    iteru += 1

####################################################################################################
# V-solving loop

while iterv < maxiter and resv > resmax:
    # Points upstream of airfoil
    for i in range(1, nLE-2):
        for j in [0]:
            Av[j] = 0
            Bv[j] = 1
            Cv[j] = 0
            Dv[j] = uinf * np.sin(aoa_rad)

        for j in range(1, ny - 1):
            Av[j] = 1 / dy ** 2
            Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
            Cv[j] = 1 / dy ** 2
            Dv[j] = -(vold[i - 1, j] + vold[i + 1, j]) / dx ** 2

        for j in [ny - 1]:
            Av[j] = 0
            Bv[j] = 1
            Cv[j] = 0
            Dv[j] = uinf * np.sin(aoa_rad)

        v[i, :] = tridiagscalar(Av, Bv, Cv, Dv)
        v[i, :] = vold[i, :] + omega * (v[i, :] - vold[i, :])
        vold[i, :] = v[i, :]

    # Line just before LE
    for i in [nLE - 2]:
        for j in [0]:
            Av[j] = 0
            Bv[j] = 1
            Cv[j] = 0
            Dv[j] = uinf * np.sin(aoa_rad)

        for j in range(1, ny // 2 - 1):
            Av[j] = 1 / dy ** 2
            Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
            Cv[j] = 1 / dy ** 2
            Dv[j] = -(vold[i - 1, j] + vold[i + 1, j]) / dx ** 2

        for j in [ny // 2 - 1]:
            Av[j] = 0
            Bv[j] = 1
            Cv[j] = 0
            Dv[j] = 0.15

        for j in range(ny // 2, ny - 1):
            Av[j] = 1 / dy ** 2
            Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
            Cv[j] = 1 / dy ** 2
            Dv[j] = -(vold[i - 1, j] + vold[i + 1, j]) / dx ** 2

        for j in [ny - 1]:
            Av[j] = 0
            Bv[j] = 1
            Cv[j] = 0
            Dv[j] = uinf * np.sin(aoa_rad)

        v[i, :] = tridiagscalar(Av, Bv, Cv, Dv)
        v[i, :] = vold[i, :] + omega * (v[i, :] - vold[i, :])
        vold[i, :] = v[i, :]

    # Points across airfoil
    for i in range(nLE - 1, nTE):
        for j in [0]:
            Av[j] = 0
            Bv[j] = 1
            Cv[j] = 0
            Dv[j] = uinf * np.sin(aoa_rad)

        # Region below body
        for j in range(1, ny // 2 - 1):
            Av[j] = 1 / dy ** 2
            Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
            Cv[j] = 1 / dy ** 2
            Dv[j] = -(vold[i - 1, j] + vold[i + 1, j]) / dx ** 2

        # The body
        for j in [ny // 2 - 1]:
            Av[j] = 0
            Bv[j] = 1
            Cv[j] = 0
            Dv[j] = 0

        # Region above the body
        for j in range(ny // 2, ny - 1):
            Av[j] = 1 / dy ** 2
            Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
            Cv[j] = 1 / dy ** 2
            Dv[j] = -(vold[i - 1, j] + vold[i + 1, j]) / dx ** 2

        for j in [ny - 1]:
            Av[j] = 0
            Bv[j] = 1
            Cv[j] = 0
            Dv[j] = uinf * np.sin(aoa_rad)

        v[i, :] = tridiagscalar(Av, Bv, Cv, Dv)
        v[i, :] = vold[i, :] + omega * (v[i, :] - vold[i, :])
        vold[i, :] = v[i, :]

    # Line just after TE
    for i in [nTE]:
        for j in [0]:
            Av[j] = 0
            Bv[j] = 1
            Cv[j] = 0
            Dv[j] = uinf * np.sin(aoa_rad)

        for j in range(1, ny // 2 - 1):
            Av[j] = 1 / dy ** 2
            Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
            Cv[j] = 1 / dy ** 2
            Dv[j] = -(vold[i - 1, j] + vold[i + 1, j]) / dx ** 2

        for j in [ny // 2 - 1]:
            Av[j] = 0
            Bv[j] = 1
            Cv[j] = 0
            Dv[j] = 0.15

        for j in range(ny // 2, ny - 1):
            Av[j] = 1 / dy ** 2
            Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
            Cv[j] = 1 / dy ** 2
            Dv[j] = -(vold[i - 1, j] + vold[i + 1, j]) / dx ** 2

        for j in [ny - 1]:
            Av[j] = 0
            Bv[j] = 1
            Cv[j] = 0
            Dv[j] = uinf * np.sin(aoa_rad)

        v[i, :] = tridiagscalar(Av, Bv, Cv, Dv)
        v[i, :] = vold[i, :] + omega * (v[i, :] - vold[i, :])
        vold[i, :] = v[i, :]

    # Points downstream of airfoil
    for i in range(nTE + 1, nx - 1):
        for j in [0]:
            Av[j] = 0
            Bv[j] = 1
            Cv[j] = 0
            Dv[j] = uinf * np.sin(aoa_rad)

        for j in range(1, ny - 1):
            Av[j] = 1 / dy ** 2
            Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
            Cv[j] = 1 / dy ** 2
            Dv[j] = -(vold[i - 1, j] + vold[i + 1, j]) / dx ** 2

        for j in [ny - 1]:
            Av[j] = 0
            Bv[j] = 1
            Cv[j] = 0
            Dv[j] = uinf * np.sin(aoa_rad)

        v[i, :] = tridiagscalar(Av, Bv, Cv, Dv)
        v[i, :] = vold[i, :] + omega * (v[i, :] - vold[i, :])
        vold[i, :] = v[i, :]

    # Residual
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            residv[i, j] = abs( \
                (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx**2 + \
                (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / dy**2)

    for i in range(nLE-2, nTE + 1):
        for j in range(ny // 2 - 2, ny // 2 + 2):
            residv[i, j] = 0

    # Update residual and iteration counter
    resv = np.max(residv)
    resplotv[iterv] = resv;
    iterv += 1

####################################################################################################

xcp = []
cpbot = []
cptop = []

for i in range(nLE-1, nTE):
    xcp.append(x[i])
    cpbot.append(-2 * u[i, ny // 2 - 1] / uinf)
    cptop.append(-2 * u[i, ny // 2] / uinf)

####################################################################################################

TITLE_FONT_SIZE = 10
TICK_FONT_SIZE = 8

# Create Figure 1 with subplots
fig1, axs = plt.subplots(3, 2, figsize=(12, 8), dpi=200, facecolor='w', edgecolor='k')

# Plot 3: u countours around airfoil
c3 = axs[1, 0].contour(x, y, u.T, 100)
axs[1, 0].set_title(f'u Countours Around Airfoil AOA: {aoa_deg} deg.', fontsize=TITLE_FONT_SIZE)
axs[1, 0].set_xlabel('u-direction')
axs[1, 0].set_ylabel('v-direction')

norm3 = matplotlib.colors.Normalize(vmin=c3.cvalues.min(), vmax=c3.cvalues.max())
sm3 = plt.cm.ScalarMappable(norm=norm3, cmap = c3.cmap)
sm3.set_array([])

# Set the colorbar to the first subplot (for spacing only), remove it, and set the colorbar on
# the intended subplot.
cb1 = plt.colorbar(sm3, ax=axs[0, 0], orientation='vertical', fraction=0.046, pad=0.04)
cb1.remove()
cb3 = plt.colorbar(sm3, ax=axs[1, 0], orientation='vertical', fraction=0.046, pad=0.04)
cb3.ax.tick_params(labelsize=TICK_FONT_SIZE)

# Plot 1: residual versus iteration count
axs[0, 0].semilogy(resplotv[resplotv != 0], '--b', label='v')
axs[0, 0].semilogy(resplotu[resplotu != 0], '-b', label='u')
axs[0, 0].set_title('Residual versus Iteration Count', fontsize=TITLE_FONT_SIZE)
axs[0, 0].set_xlabel('Number of Iterations')
axs[0, 0].set_ylabel('Residual')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot 2: -C_P across airfoil
axs[0, 1].plot(xcp, -np.array(cptop), '-b', label='Top')
axs[0, 1].plot(xcp, -np.array(cpbot), '--b', label='Bottom')
axs[0, 1].set_title('-C_P Across Airfoil', fontsize=TITLE_FONT_SIZE)
axs[0, 1].set_xlabel('Location Along Chord')
axs[0, 1].set_ylabel('-C_P')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot 4: u perturbation velocity above and below airfoil
axs[1, 1].plot(x, u[:, ny // 2 - 1], '--b', label='Bottom')
axs[1, 1].plot(x, u[:, ny // 2], '-b', label='Top')
axs[1, 1].set_title('u Perturbation Velocity Above and Below Airfoil', fontsize=TITLE_FONT_SIZE)
axs[1, 1].legend()
axs[1, 1].grid(True)

# Plot 5: v contours around airfoil
c5 = axs[2, 0].contour(x, y, v.T, 100)
axs[2, 0].set_title(f'v Contours Around Airfoil AOA: {aoa_deg} deg.', fontsize=TITLE_FONT_SIZE)
axs[2, 0].set_xlabel('u-direction')
axs[2, 0].set_ylabel('v-direction')

norm5 = matplotlib.colors.Normalize(vmin=c5.cvalues.min(), vmax=c5.cvalues.max())
sm5 = plt.cm.ScalarMappable(norm=norm5, cmap = c5.cmap)
sm5.set_array([])
cb5 = plt.colorbar(sm5, ax=axs[2, 0], orientation='vertical', fraction=0.046, pad=0.04)
cb5.ax.tick_params(labelsize=TICK_FONT_SIZE)

# Plot 6: v perturbation velocity above and below airfoil
axs[2, 1].plot(x, v[:, ny // 2 - 1], '--b', label='Bottom')
axs[2, 1].plot(x, v[:, ny // 2], '-b', label='Top')
axs[2, 1].set_title('v Perturbation Velocity Above and Below Airfoil', fontsize=TITLE_FONT_SIZE)
axs[2, 1].legend()
axs[2, 1].grid(True)

plt.tight_layout()
plt.savefig('../fig/project6_part3.png', bbox_inches='tight')

# Show all plots
plt.show()
