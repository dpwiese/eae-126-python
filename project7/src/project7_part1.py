"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 7: Steady, inviscid, adiabatic, compressible, and irrotational 2D flows over airfoils -
    numerical solutions: supersonic
Part 1: Supersonic flow over airfoils (Nonlinear Mach number)
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
rhoinf = 1

# Nonlinear Mach number: set gamma = 1.4 and iteratormax sufficiently large
# Linear Mach number: set gamma = -1 and iteratormax = 1
gamma = 1.4
iteratormax = 100

# ny must be even
nx = 200
ny = 100

xmin = -10
xmax = 10
ymin = -20
ymax = 20

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

nLE = round(2 * nx // 5)
nTE = round(3 * nx // 5) + 1

# tau - thickness ratio of parabolic profiles and NACA
# sh - vertical shift for parabolic profiles
# coe - 'A' coefficient for parabolic profiles
# chord - chordlength of airfoil points on grid
tau = 0.1
sh = tau * x[nTE - 1]
coe = sh / x[nLE - 1] ** 2
chord = (x[nTE - 1] - x[nLE - 1])

# For diamond airfoil
diamond_slope = tau

# Initialize arrays
BS = np.zeros((nx, ny))
u = np.zeros((nx, ny))
Ma = np.zeros((nx, ny))

# Case 1: (default) flat plate
yBT = np.zeros(nx)
yBB = np.zeros(nx)

# # Case 2: Biconvex
# for i in range(nLE-1, nTE):
#     yBT[i] = -coe * x[i] ** 2 + sh
#     yBB[i] = coe * x[i] ** 2 + sh

# # Case 3: diamond airfoil (first half)
# for i in range(nLE - 1, nx // 2):
#     yBT[i] = diamond_slope * x[i] + tau * chord / 2
#     yBB[i] = -diamond_slope * x[i] - tau * chord / 2

# # Case 3: diamond airfoil (second half)
# for i in range(nx // 2, nTE):
#     yBT[i] = -diamond_slope * x[i] + tau * chord / 2
#     yBB[i] = diamond_slope * x[i] - tau * chord / 2

# Initialize arrays to store d2yBBdx2 and d2yBTdx2
d2yBBdx2 = np.zeros(nx)
d2yBTdx2 = np.zeros(nx)

# Calculate d2yBBdx2
for i in [nLE - 1]:
    d2yBBdx2[i] = (yBB[i + 2] - 2 * yBB[i + 1] + yBB[i]) / dx ** 2
    d2yBTdx2[i] = (yBT[i + 2] - 2 * yBT[i + 1] + yBT[i]) / dx ** 2

for i in range(nLE, nTE - 1):
    d2yBBdx2[i] = (yBB[i + 1] - 2 * yBB[i] + yBB[i - 1]) / dx ** 2
    d2yBTdx2[i] = (yBT[i + 1] - 2 * yBT[i] + yBT[i - 1]) / dx ** 2

for i in [nTE - 1]:
    d2yBBdx2[i] = (yBB[i] - 2 * yBB[i - 1] + yBB[i - 2]) / dx ** 2
    d2yBTdx2[i] = (yBT[i] - 2 * yBT[i - 1] + yBT[i - 2]) / dx ** 2

Au = np.zeros(ny)
Bu = np.zeros(ny)
Cu = np.zeros(ny)
Du = np.zeros(ny)

# TODO@dpwiese - relationship between Mach number and uinf?
mach_num_array = [1.6]

# Main loop
for idx, mach_num in enumerate(mach_num_array):
    Minf = mach_num

    # Initialize BS
    for i in range(nx):
        for j in range(ny):
            BS[i, j] = (1 - Minf ** 2)

    # Initialize u
    for i in range(2):
        for j in range(ny):
            u[i, j] = uinf * np.cos(aoa_rad)

    for i in range(nx):
        u[i, 0] = uinf * np.cos(aoa_rad)
        u[i, ny - 1] = uinf * np.cos(aoa_rad)

    # Points upstream of airfoil
    for i in range(2, nLE - 1):
        iterator = 0
        while iterator < iteratormax:
            for j in [0]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = uinf * np.cos(aoa_rad)

            for j in range(1, ny - 1):
                Au[j] = 1 / dy ** 2
                Bu[j] = -2 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2

            for j in [ny - 1]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = uinf * np.cos(aoa_rad)

            u[i, :] = tridiagscalar(Au, Bu, Cu, Du)

            # Update Ma and BS
            for j in range(ny):
                Ma[i, j] = np.sqrt(Minf ** 2 + (gamma + 1) * Minf ** 2 * (u[i, j] - uinf))
                BS[i, j] = (1 - Ma[i, j] ** 2)

            iterator += 1

    # Points on vertical LE line
    for i in [nLE - 1]:
        iterator = 0
        while iterator < iteratormax:
            for j in [0]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = uinf * np.cos(aoa_rad)

            # Region below body
            for j in range(1, ny // 2 - 1):
                Au[j] = 1 / dy ** 2
                Bu[j] = -2 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2

            # Just before body Bottom
            for j in [ny // 2 - 1]:
                Au[j] = 1 / dy ** 2
                Bu[j] = -1 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 0
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2 + \
                            uinf * np.sin(aoa_rad) / (dx * dy) - \
                            (yBB[nLE] - yBB[nLE - 1]) / (dy * dx ** 2)

            # Just after body Top
            for j in [ny // 2]:
                Au[j] = 0
                Bu[j] = -1 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2 - \
                            uinf * np.sin(aoa_rad) / (dx * dy) + \
                            (yBT[nLE] - yBT[nLE - 1]) / (dy * dx ** 2)

            # Region above body
            for j in range(ny // 2 + 1, ny - 1):
                Au[j] = 1 / dy ** 2
                Bu[j] = -2 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2

            for j in [ny - 1]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = uinf * np.cos(aoa_rad)

            u[i, :] = tridiagscalar(Au, Bu, Cu, Du)

            # Update Ma and BS
            for j in range(ny):
                Ma[i, j] = np.sqrt(Minf ** 2 + (gamma + 1) * Minf ** 2 * (u[i, j] - uinf))
                BS[i, j] = (1 - Ma[i, j] ** 2)

            iterator += 1

    # Points where airfoil is
    for i in range(nLE, nTE - 1):
        iterator = 0
        while iterator < iteratormax:
            for j in [0]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = uinf * np.cos(aoa_rad)

            # Region below body
            for j in range(1, ny // 2 - 1):
                Au[j] = 1 / dy ** 2
                Bu[j] = -2 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2

            # Just before body Bottom
            for j in [ny // 2 - 1]:
                Au[j] = 1 / dy ** 2
                Bu[j] = -1 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 0
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2 - d2yBBdx2[i] / dy

            # Just after body Top
            for j in [ny // 2]:
                Au[j] = 0
                Bu[j] = -1 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2 + d2yBTdx2[i] / dy

            # Region above body
            for j in range(ny // 2 + 1, ny - 1):
                Au[j] = 1 / dy ** 2
                Bu[j] = -2 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2

            for j in [ny - 1]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = uinf * np.cos(aoa_rad)

            u[i, :] = tridiagscalar(Au, Bu, Cu, Du)

            # Update Ma and BS
            for j in range(ny):
                Ma[i, j] = np.sqrt(Minf ** 2 + (gamma + 1) * Minf ** 2 * (u[i, j] - uinf))
                BS[i, j] = (1 - Ma[i, j] ** 2)

            iterator += 1

    # Points on vertical TE line
    for i in [nTE - 1]:
        iterator = 0
        while iterator < iteratormax:
            for j in [0]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = uinf * np.cos(aoa_rad)

            # Region below body
            for j in range(1, ny // 2 - 1):
                Au[j] = 1 / dy ** 2
                Bu[j] = -2 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2

            # Just before body
            for j in [ny // 2 - 1]:
                Au[j] = 1 / dy ** 2
                Bu[j] = -1 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 0
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2 - \
                            uinf * np.sin(aoa_rad) / (dx * dy) + \
                            (yBB[nTE - 1] - yBB[nTE - 2]) / (dy * dx ** 2)

            # Just after body
            for j in [ny // 2]:
                Au[j] = 0
                Bu[j] = -1 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2 + \
                            uinf * np.sin(aoa_rad) / (dx * dy) - \
                            (yBT[nTE - 1] - yBT[nTE - 2]) / (dy * dx ** 2)

            # Region above body
            for j in range(ny // 2 + 1, ny - 1):
                Au[j] = 1 / dy ** 2
                Bu[j] = -2 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2

            for j in [ny - 1]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = uinf * np.cos(aoa_rad)

            u[i, :] = tridiagscalar(Au, Bu, Cu, Du)

            # Update Ma and BS
            for j in range(ny):
                Ma[i, j] = np.sqrt(Minf ** 2 + (gamma + 1) * Minf ** 2 * (u[i, j] - uinf))
                BS[i, j] = (1 - Ma[i, j] ** 2)

            iterator += 1

    # Points downstream of airfoil
    for i in range(nTE, nx):
        iterator = 0
        while iterator < iteratormax:
            for j in [0]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = uinf * np.cos(aoa_rad)

            # Region below body
            for j in range(1, ny - 1):
                Au[j] = 1 / dy ** 2
                Bu[j] = -2 / dy ** 2 + BS[i, j] / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -BS[i, j] * (-2 * u[i - 1, j] + u[i - 2, j]) / dx ** 2

            for j in [ny - 1]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = uinf * np.cos(aoa_rad)

            u[i, :] = tridiagscalar(Au, Bu, Cu, Du)

            # Update Ma and BS
            for j in range(ny):
                Ma[i, j] = np.sqrt(Minf ** 2 + (gamma + 1) * Minf ** 2 * (u[i, j] - uinf))
                BS[i, j] = (1 - Ma[i, j] ** 2)

            iterator += 1

    # Calculate lift and other aerodynamic parameters
    xcp = np.zeros(nTE - nLE + 1)
    cpbot = np.zeros(nTE - nLE + 1)
    cptop = np.zeros(nTE - nLE + 1)
    cm = np.zeros(len(xcp))

    for i in range(nLE - 1, nTE):
        xcp[i - nLE + 1] = x[i];
        cpbot[i - nLE + 1] = -2 * (u[i, ny // 2 - 1] - uinf) / uinf;
        cptop[i - nLE + 1] = -2 * (u[i, ny // 2] - uinf) / uinf;

    uBot = 0
    uTop = 0

    for i in range(nLE, nTE):
        # Just before body
        j = ny // 2 - 1
        uBot += u[i, j]

        # Just after body
        j = ny // 2
        uTop += u[i, j]

    CM = 0

    for i in range(len(xcp)):
        cm[i] = (xcp[i] - x[nLE]) * (cptop[i] - cpbot[i])
        CM += cm[i]

    Gamma = dx * (uBot - uTop)
    LIFT = rhoinf * uinf * Gamma
    CL = LIFT / (0.5 * rhoinf * uinf ** 2)
    XCP = CM / CL

    # Set string to put in plots as to whether linear or nonlinear mach number was used
    linear_or_nonlinear_mach = 'Nonlinear'
    if gamma == -1:
        linear_or_nonlinear_mach = 'Linear'

    # Plot results
    fig = plt.figure(figsize=(10, 6), dpi=200, facecolor='w', edgecolor='k')

    TITLE_FONT_SIZE = 10
    TICK_FONT_SIZE = 8

    plt.subplot(2, 2, 1)
    c1 = plt.contour(x, y, u.T, 100)
    plt.title(f'u Countours - {linear_or_nonlinear_mach} Mach: {mach_num}, AOA: {aoa_deg} deg.', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('u-direction')
    plt.ylabel('v-direction')
    plt.grid(True)
    plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

    norm1 = matplotlib.colors.Normalize(vmin=c1.cvalues.min(), vmax=c1.cvalues.max())
    sm1 = plt.cm.ScalarMappable(norm=norm1, cmap=c1.cmap)
    sm1.set_array([])
    cb1 = plt.colorbar(sm1, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=TICK_FONT_SIZE)

    plt.subplot(2, 2, 2)
    plt.plot(x, u[:, ny // 2 - 1], '--b')
    plt.plot(x, u[:, ny // 2], '-b')
    plt.title('u Velocity Above and Below Airfoil', fontsize=TITLE_FONT_SIZE)
    plt.legend(['Bottom', 'Top'])
    plt.grid(True)
    plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

    plt.subplot(2, 2, 3)
    plt.plot(xcp, -cpbot, '--b')
    plt.plot(xcp, -cptop, '-b')
    plt.title('-C_P Across Airfoil', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Location along chord')
    plt.ylabel('-Cp')
    plt.legend(['Bottom', 'Top'])
    plt.grid(True)
    plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

    plt.subplot(2, 2, 4)
    plt.plot(x, yBB, '--b')
    plt.title('Airfoil Profile', fontsize=TITLE_FONT_SIZE)
    plt.plot(x, yBT, '-b')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([-4, 4, -1, 1])
    plt.legend(['Bottom', 'Top'])
    plt.grid(True)
    plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

    plt.tight_layout()
    plt.savefig(f'../fig/project7_part1_{linear_or_nonlinear_mach.lower()}_{idx}.png', bbox_inches='tight')

# Show all plots
plt.show()
