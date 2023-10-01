"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 8: Transonic flow and boundary layers
Part 2: Boundary layers
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from matplotlib import cm

sys.path.insert(1, '../..')

from tools.tridiagscalar import tridiagscalar

reynolds_num = 10000

# ny must be even
nx = 20
ny = 20

xmin = -4
xmax = 4
ymin = 0
ymax = 10 / np.sqrt(reynolds_num)

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

X, Y = np.meshgrid(x, y)

nLE = round(2 * nx // 5)
nTE = round(3 * nx // 5) + 1

# tau - thickness ratio of parabolic profiles and NACA
# sh - vertical shift for parabolic profiles points on grid
# coe - 'A' coefficient for parabolic profiles points on grid
# chord - chord length of airfoil points on grid
tau = 0.1
sh = tau * x[nTE - 1]
coe = sh / x[nLE - 1] ** 2
chord = x[nTE - 1] - x[nLE - 1]

# For diamond airfoil
diamond_slope = tau

itermax = 20
iter2max = 10

####################################################################################################

# Case 1: flat plate (default)
yB = np.zeros(nx)

# # Case 2: Biconvex
# for i in range(nLE - 1, nTE):
#     yB[i] = -coe * x[i] ** 2 + sh

# # Case 3: Diamond Airfoil (first half)
# for i in range(nLE - 1, nx // 2):
#     yB[i] = diamond_slope * x[i] + tau * chord / 2

# # Case 3: Diamond Airfoil (second half)
# for i in range(nx // 2, nTE):
#     yB[i] = -diamond_slope * x[i] + tau * chord / 2

# # NACA 00xx
# for i in range(nLE - 1, nTE):
#     yB[i] = 10 * tau * chord * ( \
#         0.2969 * np.sqrt((x[i] + chord / 2) / chord) - \
#         0.1260 * ((x[i] + chord / 2) / chord) - \
#         0.3537 * ((x[i] + chord / 2) / chord) ** 2 + \
#         0.2843 * ((x[i] + chord / 2) / chord) ** 3 - \
#         0.1015 * ((x[i] + chord / 2) / chord) ** 4)

d2yBdx2 = np.zeros(nx)

# Calculate numerical second derivative
for i in range(nLE - 1, nTE):
    d2yBdx2[i] = (yB[i + 1] - 2 * yB[i] + yB[i - 1]) / dx ** 2

####################################################################################################

u = np.ones((nx, ny))
v = np.ones((nx, ny))
w = np.ones((nx, ny))

v[0, :] = 0
w[0, :] = 0

Au = np.zeros(nx)
Bu = np.zeros(nx)
Cu = np.zeros(nx)
Du = np.zeros(nx)

Av = np.zeros(ny)
Bv = np.zeros(ny)
Cv = np.zeros(ny)
Dv = np.zeros(ny)

Aw = np.zeros(ny)
Bw = np.zeros(ny)
Cw = np.zeros(ny)
Dw = np.zeros(ny)

####################################################################################################
# Solve for U

iter2 = 0

while iter2 < iter2max:

    # Points before plate
    for i in range(1, nLE - 1):

        iterator = 0

        while iterator < itermax:
            for j in [0]:
                Au[j] = 0
                Bu[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cu[j] = 2 / dy ** 2
                Du[j] = (-1 / dx ** 2) * (u[i + 1, j] + u[i - 1, j])

                Av[j] = 0
                Bv[j] = 1
                Cv[j] = 0
                Dv[j] = 0

            for j in range(1, ny - 1):
                Au[j] = 1 / dy ** 2
                Bu[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -1 / dx ** 2 * (u[i + 1, j] + u[i - 1, j]) - \
                        1 / (2 * dy) * (w[i, j + 1] - w[i, j - 1])

                Av[j] = 1 / dy ** 2
                Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cv[j] = 1 / dy ** 2
                Dv[j] = -1 / dx ** 2 * (v[i + 1, j] + v[i - 1, j]) + \
                        1 / (2 * dx) * (w[i + 1, j] - w[i - 1, j])

            for j in [ny - 1]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = 1

                Av[j] = 2/dy ** 2;
                Bv[j] = -2/dy ** 2 - 2/dx ** 2;
                Cv[j] = 0;
                Dv[j] = (-1/dx ** 2) * (v[i + 1, j] + v[i - 1, j]) + \
                        ( 1 / (2 * dx)) * (w[i + 1,j ] - w[i - 1, j]);

            u[i, :] = tridiagscalar(Au, Bu, Cu, Du)
            v[i, :] = tridiagscalar(Av, Bv, Cv, Dv)

            for j in [0]:
                Aw[j] = 0
                Bw[j] = 1
                Cw[j] = 0
                Dw[j] = 0

            for j in range(1, ny - 1):
                Aw[j] = -1 / (reynolds_num * dy ** 2) - v[i, j] / (2 * dy)
                Bw[j] = u[i, j] / dx + 2 / (reynolds_num * dx ** 2) + 2 / (reynolds_num * dy ** 2)
                Cw[j] = v[i, j] / (2 * dy) - 1 / (reynolds_num * dy ** 2)
                Dw[j] = (w[i - 1, j] * u[i, j]) / dx + \
                        (w[i + 1, j] + w[i - 1, j]) / (reynolds_num * dx ** 2)

            for j in [ny - 1]:
                Aw[j] = 0
                Bw[j] = 1
                Cw[j] = 0
                Dw[j] = 0

            w[i, :] = tridiagscalar(Aw, Bw, Cw, Dw)
            iterator += 1

    # Points across plate
    for i in range(nLE - 1, nTE):

        iterator = 0

        while iterator < itermax:
            for j in [0]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = 0

                Av[j] = 0
                Bv[j] = 1
                Cv[j] = 0
                Dv[j] = 0

            for j in range(1, ny - 1):
                Au[j] = 1 / dy ** 2
                Bu[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -1 / dx ** 2 * (u[i + 1, j] + u[i - 1, j]) - \
                        1 / (2 * dy) * (w[i, j + 1] - w[i, j - 1])

                Av[j] = 1 / dy ** 2
                Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cv[j] = 1 / dy ** 2
                Dv[j] = -1 / dx ** 2 * (v[i + 1, j] + v[i - 1, j]) + \
                        1 / (2 * dx) * (w[i + 1, j] - w[i - 1, j])

            for j in [ny - 1]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = 1

                Av[j] = 2 / dy ** 2
                Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cv[j] = 0
                Dv[j] = -1 / dx ** 2 * (v[i + 1, j] + v[i - 1, j]) + \
                        1 / (2 * dx) * (w[i + 1, j] - w[i - 1, j])

            u[i, :] = tridiagscalar(Au, Bu, Cu, Du)
            v[i, :] = tridiagscalar(Av, Bv, Cv, Dv)

            # Boundary condition for 'w' across flat plate
            for j in [0]:
                Aw[j] = 0
                Bw[j] = 1
                Cw[j] = 0
                Dw[j] = -u[i, j + 1] / dy

            for j in range(1, ny - 1):
                Aw[j] = -1 / (reynolds_num * dy ** 2) - v[i, j] / (2 * dy)
                Bw[j] = u[i, j] / dx + 2 / (reynolds_num * dx ** 2) + 2 / (reynolds_num * dy ** 2)
                Cw[j] = v[i, j] / (2 * dy) - 1 / (reynolds_num * dy ** 2)
                Dw[j] = (w[i - 1, j] * u[i, j]) / dx + \
                        (w[i + 1, j] + w[i - 1, j]) / (reynolds_num * dx ** 2)

            for j in [ny - 1]:
                Aw[j] = 0
                Bw[j] = 1
                Cw[j] = 0
                Dw[j] = 0

            w[i, :] = tridiagscalar(Aw, Bw, Cw, Dw)
            iterator += 1

    # Trailing edge to end of grid
    for i in range(nTE, nx - 1):

        iterator = 0

        while iterator < itermax:
            for j in [0]:
                Au[j] = 0
                Bu[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cu[j] = 2 / dy ** 2
                Du[j] = -1 / dx ** 2 * (u[i + 1, j] + u[i - 1, j])

                Av[j] = 0
                Bv[j] = 1
                Cv[j] = 0
                Dv[j] = 0

            for j in range(1, ny - 1):
                Au[j] = 1 / dy ** 2
                Bu[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -1 / dx ** 2 * (u[i + 1, j] + u[i - 1, j]) - \
                        1 / (2 * dy) * (w[i, j + 1] - w[i, j - 1])

                Av[j] = 1 / dy ** 2
                Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cv[j] = 1 / dy ** 2
                Dv[j] = -1 / dx ** 2 * (v[i + 1, j] + v[i - 1, j]) + \
                        1 / (2 * dx) * (w[i + 1, j] - w[i - 1, j])

            for j in [ny - 1]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = 1

                Av[j] = 2 / dy ** 2
                Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cv[j] = 0
                Dv[j] = -1 / dx ** 2 * (v[i + 1, j] + v[i - 1, j]) + \
                        1 / (2 * dx) * (w[i + 1, j] - w[i - 1, j])

            u[i, :] = tridiagscalar(Au, Bu, Cu, Du)
            v[i, :] = tridiagscalar(Av, Bv, Cv, Dv)

            for j in [0]:
                Aw[j] = 0
                Bw[j] = 1
                Cw[j] = 0
                Dw[j] = 0

            for j in range(1, ny - 1):
                Aw[j] = -1 / (reynolds_num * dy ** 2) - v[i, j] / (2 * dy)
                Bw[j] = u[i, j] / dx + 2 / (reynolds_num * dx ** 2) + 2 / (reynolds_num * dy ** 2)
                Cw[j] = v[i, j] / (2 * dy) - 1 / (reynolds_num * dy ** 2)
                Dw[j] = (w[i - 1, j] * u[i, j]) / dx + (w[i + 1, j] + w[i - 1, j]) / (reynolds_num * dx ** 2)

            for j in [ny - 1]:
                Aw[j] = 0
                Bw[j] = 1
                Cw[j] = 0
                Dw[j] = 0

            w[i, :] = tridiagscalar(Aw, Bw, Cw, Dw)
            iterator += 1

    # End of grid
    for i in [nx - 1]:

        iterator = 0

        while iterator < itermax:

            for j in [0]:
                Au[j] = 0
                Bu[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cu[j] = 2 / dy ** 2
                Du[j] = -1 / dx ** 2 * (2 * u[i - 1, j])

                Av[j] = 0
                Bv[j] = 1
                Cv[j] = 0
                Dv[j] = 0

            for j in range(1, ny - 1):
                Au[j] = 1 / dy ** 2
                Bu[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cu[j] = 1 / dy ** 2
                Du[j] = -1 / dx ** 2 * (2 * u[i - 1, j]) - \
                        1 / (2 * dy) * (w[i, j + 1] - w[i, j - 1])

                Av[j] = 1 / dy ** 2
                Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cv[j] = 1 / dy ** 2
                Dv[j] = -1 / dx ** 2 * (2 * v[i - 1, j])

            for j in [ny - 1]:
                Au[j] = 0
                Bu[j] = 1
                Cu[j] = 0
                Du[j] = 1

                Av[j] = 2 / dy ** 2
                Bv[j] = -2 / dy ** 2 - 2 / dx ** 2
                Cv[j] = 0
                Dv[j] = -1 / dx ** 2 * (2 * v[i - 1, j])

            u[i, :] = tridiagscalar(Au, Bu, Cu, Du)
            v[i, :] = tridiagscalar(Av, Bv, Cv, Dv)

            for j in [0]:
                Aw[j] = 0
                Bw[j] = 1
                Cw[j] = 0
                Dw[j] = 0

            for j in range(1, ny - 1):
                Aw[j] = -1 / (reynolds_num * dy ** 2) - v[i, j] / (2 * dy)
                Bw[j] = u[i, j] / dx + 2 / (reynolds_num * dx ** 2) + 2 / (reynolds_num * dy ** 2)
                Cw[j] = v[i, j] / (2 * dy) - 1 / (reynolds_num * dy ** 2)
                Dw[j] = (w[i - 1, j] * u[i, j]) / dx + (2 * w[i - 1, j]) / (reynolds_num * dx ** 2)

            for j in [ny - 1]:
                Aw[j] = 0
                Bw[j] = 1
                Cw[j] = 0
                Dw[j] = 0

            w[i, :] = tridiagscalar(Aw, Bw, Cw, Dw)
            iterator += 1

    iter2 += 1

####################################################################################################

# Plot results
fig = plt.figure(figsize=(10, 6), dpi=200, facecolor='w', edgecolor='k')

TITLE_FONT_SIZE = 10
TICK_FONT_SIZE = 8

plt.subplot(3, 1, 1)
c1 = plt.contour(x, y, u.T, 200, cmap=cm.jet)
plt.plot(x, yB, '-k', zorder=11)
plt.fill_between(x, yB, color='r', zorder=10)
plt.title('u Countours Around Airfoil', fontsize=TITLE_FONT_SIZE)
plt.xlabel('u-direction')
plt.ylabel('v-direction')
plt.grid(True)
plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

norm1 = matplotlib.colors.Normalize(vmin=c1.cvalues.min(), vmax=c1.cvalues.max())
sm1 = plt.cm.ScalarMappable(norm=norm1, cmap=c1.cmap)
sm1.set_array([])
cb1 = plt.colorbar(sm1, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
cb1.ax.tick_params(labelsize=TICK_FONT_SIZE)

plt.subplot(3, 1, 2)
c2 = plt.contour(x, y, v.T, 200, cmap=cm.jet)
plt.plot(x, yB, '-k', zorder=11)
plt.fill_between(x, yB, color='r', zorder=10)
plt.title('v Countours Around Airfoil', fontsize=TITLE_FONT_SIZE)
plt.xlabel('u-direction')
plt.ylabel('v-direction')
plt.grid(True)
plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

norm2 = matplotlib.colors.Normalize(vmin=c2.cvalues.min(), vmax=c2.cvalues.max())
sm2 = plt.cm.ScalarMappable(norm=norm2, cmap=c2.cmap)
sm2.set_array([])
cb2 = plt.colorbar(sm2, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
cb2.ax.tick_params(labelsize=TICK_FONT_SIZE)

plt.subplot(3, 1, 3)
c3 = plt.contour(x, y, w.T, 200, cmap=cm.jet)
plt.plot(x, yB, '-k', zorder=11)
plt.fill_between(x, yB, color='r', zorder=10)
plt.title('w Countours Around Airfoil', fontsize=TITLE_FONT_SIZE)
plt.xlabel('u-direction')
plt.ylabel('v-direction')
plt.grid(True)
plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

norm3 = matplotlib.colors.Normalize(vmin=c3.cvalues.min(), vmax=c3.cvalues.max())
sm3 = plt.cm.ScalarMappable(norm=norm3, cmap=c3.cmap)
sm3.set_array([])
cb3 = plt.colorbar(sm1, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
cb3.ax.tick_params(labelsize=TICK_FONT_SIZE)

plt.tight_layout()
plt.savefig(f'../fig/project8_boundary.png', bbox_inches='tight')

# Show all plots
plt.show()
