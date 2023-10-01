"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 5: Steady, inviscid, adiabatic, compressible, and irrotational flows over airfoils -
    numerical solutions to thickness problem
Part 3: 3D rectangular wing with symmetric profile at zero angle of attack

Profiles: biconvex and NACA 0012
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import cm

sys.path.insert(1, '../..')

from tools.tridiagscalar import tridiagscalar

uinf = 1
mach_num = 0.5

# Grid - ny must be an even number
nx = 40
ny = 40
nz = 40

xmin = -10
xmax = 10
ymin = -10
ymax = 10
zmin = -10
zmax = 10

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)
dz = (zmax - zmin) / (nz - 1)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
z = np.linspace(zmin, zmax, nz)

n_left  = round(2 * nz / 5)
n_right = round(3 * nz / 5) + 1

nLE = np.zeros(nz, dtype=int)
nTE = np.zeros(nz, dtype=int)

for k in range(0, n_left-1):
    nLE[k] = 1
    nTE[k] = 1

for k in range(n_left - 1, n_right):
    nLE[k] = round(2 * nx / 5)
    nTE[k] = round(3 * nx / 5) + 1

for k in range(n_right, nz):
    nLE[k] = 1
    nTE[k] = 1

maxiterator = 300
omega = 1.95
resmax = 1e-6

# tau - thickness ratio of parabolic profiles and NACA
# sh - vertical shift for parabolic profiles
# coe - 'A' coefficient for parabolic profiles
# chord - chordlength of airfoil points on grid
tau = 0.06
sh = tau * (x[nTE[nz // 2 - 1] - 1] + dx / 2)
coe = sh / (x[nLE[nz // 2 - 1] - 1] ** 2)
chord = x[nTE[nz // 2 - 1] - 1] - x[nLE[nz // 2 - 1] - 1]

# Case 1: Flat Plate (default)
yBT = np.zeros((nx, nz))
yBB = np.zeros((nx, nz))

# # Case 2: Cambered plate
# for k in range(n_left - 1, n_right):
#     for i in range(nLE[k], nTE[k]):
#         yBT[i, k] = -coe * x[i]**2 + sh
#         yBB[i, k] = -coe * x[i]**2 + sh

# Case 3: Biconvex
for k in range(n_left - 1, n_right):
    for i in range(nLE[k] - 1, nTE[k]):
        yBT[i, k] = -coe * x[i]**2 + sh
        yBB[i, k] = coe * x[i]**2 - sh

# # Case 4: NACA 00xx
# for k in range(n_left - 1, n_right):
#     for i in range(nLE[k], nTE[k]):
#         yBT[i, k] = 10 * tau * chord * ( \
#             0.2969 * np.sqrt((x[i] + chord / 2) / chord) - \
#             0.1260 * ((x[i] + chord / 2) / chord) - \
#             0.3537 * ((x[i] + chord / 2) / chord) ** 2 + \
#             0.2843 * ((x[i] + chord / 2) / chord) ** 3 - \
#             0.1015 * ((x[i] + chord / 2) / chord) ** 4)
#         yBB[i, k] = -yBT[i, k]

Beta = np.sqrt(1 - mach_num ** 2) * np.ones((nx, ny, nz))

u = np.zeros((nx, ny, nz))
uold = np.zeros((nx, ny, nz))

res = 1
iterator = 0
resplot = np.zeros(maxiterator)

A = np.zeros(ny)
B = np.ones(ny)
C = np.zeros(ny)
D = np.zeros(ny)

####################################################################################################

while iterator < maxiterator and res > resmax:

    # Planes before left wingtip
    for k in range(1, n_left - 1):
        for i in range(1, nx - 1):
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, ny - 1):
                A[j] = 1 / dy**2
                B[j] = -2 / dy**2 - 2 / dx**2 - 2 / dz**2
                C[j] = 1 / dy**2
                D[j] = -(uold[i - 1, j, k] + uold[i + 1, j, k]) / dx**2 - \
                        (uold[i, j, k - 1] + uold[i, j, k + 1]) / dz**2

            for j in [ny - 1]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            u[i, :, k] = tridiagscalar(A, B, C, D)
            u[i, :, k] = uold[i, :, k] + omega * (u[i, :, k] - uold[i, :, k])
            uold[i, :, k] = u[i, :, k]

    # Planes across wing
    for k in range(n_left - 1, n_right):

        # Points upstream of airfoil
        for i in range(1, nLE[k] - 1):
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, ny - 1):
                A[j] = 1 / dy**2
                B[j] = -2 / dy**2 - 2 / dx**2 - 2 / dz**2
                C[j] = 1 / dy**2
                D[j] = -(uold[i - 1, j, k] + uold[i + 1, j, k]) / dx**2 - \
                        (uold[i, j, k - 1] + uold[i, j, k + 1]) / dz**2

            for j in [ny - 1]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            u[i, :, k] = tridiagscalar(A, B, C, D)
            u[i, :, k] = uold[i, :, k] + omega * (u[i, :, k] - uold[i, :, k])
            uold[i, :, k] = u[i, :, k]

        # Points where airfoil is
        for i in range(nLE[k] - 1, nTE[k]):
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            # Region below body
            for j in range(1, ny // 2 - 1):
                A[j] = 1 / dy**2
                B[j] = -2 / dy**2 - 2 / dx**2 - 2 / dz**2
                C[j] = 1 / dy**2
                D[j] = -(uold[i - 1, j, k] + uold[i + 1, j, k]) / dx**2 - \
                        (uold[i, j, k - 1] + uold[i, j, k + 1]) / dz**2

            # Just before body
            for j in [ny // 2 - 1]:
                A[j] = 1 / dy**2
                B[j] = -1 / dy**2 - 2 / dx**2 - 2 / dz**2
                C[j] = 0
                D[j] = -(uold[i - 1, j, k] + uold[i + 1, j, k]) / dx**2 - \
                        (uold[i, j, k - 1] + uold[i, j, k + 1]) / dz**2 - \
                        (yBB[i + 1, k] - 2 * yBB[i, k] + yBB[i - 1, k]) / (dx**2 * dy)

            # Just after body
            for j in [ny // 2]:
                A[j] = 0
                B[j] = -1 / dy**2 - 2 / dx**2 - 2 / dz**2
                C[j] = 1 / dy**2
                D[j] = -(uold[i - 1, j, k] + uold[i + 1, j, k]) / dx**2 - \
                        (uold[i, j, k - 1] + uold[i, j, k + 1]) / dz**2 + \
                        (yBT[i + 1, k] - 2 * yBT[i, k] + yBT[i - 1, k]) / (dx**2 * dy)

            # Region above body
            for j in range(ny // 2 + 1, ny - 1):
                A[j] = 1 / dy**2
                B[j] = -2 / dy**2 - 2 / dx**2 - 2 / dz**2
                C[j] = 1 / dy**2
                D[j] = -(uold[i - 1, j, k] + uold[i + 1, j, k]) / dx**2 - \
                        (uold[i, j, k - 1] + uold[i, j, k + 1]) / dz**2

            for j in [ny - 1]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            u[i, :, k] = tridiagscalar(A, B, C, D)
            u[i, :, k] = uold[i, :, k] + omega * (u[i, :, k] - uold[i, :, k])
            uold[i, :, k] = u[i, :, k]

        # Points downstream of airfoil
        for i in range(nTE[k], nx - 1):
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, ny - 1):
                A[j] = 1 / dy**2
                B[j] = -2 / dy**2 - 2 / dx**2 - 2 / dz**2
                C[j] = 1 / dy**2
                D[j] = -(uold[i - 1, j, k] + uold[i + 1, j, k]) / dx**2 - \
                        (uold[i, j, k - 1] + uold[i, j, k + 1]) / dz**2

            for j in [ny - 1]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            u[i, :, k] = tridiagscalar(A, B, C, D)
            u[i, :, k] = uold[i, :, k] + omega * (u[i, :, k] - uold[i, :, k])
            uold[i, :, k] = u[i, :, k]

    # Planes after right wingtip
    for k in range(n_right, nz - 1):
        for i in range(1, nx - 1):
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, ny - 1):
                A[j] = 1 / dy**2
                B[j] = -2 / dy**2 - 2 / dx**2 - 2 / dz**2
                C[j] = 1 / dy**2
                D[j] = -(uold[i - 1, j, k] + uold[i + 1, j, k]) / dx**2 - \
                        (uold[i, j, k - 1] + uold[i, j, k + 1]) / dz**2

            for j in [ny - 1]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            u[i, :, k] = tridiagscalar(A, B, C, D)
            u[i, :, k] = uold[i, :, k] + omega * (u[i, :, k] - uold[i, :, k])
            uold[i, :, k] = u[i, :, k]

    # Residual
    resid = np.zeros((nx, ny, nz))

    for k in range(1, nz - 1):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                resid[i, j, k] = abs( \
                    (u[i + 1, j, k] - 2 * u[i, j, k] + u[i - 1, j, k]) / dx**2 + \
                    (u[i, j + 1, k] - 2 * u[i, j, k] + u[i, j - 1, k]) / dy**2 + \
                    (u[i, j, k + 1] - 2 * u[i, j, k] + u[i, j, k - 1]) / dz**2)

    for k in range(n_left - 1, n_right):
        for i in range(nLE[k] - 1, nTE[k]):
            for j in range(ny // 2 - 1, ny // 2 + 1):
                resid[i, j, k] = 0

    res = np.max(resid)
    resplot[iterator] = res
    iterator += 1

####################################################################################################

cpbot = np.zeros((nx, nz))
cptop = np.zeros((nx, nz))

for k in range(0, nz):
    for i in range(0, nx):
        cpbot[i, k] = -2 * u[i, ny // 2 - 1, k] / uinf
        cptop[i, k] = -2 * u[i, ny // 2, k] / uinf

####################################################################################################
# Plotting

fig = plt.figure(figsize=(10, 6), dpi=200, facecolor='w', edgecolor='k')

TITLE_FONT_SIZE = 10
TICK_FONT_SIZE = 8

X, Z = np.meshgrid(x, z)

plt.subplot(2, 2, 1)
plt.plot(x, yBB[:, nz // 2 - 1], '--b', label='Bottom')
plt.plot(x, yBT[:, nz // 2 - 1], '-b', label='Top')
plt.title('Airfoil Profile', fontsize=TITLE_FONT_SIZE)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().tick_params(labelsize=TICK_FONT_SIZE)
plt.legend()
plt.axis([-4, 4, -1, 1])
plt.grid(True)

ax = plt.subplot(2, 2, 2, projection='3d')
ax.plot_surface(X, Z, -cptop.T, cmap=cm.jet)
ax.view_init(elev=15, azim=225)
plt.title(f'-C_P Across Top of Wing Mach: {mach_num}', fontsize=TITLE_FONT_SIZE)
plt.xlabel('x')
plt.ylabel('z')
plt.gca().tick_params(labelsize=TICK_FONT_SIZE)
plt.grid(True)
plt.gca().set_zlim(-0.2, 0.2)

plt.subplot(2, 2, 3)
plt.semilogy(resplot[resplot != 0])
plt.title('Residual versus Iteration Count', fontsize=TITLE_FONT_SIZE)
plt.xlabel('Number of Iterations')
plt.ylabel('Residual')
plt.gca().tick_params(labelsize=TICK_FONT_SIZE)
plt.grid(True)

plt.tight_layout()
fig.savefig('../fig/project5_part3.png', bbox_inches='tight')

# Show all plots
plt.show()
