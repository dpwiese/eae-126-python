"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 6: Steady, inviscid, adiabatic, compressible, and irrotational flows over airfoils -
    numerical solutions to lifting Problem
Part 4: Wing problem: rectangular wing AR = 6

Profiles: TBD
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import cm

sys.path.insert(1, '../..')

from tools.tridiagscalar import tridiagscalar

aoa_deg = 5
aoa_rad = np.deg2rad(aoa_deg)
uinf = 1

Ma = 0.0

# Grid - ny must be an even number
nx = 30
ny = 20
nz = 100

xmin = -20
xmax = 10
ymin = -ny / 2
ymax = ny / 2
zmin = -nz / 2
zmax = nz / 2

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)
dz = (zmax - zmin) / (nz - 1)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
z = np.linspace(zmin, zmax, nz)

n_left = 22
n_right = 78

nLE = np.zeros(nz, dtype=int)
nTE = np.zeros(nz, dtype=int)

for k in range(n_left - 1):
    nLE[k] = 1
    nTE[k] = 1

for k in range(n_left - 1, n_right):
    nLE[k] = 11
    nTE[k] = 20

for k in range(n_right, nz):
    nLE[k] = 1
    nTE[k] = 1

maxiterator = 300
omega = 1.95
resmax = 10**-6

# Define chordlength, span, and aspect ratio
chord = x[nTE[nz // 2 - 1] - 1] - x[nLE[nz // 2 - 1] - 1]
span = z[n_right] - z[n_left]
AR = span / chord

yBT = np.zeros((nx, nz))
yBB = np.zeros((nx, nz))

for k in range(n_left - 1):
    for i in range(nx):
        yBT[i, k] = 0
        yBB[i, k] = 0

for k in range(n_left - 1, n_right):
    for i in range(0, nLE[k] - 1):
        yBT[i, k] = 0
        yBB[i, k] = 0

    for i in range(nLE[k] - 1, nTE[k]):
        yBT[i, k] = 0
        yBB[i, k] = 0

    for i in range(nTE[k], nx):
        yBT[i, k] = 0
        yBB[i, k] = 0

for k in range(n_right, nz):
    for i in range(nx):
        yBT[i, k] = 0
        yBB[i, k] = 0

Beta = np.sqrt(1 - Ma**2)

u = np.zeros((nx, ny, nz))
uold = np.zeros((nx, ny, nz))

for i in [0]:
    u[i, :, :] = uinf * np.cos(aoa_rad)
    uold[i, :, :] = uinf * np.cos(aoa_rad)

for i in [nx - 1]:
    u[i, :, :] = uinf * np.cos(aoa_rad)
    uold[i, :, :] = uinf * np.cos(aoa_rad)

for j in [0]:
    u[:, j, :] = uinf * np.cos(aoa_rad)
    uold[:, j, :] = uinf * np.cos(aoa_rad)

for j in [ny - 1]:
    u[:, j, :] = uinf * np.cos(aoa_rad)
    uold[:, j, :] = uinf * np.cos(aoa_rad)

for k in [0]:
    u[:, :, k] = uinf * np.cos(aoa_rad)
    uold[:, :, k] = uinf * np.cos(aoa_rad)

for k in [nz - 1]:
    u[:, :, k] = uinf * np.cos(aoa_rad)
    uold[:, :, k] = uinf * np.cos(aoa_rad)

res = 1
iterator = 0
resplot = []

A = np.zeros(ny)
B = np.zeros(ny)
C = np.zeros(ny)
D = np.zeros(ny)

while iterator < maxiterator and res > resmax:

    # Planes before left wingtip
    for k in range(1, n_left - 1):

        for i in range(1, nx - 1):

            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = uinf * np.cos(aoa_rad)

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
                D[j] = uinf * np.cos(aoa_rad)

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
                D[j] = uinf * np.cos(aoa_rad)

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
                D[j] = uinf * np.cos(aoa_rad)

            u[i, :, k] = tridiagscalar(A, B, C, D)
            u[i, :, k] = uold[i, :, k] + omega * (u[i, :, k] - uold[i, :, k])
            uold[i, :, k] = u[i, :, k]

        # Points on vertical LE line
        for i in [nLE[k] - 1]:

            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = uinf * np.cos(aoa_rad)

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
                        (yBB[i + 1, k] - 2*yBB[i, k] + yBB[i - 1, k]) / (dx**2 * dy) + 0.30 / (2*dx)

            # Just after body
            for j in [ny // 2]:
                A[j] = 0
                B[j] = -1 / dy**2 - 2 / dx**2 - 2 / dz**2
                C[j] = 1 / dy**2
                D[j] = -(uold[i - 1, j, k] + uold[i + 1, j, k]) / dx**2 - \
                        (uold[i, j, k - 1] + uold[i, j, k + 1]) / dz**2 + \
                        (yBT[i + 1, k] - 2*yBT[i, k] + yBT[i - 1, k]) / (dx**2 * dy) + 0.30 / (2*dx)

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
                D[j] = uinf * np.cos(aoa_rad)

            u[i, :, k] = tridiagscalar(A, B, C, D)
            u[i, :, k] = uold[i, :, k] + omega * (u[i, :, k] - uold[i, :, k])
            uold[i, :, k] = u[i, :, k]

        # Points where airfoil is
        for i in range(nLE[k], nTE[k] - 1):

            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = uinf * np.cos(aoa_rad)

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
                        (yBB[i + 1, k] - 2*yBB[i, k] + yBB[i - 1, k]) / (dx**2 * dy)

            # Just after body
            for j in [ny // 2]:
                A[j] = 0
                B[j] = -1 / dy**2 - 2 / dx**2 - 2 / dz**2
                C[j] = 1 / dy**2
                D[j] = -(uold[i - 1, j, k] + uold[i + 1, j, k]) / dx**2 - \
                        (uold[i, j, k - 1] + uold[i, j, k + 1]) / dz**2 + \
                        (yBT[i + 1, k] - 2*yBT[i, k] + yBT[i - 1, k]) / (dx**2 * dy)

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
                D[j] = uinf * np.cos(aoa_rad)

            u[i, :, k] = tridiagscalar(A, B, C, D)
            u[i, :, k] = uold[i, :, k] + omega * (u[i, :, k] - uold[i, :, k])
            uold[i, :, k] = u[i, :, k]

        # Points on vertical TE line
        for i in [nTE[k] - 1]:

            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = uinf * np.cos(aoa_rad)

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
                        (yBB[i + 1, k] - 2*yBB[i, k] + yBB[i - 1, k]) / (dx**2 * dy)

            # Just after body
            for j in [ny // 2]:
                A[j] = 0
                B[j] = -1 / dy**2 - 2 / dx**2 - 2 / dz**2
                C[j] = 1 / dy**2
                D[j] = -(uold[i - 1, j, k] + uold[i + 1, j, k]) / dx**2 - \
                        (uold[i, j, k - 1] + uold[i, j, k + 1]) / dz**2 + \
                        (yBT[i + 1, k] - 2*yBT[i, k] + yBT[i - 1, k]) / (dx**2 * dy)

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
                D[j] = uinf * np.cos(aoa_rad)

            u[i, :, k] = tridiagscalar(A, B, C, D)
            u[i, :, k] = uold[i, :, k] + omega * (u[i, :, k] - uold[i, :, k])
            uold[i, :, k] = u[i, :, k]

        # Points downstream of airfoil
        for i in range(nTE[k], nx - 1):

            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = uinf * np.cos(aoa_rad)

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
                D[j] = uinf * np.cos(aoa_rad)

            u[i, :, k] = tridiagscalar(A, B, C, D)
            u[i, :, k] = uold[i, :, k] + omega * (u[i, :, k] - uold[i, :, k])
            uold[i, :, k] = u[i, :, k]

    # Planes after right tip of wing
    for k in range(n_right, nz - 1):

        for i in range(1, nx - 1):

            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = uinf * np.cos(aoa_rad)

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
                D[j] = uinf * np.cos(aoa_rad)

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
    resplot.append(res)
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

fig.suptitle("Results")

# Plot Residuals
ax1 = fig.add_subplot(2, 2, 1)
ax1.semilogy(resplot)
ax1.set_title("Residual vs. Iteration Count", fontsize=TITLE_FONT_SIZE)
ax1.set_xlabel("Number of Iterations")
ax1.set_ylabel("Residual")
ax1.grid(True)
ax1.tick_params(labelsize=TICK_FONT_SIZE)

# Plot Cp distribution
ax2 = fig.add_subplot(2, 2, 2, projection="3d")
X, Z = np.meshgrid(x, z)
ax2.plot_surface(X, Z, -cptop.T, cmap=cm.jet)
ax2.view_init(elev=15, azim=225)
ax2.set_title("-Cp Across Top of Wing", fontsize=TITLE_FONT_SIZE)
ax2.set_xlabel("X")
ax2.set_ylabel("Z")
ax2.tick_params(labelsize=TICK_FONT_SIZE)

# Plot Airfoil Profile
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(x, yBB[:, int(nz / 2)], "--b", label="Bottom")
ax3.plot(x, yBT[:, int(nz / 2)], "-b", label="Top")
ax3.set_title("Airfoil Profile", fontsize=TITLE_FONT_SIZE)
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.axis("equal")
ax3.legend()
ax3.grid(True)
ax3.tick_params(labelsize=TICK_FONT_SIZE)

plt.tight_layout()
fig.savefig('../fig/project6_part4.png', bbox_inches='tight')

# Show all plots
plt.show()
