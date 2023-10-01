"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 1: Steady, inviscid, adiabatic, incompressible, and irrotational 2D flows over cylinder
Part 6: Flow over rotating cylinder with fins using numerical methods

Discussion of relationship between lift and circulation as well as discrepancies between analytical
and experimental results for lift and drag. As well as a plot of lift vs. circulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, '../..')

from tools.tridiagscalar import tridiagscalar

# Set up geometric parameters and flow parameters. Omega is a relaxation parameter
omega = 1.00
maxiterator = 500
uinf = 1

# Specify min and max radius, and number of points in the radial direction
rmin = 1
rmax = 4 * rmin
nr = 22
dr = (rmax - rmin) / (nr - 1)

nf = round(2 * nr / 3)

# Set number of points and grid spacing in the angular direction
thetamin = 0
ntheta = 60
dtheta = 2 * np.pi / ntheta
thetamax = thetamin + ntheta * dtheta
thetaftop = thetamax / 4
thetafbot = 3 * thetamax / 4
lf = 2 * rmin

# Create r and theta arrays
r = np.linspace(rmin, rmax, nr)
theta = np.linspace(thetamin, thetamax - dtheta, ntheta)

# Initialize arrays
rubar = np.ones((ntheta, nr))
rubarold = np.ones((ntheta, nr))
rvbar = np.ones((ntheta, nr))
rvbarold = np.ones((ntheta, nr))

Gamma = 0

# Preallocation
A = np.zeros(nr)
B = np.zeros(nr)
C = np.zeros(nr)
D = np.zeros(nr)

# Define functions
def set_params(i, j, A, B, C, D):
    A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
    B[j] = -(((r[j+1] + 2*r[j] + r[j-1]) / (2*dr**2)) + (2 / (r[j] * dtheta**2)))
    C[j] = ((r[j] + r[j+1]) / (2*dr**2))
    D[j] = -(rubarold[i+1, j] + rubarold[ntheta-1, j]) / (r[j] * dtheta**2)

# Main loop for U BAR
for iterator in range(maxiterator):
    ################################################################################################
    for i in [0]:
        for j in [0]:
            A[j] = 0
            B[j] = 1
            C[j] = 0
            D[j] = 0

        for j in range(1, nr-1):
            A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
            B[j] = -(((r[j+1] + 2 * r[j] + r[j-1]) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
            C[j] = (r[j] + r[j+1]) / (2 * dr**2)
            D[j] = -(rubarold[i+1, j] + rubarold[-1, j]) / (r[j] * dtheta**2)

        for j in [-1]:
            A[j] = 0
            B[j] = 1
            C[j] = 0
            D[j] = r[j] * uinf * np.cos(theta[i])

        rubar[i, :] = tridiagscalar(A, B, C, D)
        rubar[i, :] = rubarold[i, :] + omega * (rubar[i, :] - rubarold[i, :])
        rubarold[i, :] = rubar[i, :]

    ################################################################################################
    for i in range(1, ntheta-1):
        for j in [0]:
            A[j] = 0
            B[j] = 1
            C[j] = 0
            D[j] = 0

        for j in range(1, nr-1):
            A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
            B[j] = -(((r[j+1] + 2 * r[j] + r[j-1]) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
            C[j] = (r[j] + r[j+1]) / (2 * dr**2)
            D[j] = -(rubarold[i+1, j] + rubarold[i-1, j]) / (r[j] * dtheta**2)

        for j in [-1]:
            A[j] = 0
            B[j] = 1
            C[j] = 0
            D[j] = r[j] * uinf * np.cos(theta[i])

        # Top fin
        if i == round(ntheta/4):
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, nf+1):
                A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
                B[j] = -(((r[j+1] + 2 * r[j] + r[j-1]) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
                C[j] = (r[j] + r[j+1]) / (2 * dr**2)
                D[j] = -(rubarold[i+1, j] + rubarold[i-1, j]) / (r[j] * dtheta**2)

            for j in range(nf+1, nr):
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = r[j] * uinf * np.cos(theta[i])

        if i == round(ntheta/4)+1:
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, nf+1):
                A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
                B[j] = -(((r[j+1] + 2 * r[j] + r[j-1]) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
                C[j] = (r[j] + r[j+1]) / (2 * dr**2)
                D[j] = -(2 * rubarold[i-1, j]) / (r[j] * dtheta**2)

            for j in range(nf+1, nr):
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = r[j] * uinf * np.cos(theta[i])

        if i == round(ntheta/4)+2:
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, nf+1):
                A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
                B[j] = -(((r[j+1] + 2 * r[j] + r[j-1]) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
                C[j] = (r[j] + r[j+1]) / (2 * dr**2)
                D[j] = -(rubarold[i+1, j] + rubarold[i-1, j]) / (r[j] * dtheta**2)

            for j in range(nf+1, nr):
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = r[j] * uinf * np.cos(theta[i])

        # Bottom Fin
        if i == round(3 * ntheta/4):
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, nf+1):
                A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
                B[j] = -(((r[j+1] + 2 * r[j] + r[j-1]) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
                C[j] = (r[j] + r[j+1]) / (2 * dr**2)
                D[j] = -(rubarold[i+1, j] + rubarold[i-1, j]) / (r[j] * dtheta**2)

            for j in range(nf+1, nr):
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = r[j] * uinf * np.cos(theta[i])

        if i == round(3 * ntheta/4)+1:
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, nf+1):
                A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
                B[j] = -(((r[j+1] + 2 * r[j] + r[j-1]) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
                C[j] = (r[j] + r[j+1]) / (2 * dr**2)
                D[j] = -(rubarold[i+1, j] + rubarold[i-1, j]) / (r[j] * dtheta**2)

            for j in range(nf+1, nr):
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = r[j] * uinf * np.cos(theta[i])

        if i == round(3 * ntheta/4)+2:
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, nf+1):
                A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
                B[j] = -(((r[j+1] + 2 * r[j] + r[j-1]) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
                C[j] = (r[j] + r[j+1]) / (2 * dr**2)
                D[j] = -(rubarold[i+1, j] + rubarold[i-1, j]) / (r[j] * dtheta**2)

            for j in range(nf+1, nr):
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = r[j] * uinf * np.cos(theta[i])

        rubar[i, :] = tridiagscalar(A, B, C, D)
        rubar[i, :] = rubarold[i, :] + omega * (rubar[i, :] - rubarold[i, :])
        rubarold[i, :] = rubar[i, :]

    ################################################################################################
    for i in [-1]:
        for j in [0]:
            A[j] = 0
            B[j] = 1
            C[j] = 0
            D[j] = 0

        for j in range(1, nr-1):
            A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
            B[j] = -(((r[j+1] + 2 * r[j] + r[j-1]) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
            C[j] = (r[j] + r[j+1]) / (2 * dr**2)
            D[j] = -(rubarold[i+1, j] + rubarold[-1, j]) / (r[j] * dtheta**2)

        for j in [-1]:
            A[j] = 0
            B[j] = 1
            C[j] = 0
            D[j] = r[j] * uinf * np.cos(theta[i])

        rubar[i, :] = tridiagscalar(A, B, C, D)
        rubar[i, :] = rubarold[i, :] + omega * (rubar[i, :] - rubarold[i, :])
        rubarold[i, :] = rubar[i, :]

# Main loop for V BAR
for iterator in range(maxiterator):
    ################################################################################################
    for i in [0]:
        for j in [0]:
            A[j] = 0
            B[j] = -(((r[j+1] + 2*r[j] + (r[j] - dr)) / (2*dr**2)) + (2 / (r[j] * dtheta**2)))
            C[j] = ((r[j] + 2*r[j+1] + (r[j] - dr)) / (2*dr**2))
            D[j] = -(rvbarold[i+1, j] + rvbarold[-1, j]) / (r[j] * dtheta**2)

        for j in range(1, nr-1):
            A[j] = 0
            B[j] = -(((r[j+1] + 2*r[j] + (r[j] - dr)) / (2*dr**2)) + (2 / (r[j] * dtheta**2)))
            C[j] = ((r[j] + 2*r[j+1] + (r[j] - dr)) / (2*dr**2))
            D[j] = -(rvbarold[i+1, j] + rvbarold[-1, j]) / (r[j] * dtheta**2)

        for j in [-1]:
            A[j] = 0
            B[j] = 1
            C[j] = 0
            D[j] = -r[j] * uinf * np.sin(theta[i]) + (Gamma / (2 * np.pi))

        rvbar[i, :] = tridiagscalar(A, B, C, D)
        rvbar[i, :] = rvbarold[i, :] + omega * (rvbar[i, :] - rvbarold[i, :])
        rvbarold[i, :] = rvbar[i, :]

    ################################################################################################
    for i in range(1, ntheta-1):
        for j in [0]:
            A[j] = 0
            B[j] = -(((r[j+1] + 2*r[j] + (r[j] - dr)) / (2*dr**2)) + (2 / (r[j] * dtheta**2)))
            C[j] = ((r[j] + 2*r[j+1] + (r[j] - dr)) / (2*dr**2))
            D[j] = -(rvbarold[i+1, j] + rvbarold[i-1, j]) / (r[j] * dtheta**2)

        for j in range(1, nr-1):
            A[j] = 0
            B[j] = -(((r[j+1] + 2*r[j] + (r[j] - dr)) / (2*dr**2)) + (2 / (r[j] * dtheta**2)))
            C[j] = ((r[j] + 2*r[j+1] + (r[j] - dr)) / (2*dr**2))
            D[j] = -(rvbarold[i+1, j] + rvbarold[i-1, j]) / (r[j] * dtheta**2)

        for j in [-1]:
            A[j] = 0
            B[j] = 1
            C[j] = 0
            D[j] = -r[j] * uinf * np.cos(theta[i]) + (Gamma / (2 * np.pi))

        # Top fin
        if i == round(ntheta/4)+1:
            for j in range(0, nf):
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0
            for j in range(nf, nr):
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = -r[j] * uinf * np.cos(theta[i]) + (Gamma / (2 * np.pi))

        # Bottom fin
        if i == round(3*ntheta/4)+1:
            for j in range(0, nf):
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0
            for j in range(nf, nr):
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = -r[j] * uinf * np.cos(theta[i]) + (Gamma / (2 * np.pi))

        rvbar[i, :] = tridiagscalar(A, B, C, D)
        rvbar[i, :] = rvbarold[i, :] + omega * (rvbar[i, :] - rvbarold[i, :])
        rvbarold[i, :] = rvbar[i, :]

    ################################################################################################
    for i in [ntheta-1]:
        for j in [0]:
            A[j] = 0
            B[j] = -(((r[j+1] + 2*r[j] + (r[j] - dr)) / (2*dr**2)) + (2 / (r[j] * dtheta**2)))
            C[j] = ((r[j] + 2*r[j+1] + (r[j] - dr)) / (2*dr**2))
            D[j] = -(rvbarold[0, j] + rvbarold[-1, j]) / (r[j] * dtheta**2)

        for j in range(1, nr-1):
            A[j] = 0
            B[j] = -(((r[j+1] + 2*r[j] + (r[j] - dr)) / (2*dr**2)) + (2 / (r[j] * dtheta**2)))
            C[j] = ((r[j] + 2*r[j+1] + (r[j] - dr)) / (2*dr**2))
            D[j] = -(rvbarold[0, j] + rvbarold[-1, j]) / (r[j] * dtheta**2)

        for j in [-1]:
            A[j] = 0
            B[j] = 1
            C[j] = 0
            D[j] = -r[j] * uinf * np.sin(theta[i]) + (Gamma / (2 * np.pi))

        rvbar[i, :] = tridiagscalar(A, B, C, D)
        rvbar[i, :] = rvbarold[i, :] + omega * (rvbar[i, :] - rvbarold[i, :])
        rvbarold[i, :] = rvbar[i, :]

####################################################################################################
# Calculate x, y, ubar, vbar, ux, uy, vx, vy, xvel, yvel, vtot, ptot
x = np.zeros((ntheta, nr))
y = np.zeros((ntheta, nr))
ubar = np.zeros((ntheta, nr))
vbar = np.zeros((ntheta, nr))
ux = np.zeros((ntheta, nr))
uy = np.zeros((ntheta, nr))
vx = np.zeros((ntheta, nr))
vy = np.zeros((ntheta, nr))
xvel = np.zeros((ntheta, nr))
yvel = np.zeros((ntheta, nr))
vtot = np.zeros((ntheta, nr))
ptot = np.zeros((ntheta, nr))

for i in range(ntheta-1):
    for j in range(nr-1):
        x[i, j] = r[j] * np.cos(theta[i])
        y[i, j] = r[j] * np.sin(theta[i])
        ubar[i, j] = rubar[i, j] / r[j]
        vbar[i, j] = rvbar[i, j] / r[j]
        if j == 0:
            ux[i, j] = (rubar[i, j+1] - rubar[i, nr-1]) / (2*dr)
            uy[i, j] = (rubar[i, j] - rubar[i, nr-1]) / (2*r[j]*dtheta)
            vx[i, j] = (rvbar[i, j+1] - rvbar[i, nr-1]) / (2*dr)
            vy[i, j] = (rvbar[i, j] - rvbar[i, nr-1]) / (2*r[j]*dtheta)
        elif 1 <= j < nr-1:
            ux[i, j] = (rubar[i, j+1] - rubar[i, j-1]) / (2*dr)
            uy[i, j] = (rubar[i, j+1] - rubar[i, j-1]) / (2*r[j]*dtheta)
            vx[i, j] = (rvbar[i, j+1] - rvbar[i, j-1]) / (2*dr)
            vy[i, j] = (rvbar[i, j+1] - rvbar[i, j-1]) / (2*r[j]*dtheta)
        elif j == nr-1:
            ux[i, j] = (rubar[i, 0] - rubar[i, j-1]) / (2*dr)
            uy[i, j] = (rubar[i, 0] - rubar[i, j-1]) / (2*r[j]*dtheta)
            vx[i, j] = (rvbar[i, 0] - rvbar[i, j-1]) / (2*dr)
            vy[i, j] = (rvbar[i, 0] - rvbar[i, j-1]) / (2*r[j]*dtheta)
        xvel[i, j] = ubar[i, j] * np.cos(theta[i]) - vbar[i, j] * np.sin(theta[i])
        yvel[i, j] = ubar[i, j] * np.sin(theta[i]) + vbar[i, j] * np.cos(theta[i])
        vtot[i, j] = np.sqrt(xvel[i, j]**2 + yvel[i, j]**2)
        ptot[i, j] = -0.5 * (xvel[i, j]**2 + yvel[i, j]**2) + (Gamma / (2*np.pi)) * np.log(np.sqrt(x[i, j]**2 + y[i, j]**2))

# Plot the results
plt.figure(figsize=(10, 5))
plt.contourf(x, y, vtot, cmap='viridis', levels=100)
plt.colorbar(label='Velocity Magnitude')
plt.title('Velocity Contours')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
