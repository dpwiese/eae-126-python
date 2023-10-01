"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 1: Steady, inviscid, adiabatic, incompressible, and irrotational 2D flows over cylinder
Part 5: Flow over rotating cylinder using numerical methods

Verify the results of question 4 using numerical methods. Plot all of the results.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

sys.path.insert(1, '../..')

from tools.tridiagscalar import tridiagscalar

# Set up geometric parameters and flow parameters. Omega is a relaxation parameter
omega = 1.87
maxiterator = 100
vinf = 1
rho = 1

# Specify min and max radius, and number of points in the radial direction
rmin = 1
rmax = 3
nr = 12
dr = (rmax - rmin) / (nr - 1)

# Set number of points and grid spacing in the angular direction
thetamin = 0
ntheta = 50
dtheta = 2 * np.pi / ntheta
thetamax = thetamin + ntheta * dtheta

r = np.linspace(rmin, rmax, nr)
theta = np.linspace(thetamin, thetamax - dtheta, ntheta)

# Initialize arrays
rubar = np.ones((ntheta, nr))
rubarold = np.ones((ntheta, nr))
rvbar = np.ones((ntheta, nr))
rvbarold = np.ones((ntheta, nr))

# Negative gamma is clockwise rotation
gamma_array =[0, -np.pi, -4*np.pi, -5*np.pi]

for i_gamma, Gamma in enumerate(gamma_array):

    # Initialize arrays
    A = np.zeros(nr)
    B = np.zeros(nr)
    C = np.zeros(nr)
    D = np.zeros(nr)

    ################################################################################################
    # U BAR

    iterator = 0
    while iterator < maxiterator:
        ############################################################################################
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
                D[j] = r[j] * vinf * np.cos(theta[i])

            rubar[i, :] = tridiagscalar(A, B, C, D)
            rubar[i, :] = rubarold[i, :] + omega * (rubar[i, :] - rubarold[i, :])
            rubarold[i, :] = rubar[i, :]

        ############################################################################################
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
                D[j] = r[j] * vinf * np.cos(theta[i])

            rubar[i, :] = tridiagscalar(A, B, C, D)
            rubar[i, :] = rubarold[i, :] + omega * (rubar[i, :] - rubarold[i, :])
            rubarold[i, :] = rubar[i, :]

        ############################################################################################
        for i in [ntheta-1]:
            for j in [0]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = 0

            for j in range(1, nr-1):
                A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
                B[j] = -(((r[j+1] + 2 * r[j] + r[j-1]) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
                C[j] = (r[j] + r[j+1]) / (2 * dr**2)
                D[j] = -(rubarold[0, j] + rubarold[i-1, j]) / (r[j] * dtheta**2)

            for j in [-1]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = r[j] * vinf * np.cos(theta[i])

            rubar[i, :] = tridiagscalar(A, B, C, D)
            rubar[i, :] = rubarold[i, :] + omega * (rubar[i, :] - rubarold[i, :])
            rubarold[i, :] = rubar[i, :]

        iterator += 1

    ################################################################################################
    # V BAR

    iterator = 0
    while iterator < maxiterator:

        ############################################################################################
        for i in [0]:
            for j in [0]:
                A[j] = 0
                B[j] = -(((r[j+1] + 2 * r[j] + (r[j] - dr)) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
                C[j] = ((r[j] + 2 * r[j+1] + (r[j] - dr)) / (2 * dr**2))
                D[j] = -(rvbarold[i+1, j] + rvbarold[-1, j]) / (r[j] * dtheta**2)

            for j in range(1, nr-1):
                A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
                B[j] = -(((r[j+1] + 2 * r[j] + r[j-1]) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
                C[j] = (r[j] + r[j+1]) / (2 * dr**2)
                D[j] = -(rvbarold[i+1, j] + rvbarold[-1, j]) / (r[j] * dtheta**2)

            for j in [-1]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = -r[j] * vinf * np.sin(theta[i]) + (Gamma / (2 * np.pi))

            rvbar[i, :] = tridiagscalar(A, B, C, D)
            rvbar[i, :] = rvbarold[i, :] + omega * (rvbar[i, :] - rvbarold[i, :])
            rvbarold[i, :] = rvbar[i, :]

        ############################################################################################
        for i in range(1, ntheta-1):
            for j in [0]:
                A[j] = 0
                B[j] = -(((r[j+1] + 2 * r[j] + (r[j] - dr)) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
                C[j] = ((r[j] + 2 * r[j+1] + (r[j] - dr)) / (2 * dr**2))
                D[j] = -(rvbarold[i+1, j] + rvbarold[i-1, j]) / (r[j] * dtheta**2)

            for j in range(1, nr-1):
                A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
                B[j] = -(((r[j+1] + 2 * r[j] + r[j-1]) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
                C[j] = (r[j] + r[j+1]) / (2 * dr**2)
                D[j] = -(rvbarold[i+1, j] + rvbarold[i-1, j]) / (r[j] * dtheta**2)

            for j in [-1]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = -r[j] * vinf * np.sin(theta[i]) + (Gamma / (2 * np.pi))

            rvbar[i, :] = tridiagscalar(A, B, C, D)
            rvbar[i, :] = rvbarold[i, :] + omega * (rvbar[i, :] - rvbarold[i, :])
            rvbarold[i, :] = rvbar[i, :]

        ############################################################################################
        for i in [ntheta-1]:
            for j in [0]:
                A[j] = 0
                B[j] = -(((r[j+1] + 2 * r[j] + (r[j] - dr)) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
                C[j] = ((r[j] + 2 * r[j+1] + (r[j] - dr)) / (2 * dr**2))
                D[j] = -(rvbarold[0, j] + rvbarold[ntheta-2, j]) / (r[j] * dtheta**2)

            for j in range(1, nr-1):
                A[j] = (1 / dr**2) * (r[j-1] + r[j]) * 0.5
                B[j] = -(((r[j+1] + 2 * r[j] + r[j-1]) / (2 * dr**2)) + (2 / (r[j] * dtheta**2)))
                C[j] = (r[j] + r[j+1]) / (2 * dr**2)
                D[j] = -(rvbarold[0, j] + rvbarold[ntheta-2, j]) / (r[j] * dtheta**2)

            for j in [-1]:
                A[j] = 0
                B[j] = 1
                C[j] = 0
                D[j] = -r[j] * vinf * np.sin(theta[i]) + (Gamma / (2 * np.pi))

            rvbar[i, :] = tridiagscalar(A, B, C, D)
            rvbar[i, :] = rvbarold[i, :] + omega * (rvbar[i, :] - rvbarold[i, :])
            rvbarold[i, :] = rvbar[i, :]

        iterator += 1

    ################################################################################################

    # Calculate velocity components and total velocity
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

    for i in range(ntheta):
        for j in range(nr):
            x[i, j] = r[j] * np.cos(theta[i])
            y[i, j] = r[j] * np.sin(theta[i])
            ubar[i, j] = rubar[i, j] / r[j]
            vbar[i, j] = rvbar[i, j] / r[j]
            ux[i, j] = ubar[i, j] * np.cos(theta[i])
            uy[i, j] = ubar[i, j] * np.sin(theta[i])
            vx[i, j] = -vbar[i, j] * np.sin(theta[i])
            vy[i, j] = vbar[i, j] * np.cos(theta[i])
            xvel[i, j] = ux[i, j] + vx[i, j]
            yvel[i, j] = uy[i, j] + vy[i, j]
            vtot[i, j] = np.sqrt(xvel[i, j]**2 + yvel[i, j]**2)
            ptot[i, j] = 1 - (vtot[i, j]**2) / (vinf**2)

    ################################################################################################
    # Plot everything

    # TODO@dpwiese - solve this more elegantly
    # "Wrap" data around for continuous contour plots
    x2 = np.append(x, np.array([x[0,:]]), axis=0)
    y2 = np.append(y, np.array([y[0,:]]), axis=0)
    vtot2 = np.append(vtot, np.array([vtot[0,:]]), axis=0)
    ptot2 = np.append(ptot, np.array([ptot[0,:]]), axis=0)

    def plot_all(X, Y, x, y, vel_x, vel_y, vel, cp, pattern):
        """ Plotter for quiver and contour plots"""
        TITLE_FONT_SIZE = 10
        TICK_FONT_SIZE = 8

        fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')

        # Suplot 2: Velocity contours
        c1 = axs[0, 1].contourf(x, y, vel, 40, cmap='viridis')
        axs[0, 1].plot(x[:, 0], y[:, 0], '-k', linewidth=1)
        axs[0, 1].fill_between(x[:, 0], y[:, 0], color='r')
        axs[0, 1].set_aspect('equal', 'box')
        axs[0, 1].set_xlim([-2, 2])
        axs[0, 1].set_ylim([-2, 2])
        axs[0, 1].set_title(f'{pattern}:\nVelocity Contours', fontsize=TITLE_FONT_SIZE)
        axs[0, 1].tick_params(labelsize=TICK_FONT_SIZE)

        norm1 = matplotlib.colors.Normalize(vmin=c1.cvalues.min(), vmax=c1.cvalues.max())
        sm1 = plt.cm.ScalarMappable(norm=norm1, cmap = c1.cmap)
        sm1.set_array([])

        # Set the colorbar to the first subplot (for spacing only), remove it, and set the colorbar
        # on the intended subplot.
        cb1 = plt.colorbar(c1, ax=axs[0, 0], orientation='vertical', fraction=0.046, pad=0.04)
        cb1.remove()
        cb1 = plt.colorbar(sm1, ax=axs[0, 1], orientation='vertical', fraction=0.046, pad=0.04)
        cb1.ax.tick_params(labelsize=TICK_FONT_SIZE)

        # Subplot 1: Velocity vectors
        axs[0, 0].quiver(X, Y, vel_x, vel_y, scale=20)
        axs[0, 0].plot(x[:, 0], y[:, 0], '-k', linewidth=1)
        axs[0, 0].fill_between(x[:, 0], y[:, 0], color='r')
        axs[0, 0].set_aspect('equal', 'box')
        axs[0, 0].set_xlim([-2, 2])
        axs[0, 0].set_ylim([-2, 2])
        axs[0, 0].set_title(f'{pattern}:\nVelocity Vectors', fontsize=TITLE_FONT_SIZE)
        axs[0, 0].tick_params(labelsize=TICK_FONT_SIZE)

        # Subplot 3: Pressure contours
        c2 = axs[1, 0].contourf(x, y, cp, 40, cmap='viridis')
        axs[1, 0].plot(x[:, 0], y[:, 0], '-k', linewidth=1)
        axs[1, 0].fill_between(x[:, 0], y[:, 0], color='r')
        axs[1, 0].set_aspect('equal', 'box')
        axs[1, 0].set_xlim([-2, 2])
        axs[1, 0].set_ylim([-2, 2])
        axs[1, 0].set_title(f'{pattern}:\nPressure Contours', fontsize=TITLE_FONT_SIZE)
        axs[1, 0].tick_params(labelsize=TICK_FONT_SIZE)

        norm2 = matplotlib.colors.Normalize(vmin=c2.cvalues.min(), vmax=c2.cvalues.max())
        sm2 = plt.cm.ScalarMappable(norm=norm2, cmap = c2.cmap)
        sm2.set_array([])
        cb2 = plt.colorbar(sm2, ax=axs[1, 0], orientation='vertical', fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=TICK_FONT_SIZE)

        c3 = axs[1, 1].contourf(x, y, cp, 40, cmap='viridis')
        axs[1, 1].plot(x[:, 0], y[:, 0], '-k', linewidth=1)
        axs[1, 1].fill_between(x[:, 0], y[:, 0], color='r')
        axs[1, 1].set_aspect('equal', 'box')
        axs[1, 1].set_xlim([-2, 2])
        axs[1, 1].set_ylim([-2, 2])
        axs[1, 1].set_title(f'{pattern}:\nPressure Contours and Velocity Vectors', fontsize=TITLE_FONT_SIZE)
        axs[1, 1].tick_params(labelsize=TICK_FONT_SIZE)
        axs[1, 1].quiver(X, Y, vel_x, vel_y, scale=20)

        norm3 = matplotlib.colors.Normalize(vmin=c3.cvalues.min(), vmax=c3.cvalues.max())
        sm3 = plt.cm.ScalarMappable(norm=norm3, cmap = c3.cmap)
        sm3.set_array([])
        cb3 = plt.colorbar(sm3, ax=axs[1, 1], orientation='vertical', fraction=0.046, pad=0.04)
        cb3.ax.tick_params(labelsize=TICK_FONT_SIZE)

        plt.tight_layout()

        fig.savefig(f'../fig/project1_part5_rotating_cylinder_{i_gamma}.png', bbox_inches='tight')

    plot_all(x, y, x2, y2, xvel, yvel, vtot2, ptot2, f'Rotating Cylinder Gamma={Gamma:.2f}')


plt.show()
