"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 1: Steady, inviscid, adiabatic, incompressible, and irrotational 2D flows over cylinder
Part 4: Flow field over rotating cylinder

Derive the velocity solution for a cylinder in free stream. Plot the results for a cylinder with no
circulation as well as calculate and plot the results for circulation less than gamma critical and
gamma critical. Calculate the pressure on the surface and plot the results. Derive the lift equation
(L=-rho*u*Gamma) and plot to verify the results. Calculate and plot the relation between gamma and
theta separation. Plot the results.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

# Set up coordinates
xmin = -2
xmax = 2
dx = 0.1
Y, X = np.meshgrid(np.arange(xmin, xmax+dx, dx), np.arange(xmin, xmax+dx, dx))
ngrid = len(X)
x = np.linspace(xmin, xmax, ngrid)
y = np.linspace(xmin, xmax, ngrid)

# Negative gamma is clockwise rotation
gamma_array =[0, -np.pi, -4*np.pi, -5*np.pi]

for i_gamma, Gamma in enumerate(gamma_array):

    # Freestream flow from left to right
    vinf = 1

    # Doublet strength dictates radius
    C = 0.5

    # Calculate radius and trigonometric functions
    r = np.sqrt(X**2 + Y**2)
    cosine = X / r
    sine = Y / r

    # # Set these to identically zero to prevent numerical divide-by-zero issues
    # cosine[ngrid//2, ngrid//2] = 0
    # sine[ngrid//2, ngrid//2] = 0

    # Generate velocity vectors and pressure contours for a doublet in uniform flow: Cylinder
    vdubx = -2 * C * (cosine**2) / (r**2)
    vduby = -2 * C * (cosine * sine) / (r**2)

    # Generate velocity vectors and pressure contours for a vortex
    vvortx = -Gamma / (2 * np.pi * r) * sine
    vvorty = Gamma / (2 * np.pi * r) * cosine

    # Combine velocity components for cylinder and vortex
    vtotx = vvortx + vdubx + vinf
    vtoty = vvorty + vduby

    vtot = np.sqrt(vtotx**2 + vtoty**2)
    cptot = 1 - (vtot**2 / vinf**2)

    vtotx = np.clip(vtotx, -2, 2)
    vtoty = np.clip(vtoty, -2, 2)

    cptot = np.clip(cptot, -25, 25)
    vtot = np.clip(vtot, 0, 20)

    ################################################################################################
    # Plot everything

    def plot_all(X, Y, x, y, vel_x, vel_y, vel, cp, pattern):
        """ Plotter for quiver and contour plots"""
        TITLE_FONT_SIZE = 10
        TICK_FONT_SIZE = 8
        CIRCLE_COLOR = '#AAAAAA'
        CIRCLE_ALPHA = 0.3

        gridspec = {'width_ratios': [1, 1], 'height_ratios': [1, 1]}
        fig, axs = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw=gridspec, dpi=200, facecolor='w', edgecolor='k')

        # Plot 2: Velocity contours
        c1 = axs[0, 1].contour(x, y, vel, 400, cmap=cm.jet, linewidths=0.5)
        axs[0, 1].set_aspect('equal', 'box')
        axs[0, 1].set_xlim(xmin, xmax)
        axs[0, 1].set_ylim(xmin, xmax)
        axs[0, 1].set_title(f'{pattern}: Velocity Contours', fontsize=TITLE_FONT_SIZE)
        axs[0, 1].tick_params(labelsize=TICK_FONT_SIZE)
        axs[0, 1].add_patch(plt.Circle((0, 0), 1, color=CIRCLE_COLOR, alpha=CIRCLE_ALPHA))

        norm1 = matplotlib.colors.Normalize(vmin=c1.cvalues.min(), vmax=c1.cvalues.max())
        sm1 = plt.cm.ScalarMappable(norm=norm1, cmap = c1.cmap)
        sm1.set_array([])

        # Set the colorbar to the first subplot (for spacing only), remove it, and set the colorbar on
        # the intended subplot.
        cb1 = plt.colorbar(c1, ax=axs[0, 0], orientation='vertical', fraction=0.046, pad=0.04)
        cb1.remove()
        cb1 = plt.colorbar(sm1, ax=axs[0, 1], orientation='vertical', fraction=0.046, pad=0.04)
        cb1.ax.tick_params(labelsize=TICK_FONT_SIZE)

        # Plot 1: Velocity Vectors
        axs[0, 0].quiver(X, Y, vel_x, vel_y, scale=20)
        axs[0, 0].set_aspect('equal', 'box')
        axs[0, 0].set_xlim(xmin, xmax)
        axs[0, 0].set_ylim(xmin, xmax)
        axs[0, 0].set_title(f'{pattern}: Velocity Vectors', fontsize=TITLE_FONT_SIZE)
        axs[0, 0].tick_params(labelsize=TICK_FONT_SIZE)
        axs[0, 0].add_patch(plt.Circle((0, 0), 1, color=CIRCLE_COLOR, alpha=CIRCLE_ALPHA))

        # Plot 3: Pressure contours
        c2 = axs[1, 0].contour(x, y, cp, 200, cmap=cm.jet, linewidths=0.5)
        axs[1, 0].set_aspect('equal', 'box')
        axs[1, 0].set_xlim(xmin, xmax)
        axs[1, 0].set_ylim(xmin, xmax)
        axs[1, 0].set_title(f'{pattern}: Pressure Contours', fontsize=TITLE_FONT_SIZE)
        axs[1, 0].tick_params(labelsize=TICK_FONT_SIZE)
        axs[1, 0].add_patch(plt.Circle((0, 0), 1, color=CIRCLE_COLOR, alpha=CIRCLE_ALPHA))

        norm2 = matplotlib.colors.Normalize(vmin=c2.cvalues.min(), vmax=c2.cvalues.max())
        sm2 = plt.cm.ScalarMappable(norm=norm2, cmap = c2.cmap)
        sm2.set_array([])
        cb2 = plt.colorbar(sm2, ax=axs[1, 0], orientation='vertical', fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=TICK_FONT_SIZE)

        # Plot 4: Pressure contours and velocity vectors
        c3 = axs[1, 1].contour(x, y, cp, 200, cmap=cm.jet, linewidths=0.5)
        axs[1, 1].quiver(X, Y, vel_x, vel_y, scale=20, zorder=10)
        axs[1, 1].set_aspect('equal', 'box')
        axs[1, 1].set_xlim(xmin, xmax)
        axs[1, 1].set_ylim(xmin, xmax)
        axs[1, 1].set_title(f'{pattern}:\nPressure Contours and Velocity Vectors', fontsize=TITLE_FONT_SIZE)
        axs[1, 1].tick_params(labelsize=TICK_FONT_SIZE)
        axs[1, 1].add_patch(plt.Circle((0, 0), 1, color=CIRCLE_COLOR, alpha=CIRCLE_ALPHA))

        norm3 = matplotlib.colors.Normalize(vmin=c2.cvalues.min(), vmax=c2.cvalues.max())
        sm3 = plt.cm.ScalarMappable(norm=norm3, cmap = c3.cmap)
        sm3.set_array([])
        cb3 = plt.colorbar(sm3, ax=axs[1, 1], orientation='vertical', fraction=0.046, pad=0.04)
        cb3.ax.tick_params(labelsize=TICK_FONT_SIZE)

        plt.tight_layout()

        fig.savefig(f'../fig/project1_part4_{i_gamma}.png', bbox_inches='tight')

    plot_all(X, Y, x, y, vtotx, vtoty, vtot.T, cptot.T, f'Rotating Cylinder Gamma={Gamma:.2f}')

# Set up parameters
vinf = 1
R = 1
gamma = np.linspace(0, 4*np.pi, 50)

# Calculate theta1 and theta2
theta1 = np.arcsin(-gamma / (4 * np.pi * vinf * R))
theta2 = np.pi + theta1

# Plot theta separation versus gamma
fig2 = plt.figure(figsize=(4, 3), dpi=200, facecolor='w', edgecolor='k')
plt.plot(gamma, theta1, '--', label='Theta 1')
plt.plot(gamma, theta2, label='Theta 2')
plt.title('Theta Separation versus Gamma')
plt.xlabel('Gamma')
plt.ylabel('Theta Separation')
plt.legend()

fig2.savefig(f'../fig/project1_part4_theta_separation.png', bbox_inches='tight')

plt.show()
