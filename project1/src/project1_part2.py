"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 1: Steady, inviscid, adiabatic, incompressible, and irrotational 2D flows over cylinder
Part 2: Flow Field for Source, Vortex, and Doublet

Derive the velocity for source, vortex and doublet. Plot the velocity vectors and contours for each
case. Calculate the pressure coefficients and plot the contours of pressure.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

# Set up coordinates
xmin = -1
xmax = 1
dx = 0.05
x = np.arange(xmin, xmax+dx, dx)
y = np.arange(xmin, xmax+dx, dx)
X, Y = np.meshgrid(x, y)
ngrid = len(x)

# Clip limit to prevent large values making ugly plots
clip_lim = 10

rho = 1
Q = 4
Gamma = 4
vinf = 1
C = 0.2

r = np.sqrt(X**2 + Y**2)
cosine = X / r
sine = Y / r

# Set these to identically zero to prevent numerical divide-by-zero issues
cosine[ngrid//2, ngrid//2] = 0
sine[ngrid//2, ngrid//2] = 0

####################################################################################################
# Source

# Generate velocity vectors and pressure contours
vel_source = Q / (2 * np.pi * rho * r)
cp_source = 1 - (vel_source**2 / vinf**2)

# Clip values
vel_source = np.clip(vel_source, -clip_lim, clip_lim)
cp_source = np.clip(cp_source, -clip_lim, clip_lim)

vel_source_x = vel_source * cosine
vel_source_y = vel_source * sine

####################################################################################################
# Vortex

# Generate velocity vectors and pressure contours
vel_vortex = Gamma / (2 * np.pi * r)
cp_vortex = 1 - (vel_vortex**2 / vinf**2)

# Clip values
vel_vortex = np.clip(vel_vortex, -clip_lim, clip_lim)
cp_vortex = np.clip(cp_vortex, -clip_lim, clip_lim)

vel_vortex_x = -vel_vortex * sine
vel_vortex_y = vel_vortex * cosine

####################################################################################################
# Doublet

# Generate velocity vectors and pressure contours
vel_doublet_x = -2 * C * (cosine**2) / (r**2)
vel_doublet_y = -2 * C * (cosine * sine) / (r**2)

# Clip values
vel_doublet_x = np.clip(vel_doublet_x, -5, 5)
vel_doublet_y = np.clip(vel_doublet_y, -5, 5)

vel_doublet = np.sqrt(vel_doublet_x**2 + vel_doublet_y**2)
cp_doublet = 1 - (vel_doublet**2 / vinf**2)

# Clip values
cp_doublet = np.clip(cp_doublet, -clip_lim, clip_lim)
vel_doublet = np.clip(vel_doublet, -clip_lim, clip_lim)

# Set central point equal to the next one to avoid division by zero
vel_doublet_x[ngrid//2 - 1, ngrid//2 - 1] = vel_doublet_x[ngrid//2, ngrid//2 - 1]
vel_doublet_y[ngrid//2 - 1, ngrid//2 - 1] = vel_doublet_y[ngrid//2, ngrid//2 - 1]

####################################################################################################
# Plot everything

def plot_all(X, Y, x, y, vel_x, vel_y, vel, cp, pattern):
    """ Plotter for quiver and contour plots"""
    TITLE_FONT_SIZE = 10
    TICK_FONT_SIZE = 8

    gridspec = {'width_ratios': [1, 1], 'height_ratios': [1, 1]}
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw=gridspec, dpi=200, facecolor='w', edgecolor='k')

    # Plot 2: Velocity contours
    c1 = axs[0, 1].contour(x, y, vel, 40, cmap=cm.jet, linewidths=0.5)
    axs[0, 1].set_aspect('equal', 'box')
    axs[0, 1].set_xlim(xmin, xmax)
    axs[0, 1].set_ylim(xmin, xmax)
    axs[0, 1].set_title(f'{pattern}: Velocity Contours', fontsize=TITLE_FONT_SIZE)
    axs[0, 1].tick_params(labelsize=TICK_FONT_SIZE)

    norm1 = matplotlib.colors.Normalize(vmin=c1.cvalues.min(), vmax=c1.cvalues.max())
    sm1 = plt.cm.ScalarMappable(norm=norm1, cmap=c1.cmap)
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

    # Plot 3: Pressure contours
    c2 = axs[1, 0].contour(x, y, cp, 40, cmap=cm.jet, linewidths=0.5)
    axs[1, 0].set_aspect('equal', 'box')
    axs[1, 0].set_xlim(xmin, xmax)
    axs[1, 0].set_ylim(xmin, xmax)
    axs[1, 0].set_title(f'{pattern}: Pressure Contours', fontsize=TITLE_FONT_SIZE)
    axs[1, 0].tick_params(labelsize=TICK_FONT_SIZE)

    norm2 = matplotlib.colors.Normalize(vmin=c2.cvalues.min(), vmax=c2.cvalues.max())
    sm2 = plt.cm.ScalarMappable(norm=norm2, cmap=c2.cmap)
    sm2.set_array([])
    cb2 = plt.colorbar(sm2, ax=axs[1, 0], orientation='vertical', fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=TICK_FONT_SIZE)

    # Plot 4: Pressure contours and velocity vectors
    c3 = axs[1, 1].contour(x, y, cp, 40, cmap=cm.jet, linewidths=0.5)
    axs[1, 1].quiver(X, Y, vel_x, vel_y, scale=20, zorder=10)
    axs[1, 1].set_aspect('equal', 'box')
    axs[1, 1].set_xlim(xmin, xmax)
    axs[1, 1].set_ylim(xmin, xmax)
    axs[1, 1].set_title(f'{pattern}: Pressure Contours and Velocity Vectors', fontsize=TITLE_FONT_SIZE)
    axs[1, 1].tick_params(labelsize=TICK_FONT_SIZE)

    norm3 = matplotlib.colors.Normalize(vmin=c3.cvalues.min(), vmax=c3.cvalues.max())
    sm3 = plt.cm.ScalarMappable(norm=norm3, cmap=c3.cmap)
    sm3.set_array([])
    cb3 = plt.colorbar(sm3, ax=axs[1, 1], orientation='vertical', fraction=0.046, pad=0.04)
    cb3.ax.tick_params(labelsize=TICK_FONT_SIZE)

    plt.tight_layout()

    fig.savefig(f'../fig/project1_part2_{pattern.lower()}.png', bbox_inches='tight')

# Plot each of the cases: source, vortex, and doublet
plot_all(X, Y, x, y, vel_source_x, vel_source_y, vel_source, cp_source, 'Source')
plot_all(X, Y, x, y, vel_vortex_x, vel_vortex_y, vel_vortex, cp_vortex, 'Vortex')
plot_all(X, Y, x, y, vel_doublet_x, vel_doublet_y, vel_doublet, cp_doublet, 'Doublet')

plt.show()
