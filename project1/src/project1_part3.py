"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 1: Steady, inviscid, adiabatic, incompressible, and irrotational 2D flows over cylinder
Part 3: Flow field for Rankine body, Kelvin oval

Derive the velocity for a Rankine and Kelvin body in uniform flow. Plot the velocity vectors and
velocity contours for each case. Calculate the pressure coefficient and plot it.
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

rho = 1
Q = 50
Gamma = 50
vinf = 10
a = 0.23

r = np.sqrt(X**2 + Y**2)
cosine = X / r
sine = Y / r

uinf = np.full_like(X, vinf)

####################################################################################################

rsorc = np.sqrt((a + X)**2 + Y**2)
rsink = np.sqrt((X - a)**2 + Y**2)
rcw = np.sqrt((a + X)**2 + Y**2)
rccw = np.sqrt((X - a)**2 + Y**2)

####################################################################################################
# This section generates velocity vectors and pressure contours for a doublet with uniform flow
# around it: Rankine Body

vsorc = Q / (2 * np.pi * rho * rsorc)
vsink = -Q / (2 * np.pi * rho * rsink)

vsorcx = vsorc * ((X + a) / rsorc)
vsorcy = vsorc * (Y / rsorc)
vsinkx = vsink * ((X - a) / rsink)
vsinky = vsink * (Y / rsink)

vrankx = vsorcx + vsinkx + uinf
vranky = vsorcy + vsinky

vrank = np.sqrt(vrankx**2 + vranky**2)
prank = 1 - (vrank**2 / vinf**2)

vrankx = np.clip(vrankx, -50, 50)
vranky = np.clip(vranky, -50, 50)
prank = np.clip(prank, -10, 10)

####################################################################################################
# This section generates velocity vectors and pressure contours for a
# two vortices with uniform flow around it: Kelvin Oval

vcw = Gamma / (2 * np.pi * rcw)
vccw = Gamma / (2 * np.pi * rccw)

vcwx = vcw * (Y / rcw)
vcwy = -vcw * ((X + a) / rcw)
vccwx = -vccw * (Y / rccw)
vccwy = vccw * ((X - a) / rccw)

vkelvx = vccwx + vcwx
vkelvy = vccwy + vcwy + uinf

vkelv = np.sqrt(vkelvx**2 + vkelvy**2)
pkelv = 1 - (vkelv**2 / vinf**2)

vkelvx = np.clip(vkelvx, -50, 50)
vkelvy = np.clip(vkelvy, -50, 50)
pkelv = np.clip(pkelv, -10, 10)

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
    sm1 = plt.cm.ScalarMappable(norm=norm1, cmap = c1.cmap)
    sm1.set_array([])

    # Set the colorbar to the first subplot (for spacing only), remove it, and set the colorbar on
    # the intended subplot.
    cb1 = plt.colorbar(c1, ax=axs[0, 0], orientation='vertical', fraction=0.046, pad=0.04)
    cb1.remove()
    cb1 = plt.colorbar(sm1, ax=axs[0, 1], orientation='vertical', fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=TICK_FONT_SIZE)

    # Plot 1: Velocity Vectors
    axs[0, 0].quiver(X, Y, vel_x, vel_y, scale=400)
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
    sm2 = plt.cm.ScalarMappable(norm=norm2, cmap = c2.cmap)
    sm2.set_array([])
    cb2 = plt.colorbar(sm2, ax=axs[1, 0], orientation='vertical', fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=TICK_FONT_SIZE)

    # Plot 4: Pressure contours and velocity vectors
    c3 = axs[1, 1].contour(x, y, cp, 40, cmap=cm.jet, linewidths=0.5)
    axs[1, 1].quiver(X, Y, vel_x, vel_y, scale=400, zorder=10)
    axs[1, 1].set_aspect('equal', 'box')
    axs[1, 1].set_xlim(xmin, xmax)
    axs[1, 1].set_ylim(xmin, xmax)
    axs[1, 1].set_title(f'{pattern}: Pressure Contours and Velocity Vectors', fontsize=TITLE_FONT_SIZE)
    axs[1, 1].tick_params(labelsize=TICK_FONT_SIZE)

    norm3 = matplotlib.colors.Normalize(vmin=c2.cvalues.min(), vmax=c2.cvalues.max())
    sm3 = plt.cm.ScalarMappable(norm=norm3, cmap = c3.cmap)
    sm3.set_array([])
    cb3 = plt.colorbar(sm3, ax=axs[1, 1], orientation='vertical', fraction=0.046, pad=0.04)
    cb3.ax.tick_params(labelsize=TICK_FONT_SIZE)

    plt.tight_layout()

    fig.savefig(f'../fig/project1_part3_{pattern.lower().replace(" ", "_")}.png', bbox_inches='tight')

# Plot Rankine Body
plot_all(X, Y, x, y, vrankx, vranky, vrank, prank, 'Rankine Body')

# Plot Kelvin Oval
plot_all(X, Y, x, y, vkelvx, vkelvy, vkelv, pkelv, 'Kelvin Oval')

plt.show()
