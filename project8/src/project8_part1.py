"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 8: Transonic flow and boundary layers
Part 1: Transonic flow over symmetric airfoils
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from matplotlib import cm

sys.path.insert(1, '../..')

from tools.tridiagscalar import tridiagscalar

# ny must be even
nx = 100
ny = 100

xmin = -4
xmax = -xmin
ymin = 0
ymax = 24

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

nLE = round(2 * nx / 5)
nTE = round(3 * nx / 5) + 1

# tau - thickness ratio of parabolic profiles and NACA
# sh - vertical shift for parabolic profiles points on grid
# coe - 'A' coefficient for parabolic profiles points on grid
# chord - chord length of airfoil points on grid
tau = 0.1
sh = tau * x[nTE - 1]
coe = sh / x[nLE - 1] ** 2
chord = (x[nTE - 1] - x[nLE - 1])

# Calculation and use of half_chord below required for numeric stability
# TODO@dpwiese - look at this and figure out exactly why linspace is behaving this way
half_chord = -x[nLE - 1]

# For diamond airfoil
diamond_slope = tau

####################################################################################################

airfoil_profile_biconvex = np.zeros(nx)
airfoil_profile_diamond = np.zeros(nx)
airfoil_profile_naca = np.zeros(nx)
airf_prof_curvature = np.zeros(nx)

# Biconvex
for i in range(nLE - 1, nTE):
    airfoil_profile_biconvex[i] = -coe * x[i] ** 2 + sh

# Diamond Airfoil (first half)
for i in range(nLE - 1, nx // 2):
    airfoil_profile_diamond[i] = diamond_slope * x[i] + tau * half_chord

# Diamond Airfoil (second half)
for i in range(nx // 2, nTE):
    airfoil_profile_diamond[i] = -diamond_slope * x[i] + tau * half_chord

# NACA 00xx
for i in range(nLE - 1, nTE):
    temp_arg = (x[i] + half_chord) / chord
    airfoil_profile_naca[i] = 5 * tau * chord * (0.2969 * np.sqrt(temp_arg) - \
        0.1260 * (temp_arg) - \
        0.3537 * (temp_arg) ** 2 + \
        0.2843 * (temp_arg) ** 3 - \
        0.1015 * (temp_arg) ** 4)

####################################################################################################

u = np.zeros((nx, ny))
uold = np.zeros((nx, ny))

Au = np.zeros(ny)
Bu = np.zeros(ny)
Cu = np.zeros(ny)
Du = np.zeros(ny)

# mach_inf: [0.8, 0.9, 0.98, 1.08, 1.14, 1.4]

configs = [
    {
        'airfoil_profile':  airfoil_profile_naca,
        'mach_inf':         0.8,
        'name':             'naca',
        'iteratormax':      30,
        'iterator2max':     100,
        'omega':            0.9
    },
    {
        'airfoil_profile':  airfoil_profile_biconvex,
        'mach_inf':         0.8,
        'name':             'biconvex',
        'iteratormax':      30,
        'iterator2max':     100,
        'omega':            0.9
    },
    {
        'airfoil_profile':  airfoil_profile_diamond,
        'mach_inf':         0.9,
        'name':             'diamond',
        'iteratormax':      200,
        'iterator2max':     20,
        'omega':            0.99
    },
    {
        'airfoil_profile':  airfoil_profile_naca,
        'mach_inf':         0.9,
        'name':             'naca',
        'iteratormax':      200,
        'iterator2max':     40,
        'omega':            0.9
    },
    {
        'airfoil_profile':  airfoil_profile_biconvex,
        'mach_inf':         0.98,
        'name':             'biconvex',
        'iteratormax':      30,
        'iterator2max':     2,
        'omega':            0.9
    },
    {
        'airfoil_profile':  airfoil_profile_biconvex,
        'mach_inf':         1.08,
        'name':             'biconvex',
        'iteratormax':      400,
        'iterator2max':     40,
        'omega':            0.99
    },
    {
        'airfoil_profile':  airfoil_profile_diamond,
        'mach_inf':         1.14,
        'name':             'diamond',
        'iteratormax':      100,
        'iterator2max':     20,
        'omega':            0.999
    },
    {
        'airfoil_profile':  airfoil_profile_naca,
        'mach_inf':         1.4,
        'name':             'naca',
        'iteratormax':      100,
        'iterator2max':     20,
        'omega':            0.99
    }
]

####################################################################################################
# Solve for U

for cfg in configs:

    omega = cfg['omega']
    mach_inf = cfg['mach_inf']

    # Calculate numerical second derivative
    for i in range(nLE - 1, nTE):
        airf_prof_curvature[i] = (cfg['airfoil_profile'][i + 1] - 2 * cfg['airfoil_profile'][i] + cfg['airfoil_profile'][i - 1]) / dx ** 2

    # Value
    temp_val_1 = 1 - mach_inf ** 2

    iterator2 = 0

    while iterator2 < cfg['iterator2max']:
        print(iterator2)
        if iterator2 == 0:
            gamma = -1
        else:
            gamma = 1.4

        for i in range(2, nx - 1):

            iterator = 0

            while iterator < cfg['iteratormax']:

                for j in [0]:
                    fm2 = temp_val_1 * u[i - 2, j] - ((gamma + 1) * mach_inf ** 2 * u[i - 2, j] ** 2) / 2
                    fm1 = temp_val_1 * u[i - 1, j] - ((gamma + 1) * mach_inf ** 2 * u[i - 1, j] ** 2) / 2
                    fm0 = temp_val_1 * u[i,     j] - ((gamma + 1) * mach_inf ** 2 * u[i,     j] ** 2) / 2
                    fp1 = temp_val_1 * u[i + 1, j] - ((gamma + 1) * mach_inf ** 2 * u[i + 1, j] ** 2) / 2

                    TKm2 = temp_val_1 - (gamma + 1) * mach_inf ** 2 * u[i - 2, j]
                    TKm1 = temp_val_1 - (gamma + 1) * mach_inf ** 2 * u[i - 1, j]
                    TKm0 = temp_val_1 - (gamma + 1) * mach_inf ** 2 * u[i,     j]
                    TKp1 = temp_val_1 - (gamma + 1) * mach_inf ** 2 * u[i + 1, j]

                    # Subsonic
                    if TKm0 > 0 and TKm1 > 0:
                        Au[j] = 0
                        Bu[j] = -1 / dy ** 2 - 2 * TKm0 / dx ** 2
                        Cu[j] = 1 / dy ** 2
                        Du[j] = (-1 / dx ** 2) * (TKp1 * u[i + 1, j] + TKm1 * u[i - 1, j]) - \
                                (fp1 - 2 * fm0 + fm1) / dx ** 2 + \
                                (TKm1 * u[i - 1, j] - 2 * TKm0 * u[i, j] + TKp1 * u[i + 1, j]) / dx ** 2 + \
                                airf_prof_curvature[i] / dy

                    # Supersonic
                    if TKm0 < 0 and TKm1 < 0:
                        Au[j] = 0
                        Bu[j] = -1 / dy ** 2 + TKm0 / dx ** 2
                        Cu[j] = 1 / dy ** 2
                        Du[j] = (-1 / dx ** 2) * (-2 * TKm1 * u[i - 1, j] + TKm2 * u[i - 2, j]) - \
                                (fm0 - 2 * fm1 + fm2) / dx ** 2 + \
                                (TKm0 * u[i, j] - 2 * TKm1 * u[i - 1, j] + TKm2 * u[i - 2, j]) / dx ** 2 + \
                                airf_prof_curvature[i] / dy

                    # Parabolic
                    if TKm0 < 0 and TKm1 > 0:
                        Au[j] = 0
                        Bu[j] = -1 / dy ** 2
                        Cu[j] = 1 / dy ** 2
                        Du[j] = 0

                    # Shockpoint
                    if TKm0 > 0 and TKm1 < 0:
                        Au[j] = 0
                        Bu[j] = -1 / dy ** 2
                        Cu[j] = 1 / dy ** 2
                        Du[j] = -(fp1 - fm0) / dx ** 2

                for j in range(1, ny - 1):
                    fm2 = temp_val_1 * u[i - 2, j] - ((gamma + 1) * mach_inf ** 2 * u[i - 2, j] ** 2) / 2
                    fm1 = temp_val_1 * u[i - 1, j] - ((gamma + 1) * mach_inf ** 2 * u[i - 1, j] ** 2) / 2
                    fm0 = temp_val_1 * u[i,     j] - ((gamma + 1) * mach_inf ** 2 * u[i,     j] ** 2) / 2
                    fp1 = temp_val_1 * u[i + 1, j] - ((gamma + 1) * mach_inf ** 2 * u[i + 1, j] ** 2) / 2

                    TKm2 = temp_val_1 - (gamma + 1) * mach_inf ** 2 * u[i - 2, j]
                    TKm1 = temp_val_1 - (gamma + 1) * mach_inf ** 2 * u[i - 1, j]
                    TKm0 = temp_val_1 - (gamma + 1) * mach_inf ** 2 * u[i,     j]
                    TKp1 = temp_val_1 - (gamma + 1) * mach_inf ** 2 * u[i + 1, j]

                    # Subsonic
                    if TKm0 > 0 and TKm1 > 0:
                        Au[j] = 1 / dy ** 2
                        Bu[j] = -2 / dy ** 2 - 2 * TKm0 / dx ** 2
                        Cu[j] = 1 / dy ** 2
                        Du[j] = (-1 / dx ** 2) * (TKp1 * u[i + 1, j] + TKm1 * u[i - 1, j]) - \
                                (fp1 - 2 * fm0 + fm1) / dx ** 2 + \
                                (TKm1 * u[i - 1, j] - 2 * TKm0 * u[i, j] + TKp1 * u[i + 1, j]) / dx ** 2

                    # Supersonic
                    if TKm0 < 0 and TKm1 < 0:
                        Au[j] = 1 / dy ** 2
                        Bu[j] = -2 / dy ** 2 + TKm0 / dx ** 2
                        Cu[j] = 1 / dy ** 2
                        Du[j] = (-1 / dx ** 2) * (-2 * TKm1 * u[i - 1, j] + TKm2 * u[i - 2, j]) - \
                                (fm0 - 2 * fm1 + fm2) / dx ** 2 + \
                                (TKm0 * u[i, j] - 2 * TKm1 * u[i - 1, j] + TKm2 * u[i - 2, j]) / dx ** 2

                    # Parabolic
                    if TKm0 < 0 and TKm1 > 0:
                        Au[j] = 1 / dy ** 2
                        Bu[j] = -2 / dy ** 2
                        Cu[j] = 1 / dy ** 2
                        Du[j] = 0

                    # Shockpoint
                    if TKm0 > 0 and TKm1 < 0:
                        Au[j] = 1 / dy ** 2
                        Bu[j] = -2 / dy ** 2
                        Cu[j] = 1 / dy ** 2
                        Du[j] = -(fp1 - fm0) / dx ** 2

                for j in [ny - 1]:
                    Au[j] = 0
                    Bu[j] = 1
                    Cu[j] = 0
                    Du[j] = 0

                u[i, :] = tridiagscalar(Au, Bu, Cu, Du)

                # This is necessary to enforce some numeric stability
                # TODO@dpwiese - figure out what is going on
                u = np.clip(u, -10, 10)

                u[i, :] = uold[i, :] + omega * (u[i, :] - uold[i, :])
                uold[i, :] = u[i, :]

                iterator += 1

        iterator2 += 1

    u[nx - 1, :] = u[nx - 2, :]

    ################################################################################################
    # Plot results

    fig = plt.figure(figsize=(10, 6), dpi=200, facecolor='w', edgecolor='k')

    TITLE_FONT_SIZE = 10
    TICK_FONT_SIZE = 8

    plot_title1 = f'Transonic Flow ({cfg["name"]}): Ma = {cfg["mach_inf"]:.2f}, omega = {omega:.3f}, iter2 = {cfg["iterator2max"]}, iter= {cfg["iteratormax"]}'

    c1 = plt.contour(x, y, u.T, 200, cmap=cm.jet)
    plt.contour(x, -y, u.T, 200, cmap=cm.jet)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([xmin, xmax, xmin, xmax])
    plt.plot(x, cfg['airfoil_profile'], '-k', zorder=11)
    plt.plot(x, -cfg['airfoil_profile'], '-k', zorder=11)
    plt.fill_between(x, cfg['airfoil_profile'], color='r', zorder=10)
    plt.fill_between(x, -cfg['airfoil_profile'], color='r', zorder=10)
    plt.title(plot_title1, fontsize=TITLE_FONT_SIZE)
    plt.xlabel('u-direction')
    plt.ylabel('v-direction')
    plt.grid(True)
    plt.gca().tick_params(labelsize=TICK_FONT_SIZE)

    norm1 = matplotlib.colors.Normalize(vmin=c1.cvalues.min(), vmax=c1.cvalues.max())
    sm1 = plt.cm.ScalarMappable(norm=norm1, cmap=c1.cmap)
    sm1.set_array([])
    cb1 = plt.colorbar(sm1, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=TICK_FONT_SIZE)

    plt.tight_layout()
    plt.savefig(f'../fig/project8_trans_{cfg["name"]}_ma{str(cfg["mach_inf"]).replace(".", "")}_iter2_{cfg["iterator2max"]}_iter_{cfg["iteratormax"]}_omega_{str(omega).replace(".", "")}.png', bbox_inches='tight')

# Show all plots
plt.show()
