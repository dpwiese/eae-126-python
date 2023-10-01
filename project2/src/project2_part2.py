"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 2: Joukowski transformation and airfoils
Part 2: Joukowski transformation: flat plate, parabolic arc, symmetric airfoil, cambered airfoil

Find the velocity components and Cp in circle plane and in transformed plane for the four shapes at
angle of attack, 0 and 5 degrees. Plot velocity vectors, pressure contours, and the surface pressure
distribution. Calculate the lift, drag, moment, and center of pressure with and without enforcing
Kutta condition.

for ellipse: epsilon and mu were both set to zero, and b=0.9
"""

import numpy as np
import matplotlib.pyplot as plt

# Set up geometric parameters of the cylinder, flow parameters, and grid spacing
nr = 50
rmin = 1
rmax = 7
r = np.linspace(rmin, rmax, nr)
dr = (rmax - rmin) / (nr - 1)

ntheta = 4 * 20
dtheta = 2 * np.pi / ntheta
thetamin = 0.5 * dtheta
thetamax = 2 * np.pi - dtheta
theta = np.linspace(thetamin, thetamax, ntheta)

vinf = 1
pinf = 0
rho = 1

# Define configurations (Note: b = np.sqrt(a**2 - mu**2) + epsilon, except for ellipse and biconvex?)
configs = [
    {'a': 1,    'epsilon': 0,       'mu': 0,    'b': 0.9,   'alphadeg': 0,  'name': 'ellipse'},
    {'a': 1,    'epsilon': 0,       'mu': 0,    'b': 0.9,   'alphadeg': 5,  'name': 'ellipse at aoa'},
    {'a': 1,    'epsilon': 0,       'mu': 0,    'b': 1,     'alphadeg': 0,  'name': 'flat plate'},
    {'a': 1,    'epsilon': 0,       'mu': 0,    'b': 1,     'alphadeg': 5,  'name': 'flat plate at aoa'},
    {'a': 1,    'epsilon': 0,       'mu': 0.1,  'b': 0.99,  'alphadeg': 0,  'name': 'circular arc'},
    {'a': 1,    'epsilon': 0,       'mu': 0.1,  'b': 0.99,  'alphadeg': 5,  'name': 'circular arc at aoa'},
    {'a': 1,    'epsilon': -0.1,    'mu': 0.0,  'b': 0.9,   'alphadeg': 0,  'name': 'symmetric airfoil'},
    {'a': 1,    'epsilon': -0.1,    'mu': 0.0,  'b': 0.9,   'alphadeg': 5,  'name': 'symmetric airfoil at aoa'},
    {'a': 1,    'epsilon': -0.1,    'mu': 0.1,  'b': 0.89,  'alphadeg': 0,  'name': 'cambered airfoil'},
    {'a': 1,    'epsilon': -0.1,    'mu': 0.1,  'b': 0.89,  'alphadeg': 5,  'name': 'cambered airfoil at aoa'}
]

for idx, cfg in enumerate(configs):

    # Parameters for each transformation
    a           = cfg['a']
    epsilon     = cfg['epsilon']
    mu          = cfg['mu']
    b           = cfg['b']
    alphadeg    = cfg['alphadeg']

    alpha = np.deg2rad(alphadeg)

    D = vinf * rmin**2
    thetasep = -alpha - np.arcsin(mu / a)

    # The formula for Gamma enforces the Kutta condition
    Gamma = 2 * np.pi * vinf * rmin * np.sin(thetasep)

    # Initialize arrays
    ubar    = np.zeros((ntheta, nr))
    vbar    = np.zeros((ntheta, nr))
    x       = np.zeros((ntheta, nr))
    y       = np.zeros((ntheta, nr))
    ubarx   = np.zeros((ntheta, nr))
    ubary   = np.zeros((ntheta, nr))
    vbarx   = np.zeros((ntheta, nr))
    vbary   = np.zeros((ntheta, nr))
    u       = np.zeros((ntheta, nr))
    v       = np.zeros((ntheta, nr))
    vtot    = np.zeros((ntheta, nr))
    cp      = np.zeros((ntheta, nr))
    p       = np.zeros((ntheta, nr))
    xair    = np.zeros((ntheta, nr))
    yair    = np.zeros((ntheta, nr))
    X       = np.zeros((ntheta, nr))
    Y       = np.zeros((ntheta, nr))
    dXdx    = np.zeros((ntheta, nr))
    dYdy    = np.zeros((ntheta, nr))
    dXdy    = np.zeros((ntheta, nr))
    dYdx    = np.zeros((ntheta, nr))
    U       = np.zeros((ntheta, nr))
    V       = np.zeros((ntheta, nr))
    VTOT    = np.zeros((ntheta, nr))
    CP      = np.zeros((ntheta, nr))
    P       = np.zeros((ntheta, nr))
    DX      = np.zeros((ntheta, 1))
    DY      = np.zeros((ntheta, 1))
    DS      = np.zeros((ntheta, 1))
    dx      = np.zeros((ntheta, 1))
    dy      = np.zeros((ntheta, 1))
    ds      = np.zeros((ntheta, 1))
    F       = np.zeros((ntheta, 1))
    f       = np.zeros((ntheta, 1))
    CF      = np.zeros((ntheta, 1))
    cf      = np.zeros((ntheta, 1))
    FX      = np.zeros((ntheta, 1))
    FY      = np.zeros((ntheta, 1))
    fx      = np.zeros((ntheta, 1))
    fy      = np.zeros((ntheta, 1))
    CFX     = np.zeros((ntheta, 1))
    CFY     = np.zeros((ntheta, 1))
    cfx     = np.zeros((ntheta, 1))
    cfy     = np.zeros((ntheta, 1))
    CLSUM   = np.zeros(ntheta)
    CDSUM   = np.zeros(ntheta)

    # Calculate values
    for i in range(ntheta):
        for j in range(nr):
            ubar[i, j] = vinf * np.cos(theta[i] - alpha) - D * np.cos(theta[i] - alpha) / (r[j]**2)
            vbar[i, j] = -vinf * np.sin(theta[i] - alpha) - D * np.sin(theta[i] - alpha) / (r[j]**2) + Gamma / (2 * np.pi * r[j])

    for i in range(ntheta):
        for j in range(nr):
            x[i, j] = r[j] * np.cos(theta[i])
            y[i, j] = r[j] * np.sin(theta[i])

    for i in range(ntheta):
        for j in range(nr):
            ubarx[i, j] = ubar[i, j] * np.cos(theta[i])
            ubary[i, j] = ubar[i, j] * np.sin(theta[i])
            vbarx[i, j] = -vbar[i, j] * np.sin(theta[i])
            vbary[i, j] = vbar[i, j] * np.cos(theta[i])

    for i in range(ntheta):
        for j in range(nr):
            u[i, j] = ubarx[i, j] + vbarx[i, j]
            v[i, j] = ubary[i, j] + vbary[i, j]
            vtot[i, j] = np.sqrt(u[i, j]**2 + v[i, j]**2)
            cp[i, j] = 1 - (vtot[i, j]**2) / (vinf**2)
            p[i, j] = 0.5 * cp[i, j] * rho * vinf**2 + pinf

    for i in range(ntheta):
        for j in range(nr):
            xair[i, j] = x[i, j] + epsilon
            yair[i, j] = y[i, j] + mu

    for i in range(ntheta):
        for j in range(nr):
            X[i, j] = xair[i, j] * (1 + (b**2) / ((xair[i, j])**2 + (yair[i, j])**2))
            Y[i, j] = yair[i, j] * (1 - (b**2) / ((xair[i, j])**2 + (yair[i, j])**2))
            dXdx[i, j] = 1 - 2 * (xair[i, j]**2) * (b**2) / (xair[i, j]**2 + yair[i, j]**2)**2 + (b**2) / (xair[i, j]**2 + yair[i, j]**2)
            dYdy[i, j] = 1 + 2 * (yair[i, j]**2) * (b**2) / (xair[i, j]**2 + yair[i, j]**2)**2 - (b**2) / (xair[i, j]**2 + yair[i, j]**2)
            dXdy[i, j] = -2 * (b**2) * xair[i, j] * yair[i, j] / (xair[i, j]**2 + yair[i, j]**2)**2
            dYdx[i, j] = 2 * (b**2) * xair[i, j] * yair[i, j] / (xair[i, j]**2 + yair[i, j]**2)**2

    for i in range(ntheta):
        for j in range(nr):
            U[i, j] = (u[i, j] * dYdy[i, j] - dYdx[i, j] * v[i, j]) / (dXdx[i, j] * dYdy[i, j] - dYdx[i, j] * dXdy[i, j])
            V[i, j] = (dXdx[i, j] * v[i, j] - u[i, j] * dXdy[i, j]) / (dXdx[i, j] * dYdy[i, j] - dYdx[i, j] * dXdy[i, j])

    for i in range(ntheta):
        for j in range(nr):
            VTOT[i, j] = np.sqrt(U[i, j]**2 + V[i, j]**2)
            CP[i, j] = 1 - VTOT[i, j]**2 / vinf**2
            P[i, j] = 0.5 * CP[i, j] * rho * vinf**2 + pinf

    for i in range(ntheta - 1):
        DX[i, 0] = X[i + 1, 0] - X[i, 0]
        DY[i, 0] = Y[i + 1, 0] - Y[i, 0]
        DS[i, 0] = np.sqrt(DX[i, 0]**2 + DY[i, 0]**2)
        dx[i, 0] = x[i + 1, 0] - x[i, 0]
        dy[i, 0] = y[i + 1, 0] - y[i, 0]
        ds[i, 0] = np.sqrt(dx[i, 0]**2 + dy[i, 0]**2)

    for i in range(ntheta):
        F[i, 0] = -P[i, 0] * DS[i, 0]
        f[i, 0] = -p[i, 0] * ds[i, 0]
        CF[i, 0] = -CP[i, 0] * DS[i, 0]
        cf[i, 0] = -cp[i, 0] * ds[i, 0]

    for i in range(ntheta):
        FX[i, 0] = F[i, 0] * (DY[i, 0] / DS[i, 0])
        FY[i, 0] = -F[i, 0] * (DX[i, 0] / DS[i, 0])
        fx[i, 0] = f[i, 0] * (dy[i, 0] / ds[i, 0])
        fy[i, 0] = -f[i, 0] * (dx[i, 0] / ds[i, 0])
        CFX[i, 0] = CF[i, 0] * (DY[i, 0] / DS[i, 0])
        CFY[i, 0] = -CF[i, 0] * (DX[i, 0] / DS[i, 0])
        cfx[i, 0] = cf[i, 0] * (dy[i, 0] / ds[i, 0])
        cfy[i, 0] = -cf[i, 0] * (dx[i, 0] / ds[i, 0])

    for i in range(ntheta):
        CLSUM[i] = -rmin * CP[i, 0] * np.sin(theta[i]) * dtheta
        CDSUM[i] = -0.5 * rmin * CP[i, 0] * (X[i, 0] / (np.sqrt(X[i, 0]**2 + Y[i, 0]**2))) * dtheta

    CL = np.sum(CLSUM)
    CD = np.sum(CDSUM)
    LIFT = 0.5 * CL * rho * vinf**2

    M0MSUM = np.zeros((ntheta, 1))
    CMSUM = np.zeros((ntheta, 1))

    for i in range(ntheta):
        M0MSUM[i, 0] = -FX[i, 0] * (Y[i, 0] - Y[round(ntheta / 2), 0]) + FY[i, 0] * (X[i, 0] - X[round(ntheta / 2), 0])
        CMSUM[i, 0] = -CFX[i, 0] * (Y[i, 0] - Y[round(ntheta / 2), 0]) + CFY[i, 0] * (X[i, 0] - X[round(ntheta / 2), 0])

    MOM = np.sum(M0MSUM)
    CM = np.sum(CMSUM)

    config_text_string =    f'alpha = {alphadeg}deg, ' \
                            f'v_inf = {vinf}, ' \
                            f'rho_inf = {rho}, ' \
                            f'a = {a}, ' \
                            f'epsilon = {epsilon}, ' \
                            f'mu = {mu}, ' \
                            f'b = {b:0.2f}'

    # Cylinder plots
    fig1, axs = plt.subplots(3, 2, figsize=(16, 10), dpi=100, facecolor='w', edgecolor='k')

    # Figure 1
    contours = axs[0, 0].contour(x, y, vtot, levels=200, colors='black')
    axs[0, 0].clabel(contours, inline=1, fontsize=8)
    axs[0, 0].plot(x[:, 0], y[:, 0], '-k', linewidth=2)
    axs[0, 0].fill(x[:, 0], y[:, 0], 'r')
    axs[0, 0].set_title('Velocity Contours Around Cylinder')
    axs[0, 0].set_xlabel('x-axis: u flow direction')
    axs[0, 0].set_ylabel('y-axis: v flow direction')
    plt.colorbar(contours)
    axs[0, 0].set_aspect('equal', adjustable='box')
    axs[0, 0].text(0, 2.7, config_text_string, horizontalalignment='center', backgroundcolor='w')

    # Figure 2
    contours = plt.contour(x, y, cp, levels=200, colors='black')
    axs[0, 1].clabel(contours, inline=1, fontsize=8)
    axs[0, 1].plot(x[:, 0], y[:, 0], '-k', linewidth=2)
    axs[0, 1].fill(x[:, 0], y[:, 0], 'r')
    axs[0, 1].set_title('Cp Surface Contours On Cylinder')
    axs[0, 1].set_xlabel('x-axis: U flow direction')
    axs[0, 1].set_ylabel('y-axis: V flow direction')
    plt.colorbar(contours)
    axs[0, 1].set_aspect('equal', adjustable='box')
    axs[0, 1].text(0, 2.7, config_text_string, horizontalalignment='center', backgroundcolor='w')

    # Figure 3
    axs[1, 0].plot(x[:, 0], y[:, 0], '-k', linewidth=2)
    axs[1, 0].fill(x[:, 0], y[:, 0], 'r')
    axs[1, 0].quiver(x, y, u, v, scale=1, width=0.002)
    axs[1, 0].set_title('Velocity Vectors Around Cylinder')
    axs[1, 0].set_xlabel('x-axis: u flow direction')
    axs[1, 0].set_ylabel('y-axis: v flow direction')
    axs[1, 0].axis([-3, 3, -3, 3])
    axs[1, 0].set_aspect('equal', adjustable='box')
    axs[1, 0].text(0, 2.7, config_text_string, horizontalalignment='center', backgroundcolor='w')

    # Figure 4
    axs[1, 1].plot(x[:, 0], -cp[:, 0], linewidth=2)
    axs[1, 1].set_title('Cp Distribution Versus X For Cylinder')
    axs[1, 1].set_xlabel('x-axis: X')
    axs[1, 1].set_ylabel('y-axis: CP')
    axs[1, 1].axis([-1, 1, -1, 6])
    axs[1, 1].text(0, 5.5, config_text_string, horizontalalignment='center', backgroundcolor='w')
    axs[1, 1].grid(True)

    # Figure 5
    axs[2, 1].quiver(x[:, 0], y[:, 0], fx[:, 0], fy[:, 0], width=0.002)
    axs[2, 1].plot(x[:, 0], y[:, 0], '-k', linewidth=2)
    axs[2, 1].set_title('Forces on Cylinder Surface')
    axs[2, 1].set_xlabel('x-axis: u flow direction')
    axs[2, 1].set_ylabel('y-axis: v flow direction')
    axs[2, 1].axis([-1.5, 1.5, -1.5, 1.5])
    axs[2, 1].text(0, -1.3, config_text_string, horizontalalignment='center', backgroundcolor='w')
    axs[2, 1].grid(True)
    axs[2, 1].set_aspect('equal', adjustable='box')

    # Airfoil plots
    fig2, axs = plt.subplots(3, 2, figsize=(16, 10), dpi=100, facecolor='w', edgecolor='k')

    # Plot 6: Velocity contours
    contours = axs[0, 0].contour(X, Y, VTOT, 1000, linewidths=0.4)
    axs[0, 0].plot(X[:,0], Y[:,0], '-k', linewidth=2)
    axs[0, 0].fill(X[:,0], Y[:,0], '-r')
    axs[0, 0].set_title('Velocity Contours Around Airfoil')
    axs[0, 0].set_xlabel('x-axis: u flow direction')
    axs[0, 0].set_ylabel('y-axis: v flow direction')
    axs[0, 0].text(0, 2.7, config_text_string, horizontalalignment='center', backgroundcolor='w')
    plt.colorbar(contours)
    axs[0, 0].axis([-3, 3, -2, 2])
    axs[0, 0].set_aspect('equal', adjustable='box')

    # Plot 7: Pressure contours
    contours = axs[0, 1].contour(X, Y, CP, 4000, linewidths=0.4)
    axs[0, 1].plot(X[:,0], Y[:,0], '-k', linewidth=2)
    axs[0, 1].fill(X[:,0], Y[:,0], '-r')
    axs[0, 1].set_title('C_p Surface Contours On Airfoil')
    axs[0, 1].set_xlabel('x-axis: U flow direction')
    axs[0, 1].set_ylabel('y-axis: V flow direction')
    axs[0, 1].text(0, 2.7, config_text_string, horizontalalignment='center', backgroundcolor='w')
    plt.colorbar(contours)
    axs[0, 1].axis([-3, 3, -2, 2])
    axs[0, 1].set_aspect('equal', adjustable='box')

    # Plot 8: Velocity vectors
    axs[1, 0].plot(X[:,0], Y[:,0], '-k', linewidth=2)
    axs[1, 0].fill(X[:,0], Y[:,0], '-r')
    axs[1, 0].quiver(X, Y, U, V, scale=40, width=0.001)
    axs[1, 0].set_title('Velocity Vectors Around Airfoil')
    axs[1, 0].set_xlabel('x-axis: U flow direction')
    axs[1, 0].set_ylabel('y-axis: V flow direction')
    axs[1, 0].text(0, 2.7, config_text_string, horizontalalignment='center', backgroundcolor='w')
    axs[1, 0].axis([-3, 3, -2, 2])
    axs[1, 0].set_aspect('equal', adjustable='box')

    # Plot 9: Cp distribution
    axs[1, 1].plot(X[:,0], -CP[:,0], linewidth=2)
    axs[1, 1].set_title('C_p Distribution Versus X For Airfoil')
    axs[1, 1].set_xlabel('x-axis: X')
    axs[1, 1].set_ylabel('y-axis: CP')
    axs[1, 1].axis([-2, 2, -2, 8])
    axs[1, 1].text(0, -1.5, config_text_string, horizontalalignment='center', backgroundcolor='w')
    axs[1, 1].grid(True)

    # Plot 10: Forces
    axs[2, 1].quiver(X[:,0], Y[:,0], FX[:,0], FY[:,0], width=0.002)
    axs[2, 1].plot(X[:,0], Y[:,0], '-k', linewidth=2)
    axs[2, 1].set_title('Forces on Airfoil Surface')
    axs[2, 1].set_xlabel('x-axis: U flow direction')
    axs[2, 1].set_ylabel('y-axis: V flow direction')
    axs[2, 1].text(0, 1.3, config_text_string, horizontalalignment='center', backgroundcolor='w')
    axs[2, 1].grid(True)
    axs[2, 1].axis([-3, 3, -1, 1])
    axs[2, 1].set_aspect('equal', adjustable='box')

    fig1.tight_layout()
    fig1.savefig(f'../fig/project2_part2a.png', bbox_inches='tight')

    fig2.tight_layout()
    fig2.savefig(f'../fig/project2_part2b.png', bbox_inches='tight')

# Show all plots
plt.show()
