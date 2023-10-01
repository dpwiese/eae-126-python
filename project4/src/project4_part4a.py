"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 4: High and Low Aspect Ratio Wings
Part 4: Lift and Induced Drag of Crescent Wings
"""

import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')

# b is span
b = 5
ae1 = b / 2
be1 = 0.2 * b
ae2 = 0.95 * ae1
be2 = 0.9 * be1
uinf = 1
rhoinf = 1

ny = 401
num_p = ny - 1
dy = (b / 2) / (ny - 1)
y = np.linspace(0, b / 2, ny)
sh = 0.5
p = np.linspace(dy / 2, (b - dy) / 2, num_p)

xnLE = -be1 * np.sqrt(1 - (y / ae1) ** 2)
xnLE2 = sh - be2 * np.sqrt(1 - (y / ae2) ** 2)
xnTE2 = sh + be2 * np.sqrt(1 - (y / ae2) ** 2)

xnTE = be1 * np.sqrt(1 - (y / ae1) ** 2)
xnTE[xnTE > xnLE2] = sh - be2 * np.sqrt(1 - (y[xnTE > xnLE2] / ae2) ** 2)

chord = xnTE - xnLE
xn = xnLE + 0.25 * chord
xm = (xnTE[:-1] + xnTE[1:]) / 2
ym = y[:-1] + dy / 2

wmnR = np.zeros((num_p, num_p))
wmnL = np.zeros((num_p, num_p))

alpha_deg_array = [1, 2, 3, 4, 5]

CL = np.zeros(len(alpha_deg_array))

for idx, alpha_deg in enumerate(alpha_deg_array):

    # B = -uinf * np.ones(num_p)
    B = -uinf * np.deg2rad(alpha_deg) * np.ones(num_p)

    # Right Side aka Starboard
    for i in range(num_p):
        for j in range(num_p):
            temp1 = 1 / ((xm[i] - xn[j]) * (ym[i] - y[j + 1]) - (xm[i] - xn[j + 1]) * (ym[i] - y[j]))

            temp2 = ((xn[j + 1] - xn[j]) * (xm[i] - xn[j]) + (y[j + 1] - y[j]) * (ym[i] - y[j])) / \
                    np.sqrt((xm[i] - xn[j]) ** 2 + (ym[i] - y[j]) ** 2)

            temp3 = -((xn[j + 1] - xn[j]) * (xm[i] - xn[j + 1]) + (y[j + 1] - y[j]) * (ym[i] - y[j + 1])) / \
                    np.sqrt((xm[i] - xn[j + 1]) ** 2 + (ym[i] - y[j + 1]) ** 2)

            temp4 = (1 / (y[j] - ym[i])) * (1 + (xm[i] - xn[j]) / np.sqrt((xm[i] - xn[j]) ** 2 + (ym[i] - y[j]) ** 2))

            temp5 = -(1 / (y[j + 1] - ym[i])) * (1 + (xm[i] - xn[j + 1]) / np.sqrt((xm[i] - xn[j + 1]) ** 2 + (
                    ym[i] - y[j + 1]) ** 2))

            wmnR[i, j] = (1 / (4 * np.pi)) * (temp1 * (temp2 + temp3) + temp4 + temp5)

    # Left Side aka Port
    for i in range(num_p):
        for j in range(num_p):
            temp1 = 1 / ((xm[i] - xn[j]) * (ym[i] + y[j + 1]) - (xm[i] - xn[j + 1]) * (ym[i] + y[j]))

            temp2 = ((xn[j + 1] - xn[j]) * (xm[i] - xn[j]) + (-y[j + 1] + y[j]) * (ym[i] + y[j])) / \
                    np.sqrt((xm[i] - xn[j]) ** 2 + (ym[i] + y[j]) ** 2)

            temp3 = -((xn[j + 1] - xn[j]) * (xm[i] - xn[j + 1]) + (-y[j + 1] + y[j]) * (ym[i] + y[j + 1])) / \
                    np.sqrt((xm[i] - xn[j + 1]) ** 2 + (ym[i] + y[j + 1]) ** 2)

            temp4 = (1 / (-y[j] - ym[i])) * (1 + (xm[i] - xn[j]) / np.sqrt((xm[i] - xn[j]) ** 2 + (ym[i] + y[j]) ** 2))

            temp5 = -(1 / (-y[j + 1] - ym[i])) * (1 + (xm[i] - xn[j + 1]) / np.sqrt((xm[i] - xn[j + 1]) ** 2 + (ym[i] + y[j + 1]) ** 2))

            wmnL[i, j] = -(1 / (4 * np.pi)) * (temp1 * (temp2 + temp3) + temp4 + temp5)

    A = wmnR + wmnL

    # Perform Gaussian Elimination
    n = num_p

    # Columns
    for j in range(n - 1):

        # Rows
        for i in range(j, n - 1):
            bar = A[i + 1, j] / A[j, j]
            A[i + 1, :] = A[i + 1, :] - A[j, :] * bar
            B[i + 1] = B[i + 1] - bar * B[j]

    # Perform back substitution
    xsol = np.zeros(n)
    xsol[n - 1] = B[n - 1] / A[n - 1, n - 1]
    for j in range(n - 2, -1, -1):
        xsol[j] = (B[j] - np.dot(A[j, j + 1:], xsol[j + 1:])) / A[j, j]

    Gamma = xsol

    L = 0
    S = 0

    for i in range(num_p):
        L = L + 2 * rhoinf * uinf * Gamma[i] * dy
        S = S + ((chord[i] + chord[i + 1]) / 2) * dy

    S = 2 * S
    AR = b ** 2 / S
    CL[idx] = L / (S * 0.5 * rhoinf * uinf ** 2)

    # Plotting
    axs[0].plot(p, Gamma, '-', linewidth=2)

axs[0].set_title('Gamma(y) Across Half Wingspan')
axs[0].set_xlabel('y-axis: Span Location')
axs[0].set_ylabel('Gamma(y)')
axs[0].grid(True)

axs[1].plot(y, xnLE, '-', linewidth=2)
axs[1].plot(y, xnTE, '--', linewidth=2)
axs[1].plot(-y, xnLE, '-', linewidth=2)
axs[1].plot(-y, xnTE, '--', linewidth=2)
axs[1].set_aspect('equal', adjustable='box')
axs[1].axis([-1.1 * ae1, 1.1 * ae1, -2.0 * be1, 2.0 * be1])
axs[1].set_title('Wing Planform')
axs[1].set_xlabel('y-axis: Spanwise Direction')
axs[1].set_ylabel('x-axis: Chordwise Direction')
axs[1].legend(['Wing Leading Edge', 'Wing Trailing Edge'])
axs[1].grid(True)
axs[1].text(0, 0.5 * be1, f"{num_p} Panels, S= {S:.3f}, AR= {AR:.1f}", horizontalalignment='center', backgroundcolor='w')

axs[2].plot(alpha_deg_array, CL, '-', linewidth=2)
axs[2].set_title('C_L versus AOA')
axs[2].grid(True)
axs[2].set_xlabel('AOA (degrees)')
axs[2].set_ylabel('C_L')

plt.tight_layout()
plt.savefig('../fig/project4_part4a.png', bbox_inches='tight')

# Show all plots
plt.show()
