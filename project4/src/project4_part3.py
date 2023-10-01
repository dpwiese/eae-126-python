"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 4: High and Low Aspect Ratio Wings
Part 3: Flow Over Swept Wings
"""

import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')

chord = np.zeros(401)

# Set the angle of the leading and trailing edge to 45 or -45 depending on desired sweep
LambdaLEdeg = 45
LambdaTEdeg = 45
b           = 5
chord[0]    = 0.2 * b
LambdaLE    = np.deg2rad(LambdaLEdeg)
LambdaTE    = np.deg2rad(LambdaTEdeg)
uinf        = 1
rhoinf      = 1
ny          = 401
num_p       = ny - 1
dy          = (b / 2) / (ny - 1)
y           = np.linspace(0, b / 2, ny)
p           = np.linspace(dy / 2, (b - dy) / 2, num_p)

xnLE    = np.zeros(ny)
xnTE    = np.zeros(ny)
xn      = np.zeros(ny)
xm      = np.zeros(num_p)
ym      = np.zeros(num_p)
wmnR    = np.zeros((num_p, num_p))
wmnL    = np.zeros((num_p, num_p))
A       = np.zeros((num_p, num_p))

for i in range(ny):
    xnLE[i] = y[i] * np.tan(LambdaLE)
    xnTE[i] = chord[0] + y[i] * np.tan(LambdaTE)
    chord[i] = xnTE[i] - xnLE[i]
    xn[i] = xnLE[i] + 0.25 * chord[i]

for i in range(num_p):
    xm[i] = ((xnLE[i] + xnLE[i + 1]) / 2) + 0.75 * ((chord[i] + chord[i + 1]) / 2)
    ym[i] = y[i] + dy / 2

alpha_deg_array = [1, 2, 3, 4, 5]

CL = np.zeros(len(alpha_deg_array))

for idx, alpha_deg in enumerate(alpha_deg_array):

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

            temp5 = -(1 / (y[j + 1] - ym[i])) * (1 + (xm[i] - xn[j + 1]) / \
                    np.sqrt((xm[i] - xn[j + 1]) ** 2 + (ym[i] - y[j + 1]) ** 2))

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
            
            temp5 = -(1 / (-y[j + 1] - ym[i])) * (1 + (xm[i] - xn[j + 1]) / \
                    np.sqrt((xm[i] - xn[j + 1]) ** 2 + (ym[i] + y[j + 1]) ** 2))

            wmnL[i, j] = -(1 / (4 * np.pi)) * (temp1 * (temp2 + temp3) + temp4 + temp5)

    for i in range(num_p):
        for j in range(num_p):
            A[i, j] = wmnR[i, j] + wmnL[i, j]

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
        xsol[j] = (B[j] - np.dot(A[j, j + 1:n], xsol[j + 1:n])) / A[j, j]

    Gamma = xsol

    L = 0
    for j in range(num_p):
        L = L + 2 * rhoinf * uinf * Gamma[j] * dy

    S = (chord[0] + chord[ny - 1]) * y[ny - 1]
    CL[idx] = L / (S * 0.5 * rhoinf * uinf ** 2)

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
axs[1].set_title('Wing Planform')
axs[1].set_xlabel('y-axis: Spanwise Direction')
axs[1].set_ylabel('x-axis: Chordwise Direction')
axs[1].legend(['Wing Leading Edge', 'Wing Trailing Edge'])
axs[1].grid(True)
axs[1].text(0, 0.75 * max(abs(xnTE)), f'{num_p} Panels', horizontalalignment='center', backgroundcolor='w')
axs[1].invert_xaxis()

axs[2].plot(alpha_deg_array, CL, '-', linewidth=2)
axs[2].set_title('C_L versus AOA')
axs[2].grid(True)
axs[2].set_xlabel('AOA (degrees)')
axs[2].set_ylabel('C_L')

plt.tight_layout()
plt.savefig('../fig/project4_part3.png', bbox_inches='tight')

# Show all plots
plt.show()
