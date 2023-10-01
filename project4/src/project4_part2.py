"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 4: High and Low Aspect Ratio Wings
Part 2: Flow Over Elliptic Wing: Loading Distribution
"""

import numpy as np
import matplotlib.pyplot as plt

a = 18
AR = 12
b = (4 * a) / (np.pi * AR)

uinf = 1
alphadeg = 5
alpha = np.deg2rad(alphadeg)

ymin = -a
ymax = a
n = 100
dy = (ymax - ymin) / (n - 1)
tmin = ymin + dy / 2
tmax = ymax - dy / 2
nt = n - 1
y = np.linspace(ymin, ymax, n)
t = np.linspace(tmin, tmax, nt)

xplan = b * np.sqrt(1 - (y / a)**2)
chord = 2 * xplan

A = np.zeros((n, n))

for i in range(0, n - 1):
    for j in [0]:
        A[i, j] = 1
        A[i, j] = 1

for i in range(1, n):
    for j in [n - 1]:
        A[i, j] = 1
        A[i, j] = 1

for i in range(1, n - 1):
    for j in range(1, n - 1):
        A[i, j] = (1 / (4 * uinf * np.pi)) * ((1 / (y[i] - t[j - 1])) - (1 / (y[i] - t[j])))

for i in range(1, n - 1):
    A[i, i] = 1 / (np.pi * uinf * chord[i]) + (1 / (4 * uinf * np.pi)) * (
                (1 / (y[i] - t[i - 1])) - (1 / (y[i] - t[i])))

B = np.zeros(n)
B[1:n - 1] = alpha

# Perform Gaussian Elimination
for j in range(n - 1):  # Column
    for i in range(j, n - 1):  # Row
        bar = A[i + 1, j] / A[j, j]
        A[i + 1, :] = A[i + 1, :] - A[j, :] * bar
        B[i + 1] = B[i + 1] - bar * B[j]

# Perform back substitution
xsol = np.zeros(n)
xsol[n - 1] = B[n - 1] / A[n - 1, n - 1]
for j in range(n - 2, -1, -1):
    xsol[j] = (B[j] - np.dot(A[j, j + 1:n], xsol[j + 1:n])) / A[j, j]

Gamma = xsol

# Analytical solution
GammaS = (8 * b * np.pi * uinf * a * alpha) / (2 * b * np.pi + 4 * a)

plt.figure(figsize=(10, 12), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(2, 1, 1)
plt.plot(y, xplan, linewidth=2)
plt.plot(y, -xplan, linewidth=2)
plt.grid(True)
plt.axis([ymin, ymax, -2 * b, 2 * b])
plt.title('Wing Planform')
plt.xlabel('y-axis: Spanwise Direction')
plt.ylabel('x-axis: Chordwise Direction')
plt.gca().set_aspect('equal', adjustable='box')

plt.subplot(2, 1, 2)
plt.plot(y, Gamma, linewidth=2)
plt.grid(True)
plt.axis([ymin, ymax, 0, 1.1 * GammaS])
plt.title('Spanwise Circulation Distribution')
plt.xlabel('y-axis: Spanwise Direction')
plt.ylabel('Circulation Gamma(y)')
plt.text(0, 0.2, f'Gamma_analytical= {GammaS:.2f}', horizontalalignment='center', backgroundcolor='w')

plt.tight_layout()
plt.savefig('../fig/project4_part2.png', bbox_inches='tight')

# Show all plots
plt.show()
