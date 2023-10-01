"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 4: High and Low Aspect Ratio Wings
Part 1: Pressure Distribution of Rectangular Wing
"""

import numpy as np
import matplotlib.pyplot as plt

b = 12  # Span
AR = 12
cbar = b / AR
tau = 0.10
uinf = 1

ymin = -b / 2
ymax = b / 2
ny = 200
dy = (ymax - ymin) / (ny - 1)
y = np.linspace(ymin, ymax, ny)

xmin = -cbar / 2
xmax = cbar / 2
nx = 20
dx = (xmax - xmin) / (nx - 1)
x = np.linspace(xmin, xmax, nx)

ntx = nx - 1
tx = np.linspace(xmin + dx / 2, xmax - dx / 2, ntx)
dtx = dx

nty = ny - 1
ty = np.linspace(ymin + dy / 2, ymax - dx / 2, nty)
dty = dy

# Initialize arrays
chord = np.full(ny, cbar)
z = np.zeros((ny, nx))
u = np.zeros((ny, nx))

for i in range(ny):
    for j in range(nx):
        z[i, j] = -(tau / (cbar / 2)) * x[j]**2 + tau * (cbar / 2)

for i in range(ntx):
    dzdx = -4 * tau * tx[i] / cbar

for i in range(ny):
    for j in range(nx):
        for k in range(ntx):
            for m in range(nty):
                u[i, j] += (uinf / (2 * np.pi)) * (((z[m, k + 1] - z[m, k]) + (z[m + 1, k + 1] - z[m + 1, k])) / (2 * dx)) * (
                            (x[j] - tx[k]) * dx * dy) / ((x[j] - tx[k])**2 + (y[i] - ty[m])**2)**(3 / 2)

# Visualization of results
plt.figure(figsize=(10, 12), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(2, 1, 1)
plt.plot(y, chord / 2, linewidth=2)
plt.plot(y, -chord / 2, linewidth=2)
plt.gca().set_aspect('equal', adjustable='box')
plt.axis([-b / 2, b / 2, -cbar / 2 - 1, cbar / 2 + 1])
plt.xlabel('y-axis: Spanwise Direction')
plt.ylabel('x-axis: Chordwise Direction')
plt.title('Wing Planform')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(y, u[:, nx // 2], linewidth=2)
plt.title('u_T,max On Upper Surface of Wing')
plt.xlabel('y-axis: Spanwise Direction')
plt.ylabel('u_T')
plt.axis([-b / 2, b / 2, 0, 1.1 * np.max(u[:, nx // 2])])
plt.grid(True)

plt.tight_layout()
plt.savefig('../fig/project4_part1_a.png', bbox_inches='tight')

plt.figure(figsize=(10, 12), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(2, 1, 1)
plt.plot(x, z[nx // 2, :], linewidth=2)
plt.plot(x, -z[nx // 2, :], linewidth=2)
plt.axis([-1.0 * cbar / 2, 1.0 * cbar / 2, -1.1 * tau * cbar / 2, 1.1 * tau * cbar / 2])
plt.xlabel('x-axis: Chordwise Direction')
plt.ylabel('z-axis: Vertical Direction')
plt.title('Wing Profile Cross Section')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)

plt.subplot(2, 1, 2)
for i in range(nx):
    plt.plot(x, u[i, :])
plt.axis([-1.0 * cbar / 2, 1.0 * cbar / 2, -1.5 * np.max(u[:, nx // 2]), 1.5 * np.max(u[:, nx // 2])])
plt.title('u_T On Upper Surface of Wing')
plt.xlabel('x-axis: Chordwise Direction')
plt.ylabel('u_T')
plt.grid(True)

plt.tight_layout()
plt.savefig('../fig/project4_part1_b.png', bbox_inches='tight')

# Show all plots
plt.show()
