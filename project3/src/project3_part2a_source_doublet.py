"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 3: Small disturbance theory for airfoils and bodies of revolution
Part 2B: Simple Bodies in 3-D Flows
"""

import numpy as np
import matplotlib.pyplot as plt

Q = 1
D = 1
rho = 1
min_val = -4
max_val = 4
n = 10

x = np.linspace(min_val, max_val, n)
y = np.linspace(min_val, max_val, n)
z = np.linspace(min_val, max_val, n)

X, Y, Z = np.meshgrid(x, y, z)

# Source
vx_source = (Q / (4 * np.pi * rho)) * X / (X**2 + Y**2 + Z**2)**(3 / 2)
vy_source = (Q / (4 * np.pi * rho)) * Y / (X**2 + Y**2 + Z**2)**(3 / 2)
vz_source = (Q / (4 * np.pi * rho)) * Z / (X**2 + Y**2 + Z**2)**(3 / 2)

# Doublet
ux = -2 * D * X * Z / (X**2 + Y**2 + Z**2)**(5 / 2)
vr = (
    -3 * D * Z * np.sqrt(Z**2 + Y**2) / (X**2 + Y**2 + Z**2)**(5 / 2) +
    D * (Z / np.sqrt(Y**2 + Z**2)) + 1 / (X**2 + Y**2 + Z**2)**(3 / 2)
)
wT = D * (Y / np.sqrt(Z**2 + Y**2)) / (X**2 + Y**2 + Z**2)**(3 / 2)

vx_doublet = ux
vy_doublet = vr * Y / np.sqrt(X**2 + Y**2) + wT * Z / np.sqrt(X**2 + Y**2)
vz_doublet = vr * Z / np.sqrt(X**2 + Y**2) - wT * Y / np.sqrt(X**2 + Y**2) + 1

fig = plt.figure(figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.quiver(X, Y, Z, vx_source, vy_source, vz_source, length=10)
ax.set_title('Source')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')

plt.tight_layout()
# plt.savefig(f'../fig/project3_part2a_source_doublet_source.png', bbox_inches='tight')

fig = plt.figure(figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.quiver(X, Y, Z, vx_doublet, vy_doublet, vz_doublet, length=0.1)
ax.set_title('Sphere')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')

plt.tight_layout()
# plt.savefig(f'../fig/project3_part2a_source_doublet_sphere.png', bbox_inches='tight')

# Show all plots
plt.show()
