"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 4: High and Low Aspect Ratio Wings
Part 5: Slender Delta Wings
"""

import numpy as np
import matplotlib.pyplot as plt

uinf = 1
alphadeg = 5
alpha = np.deg2rad(alphadeg)

b = 5
chord = [8]
ny = 21
nx = 21
nt = nx - 1
y = np.linspace(0, b/2, ny)
m = b / chord[0]

xTE = np.zeros(ny)
xLE = np.zeros(ny)
dx = np.zeros(ny)

for i in range(ny):
    xTE[i] = b / (2 * m) - y[i] / m
    xLE[i] = -xTE[i]

fig, ax = plt.subplots()
ax.plot(xTE, y, '--', linewidth=2)
ax.plot(xLE, y, '-', linewidth=2)
ax.set_aspect('equal')
ax.legend(['Trailing Edge', 'Leading Edge'])
ax.grid()
plt.show()
