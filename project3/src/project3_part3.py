"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 3: Small disturbance theory for airfoils and bodies of revolution
Part 3: Bodies of revolution in 3D flows at angle of attack
"""

import numpy as np
import matplotlib.pyplot as plt

uinf = 1
a = 1
tau = 0.1
b = tau * a

xmin = -a
xmax = a
nx = 300
dx = (xmax - xmin) / (nx - 1)
x = np.linspace(xmin, xmax, nx)

tmin = xmin + dx / 2
tmax = xmax - dx / 2
nt = nx - 1
t = np.linspace(tmin, tmax, nt)

# Generate the profile shape
y = np.zeros(nx)
y[:150] = 0.1 + 0.1 * x[:150] / xmax
y[150:] = 0.1

dSdt = np.zeros(nt)
for j in range(nt):
    dSdt[j] = (np.pi * y[j + 1]**2 - np.pi * y[j]**2) / dx

ut_num = np.zeros(nx)
for i in range(nx):
    ut = np.zeros(nt)
    for j in range(nt):
        ut[j] = (a / (2 * np.pi)) * uinf * dSdt[j] * (x[i] - t[j]) / ((x[i] - t[j])**2 + y[i]**2)**(3 / 2)
    utsum = 0
    for k in range(nt - 1):
        temp = 0.5 * (ut[k] + ut[k + 1]) * dx
        utsum += temp
    ut_num[i] = utsum

cpx = np.zeros(nx)
for i in range(nx):
    cpx[i] = 2 * ut_num[i] / uinf

cpx[0] = cpx[1]
cpx[nx - 1] = cpx[nx - 2]

# Visualization of results
plt.figure(figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(1, 2, 1)
plt.plot(x, y, linewidth=2)
plt.plot(x, -y, linewidth=2)
plt.gca().set_aspect('equal', adjustable='box')
plt.axis([-a, a, -a, a])
plt.title('Profile Shape')
plt.grid(True)
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.subplot(1, 2, 2)
plt.plot(x, cpx)
plt.title('C_p Distribution Over Profile Surface')
plt.xlabel('Chord Position')
plt.ylabel('-C_p')
plt.legend(['Numerical Solution'])
plt.grid(True)

plt.tight_layout()
plt.savefig('../fig/project3_part3.png', bbox_inches='tight')
plt.show()
