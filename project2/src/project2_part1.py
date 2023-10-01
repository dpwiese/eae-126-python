"""
EAE-126 Computational Aerodynamics (Spring 2011)
Daniel Wiese

Project 2: Joukowski transformation and airfoils
Part 1: Plot four Joukowski transformations
"""

import numpy as np
import matplotlib.pyplot as plt

ntheta = 50
theta = np.linspace(0, 2*np.pi, ntheta)

configs = [
    {'a': 1,    'epsilon': 0,       'mu': 0,    'b': 1,     'name': 'flat plate'},
    {'a': 1,    'epsilon': 0,       'mu': 0.1,  'b': 0.99,  'name': 'circular arc'},
    {'a': 1,    'epsilon': -0.1,    'mu': 0.0,  'b': 0.9,   'name': 'symmetric airfoil'},
    {'a': 1,    'epsilon': -0.1,    'mu': 0.1,  'b': 0.89,  'name': 'cambered airfoil'},
    {'a': 1,    'epsilon': 0,       'mu': 0.0,  'b': 0.9,   'name': 'ellipse'}
]

fig = plt.figure(figsize=(8, 12), dpi=100, facecolor='w', edgecolor='k')

for idx, cfg in enumerate(configs):

    # Parameters for each transformation
    a       = cfg['a']
    epsilon = cfg['epsilon']
    mu      = cfg['mu']
    b       = cfg['b']

    x = epsilon + a * np.cos(theta)
    y = mu + a * np.sin(theta)

    X = x * (1 + (b**2) / (x**2 + y**2))
    Y = y * (1 - (b**2) / (x**2 + y**2))

    TITLE_FONT_SIZE = 10
    TICK_FONT_SIZE = 8

    config_text_string = f'a={a:.2f}, epsilon={epsilon:.2f}, mu={mu:.2f}, b={b:.2f}'

    # Circle Before Transformation
    ax1 = plt.subplot(5, 2, 2 * (idx + 1)-1)
    plt.plot(x, y, '-k', linewidth=1)
    plt.fill(x, y, 'r')
    plt.title('Circle Before Transformation', fontsize=TITLE_FONT_SIZE)
    plt.axis([-2, 2, -2, 2])
    plt.grid(True, linestyle='--', alpha=0.7)
    ax1.set_aspect('equal', adjustable='box')
    ax1.tick_params(labelsize=TICK_FONT_SIZE)

    # Circle After Transformation: Airfoil
    ax2 = plt.subplot(5, 2, 2 * (idx + 1))
    plt.plot(X, Y, '-k', linewidth=1)
    plt.fill(X, Y, 'r')
    plt.title('Circle After Transformation: Airfoil', fontsize=TITLE_FONT_SIZE)
    plt.text(0, 1.5, config_text_string, ha='center', backgroundcolor='w')
    plt.axis([-3, 3, -1, 1])
    plt.grid(True, linestyle='--', alpha=0.7)
    ax2.set_aspect('equal', adjustable='box')
    ax2.tick_params(labelsize=TICK_FONT_SIZE)

fig.tight_layout()
fig.savefig(f'../fig/project2_part1.png', bbox_inches='tight')

# Show all plots
plt.show()
