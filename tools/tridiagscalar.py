"""
UC Davis ENG 180 Fall 2010
Tridiagonal Solver (Scalar)
Daniel Wiese

This code solves scalar tridiagonal matrix equations. The main diagonals of the matrix are a_diag,
b_diag, and c_diag. The vector which augments this matrix equation is called d_column. The solution
to this system is a vector x_sol.

|   b0  c0  0   0   0   ...   0  |   |  x0  |     |  d0  |
|   a1  b1  c1  0   0   ...   0  |   |  x1  |     |  d1  |
|   0   a2  b2  c2  0   ...   0  | * |  x2  |  =  |  d2  |
|   :   :   :   :   :    :    :  |   |  :   |     |  :   |
|   0   0   0   0 an-2 bn-1 cn-1 |   | xn-1 |     | dn-1 |
|   0   0   0   0   0   an   bn  |   |  xn  |     |  dn  |
"""

import numpy as np

def tridiagscalar(a, b, c, d):
    n_size = len(b)

    # Preallocate
    bb = np.zeros(n_size)
    dd = np.zeros(n_size)
    x_sol = np.zeros(n_size)

    dd[0] = d[0]
    bb[0] = b[0]

    for i in range(1, n_size):
        bb[i] = b[i] - a[i] * c[i-1] / bb[i-1]
        dd[i] = d[i] - a[i] * dd[i-1] / bb[i-1]

    x_sol[-1] = dd[-1] / bb[-1]

    for i in range(n_size - 2, -1, -1):
        x_sol[i] = (dd[i] - x_sol[i + 1] * c[i]) / bb[i]

    return x_sol

# TODO@dpwiese - check that this is equivalent
# import numpy as np
# from scipy.linalg import solve_banded

# def tridiagscalar(a, b, c, d):

#     a = np.roll(a, -1)
#     c = np.roll(c, 1)

#     return solve_banded((1, 1), np.array([c, b, a]), d)
