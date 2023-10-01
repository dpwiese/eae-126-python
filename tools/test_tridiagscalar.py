"""
Test

python3 -m unittest
"""

import unittest
import numpy as np
from tools.tridiagscalar import tridiagscalar

class TestTriDiagScalar(unittest.TestCase):
    """TBD"""

    # pylint: disable-next=R0201
    def test_tridiagscalar_1(self):
        """TBD"""

        a_diag = np.zeros((2,))
        b_diag = np.ones((2,))
        c_diag = np.zeros((2,))
        d_diag = np.ones((2,))

        x_sol = np.ones((2,))

        np.testing.assert_array_equal(tridiagscalar(a_diag, b_diag, c_diag, d_diag), x_sol)

    # pylint: disable-next=R0201
    def test_tridiagscalar_2(self):
        """TBD"""

        a_diag = np.zeros((5,))
        b_diag = np.ones((5,))
        c_diag = np.zeros((5,))
        d_diag = np.ones((5,))

        x_sol = np.ones((5,))

        np.testing.assert_array_equal(tridiagscalar(a_diag, b_diag, c_diag, d_diag), x_sol)

    # pylint: disable-next=R0201
    def test_tridiagscalar_3(self):
        """TBD"""

        a_diag = np.array([1, 2, 3, 4, 5])
        b_diag = np.array([7, 6, 5, 4, 3])
        c_diag = np.array([3, 2, 4, 2, 3])
        d_diag = np.array([1, 1, 1, 1, 1])

        x_sol = np.array([0.095041, 0.111570, 0.070248, 0.078512, 0.202479])

        np.testing.assert_allclose(tridiagscalar(a_diag, b_diag, c_diag, d_diag), x_sol, rtol=1e-05, atol=1e-08, equal_nan=False)

if __name__ == '__main__':
    unittest.main()
