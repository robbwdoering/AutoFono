import unittest
from simpleNN import *
import numpy as np

class TestSimpleNN(unittest.TestCase):
    """
    Our basic test class
    """
    X = np.arange(15).reshape((5,3))
    Y = np.array([1,1,1,0,0])
    Theta1 = 1 * np.ones((4,5)) #4x5
    Theta2 = 2 * np.ones((6,5)) #6x5
    Theta3 = 3 * np.ones((6,1)) #6x1


    def test_sigGradient(self):
        self.assertEqual(sigGradient(0), 0.25)
        self.assertTrue(np.array_equal(sigGradient(np.array([0,0,0,0])),
            np.array([0.25,0.25,0.25,0.25])))
    def test_


if __name__ == '__main__':
    unittest.main()
