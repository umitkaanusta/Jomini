# This script tests the class jomini.Lanchester.Battle

import unittest
from jomini.Lanchester import Battle


class BattleTest(unittest.TestCase):
    def test_instantiate(self):
        # Checks if a Battle class that obeys all the rules can be instantiated
        self.assertTrue(Battle())

    def test_errors(self):
        # Number of soldiers can not be <= 0
        self.assertRaises(ValueError, Battle, 0, 20000, 0.01, 0.01)
        # Number of soldiers must be int
        self.assertRaises(TypeError, Battle, 0.5, 20000, 0.01, 0.01)
        # Rho and beta can not be <= 0
        self.assertRaises(ValueError, Battle, 20000, 20000, -0.01, 0.01)
        # Rho and beta must be int or float
        self.assertRaises(TypeError, Battle, 20000, 20000, "hi", 0.01)

    def test_str(self):
        # Tests __str__ function
        b = Battle()
        s = str(Battle())
        self.assertTrue(f"Red side: {b.red}" in s)
        self.assertTrue(f"Blue side: {b.blue}" in s)
        self.assertTrue(f"Rho: {b.rho}" in s)
        self.assertTrue(f"Beta: {b.beta}" in s)


if __name__ == '__main__':
    unittest.main()
