# This script tests the class LinearLaw

import unittest
from jomini.Lanchester import Battle, LinearLaw


class TestLinearLaw(unittest.TestCase):
    def test_param(self):
        b = Battle()
        self.assertRaises(TypeError, LinearLaw, "e")
        # engagement_width must be int
        self.assertRaises(TypeError, LinearLaw, b, "e")
        # engagement_width must be >= 1
        self.assertRaises(ValueError, LinearLaw, b, 0)
        # engagement_width can not be greater than the size of either side
        self.assertRaises(ValueError, LinearLaw, b, 35000)

    def test_battle(self):
        Li = LinearLaw(Battle(20000, 35000, 0.01, 0.01), 500)
        self.assertEqual(Li.get_casualty_rates(), (875, 285))
        self.assertEqual(Li.get_density(), (1.25e-06, 4.0816326530612243e-07))
        self.assertEqual(Li.get_casualties(time=5), (4375, 1425))
        self.assertEqual(Li.get_casualties(), (20000, 6555))
        self.assertEqual(Li.get_remaining(time=5), (15625, 33575))
        self.assertEqual(Li.get_remaining(), (0, 28445))

    def test_simulation(self):
        # tests simulate_battle function
        b = Battle(20000, 35000, 0.01, 0.01)
        Li = LinearLaw(b, 500)
        cas_red, cas_blue = Li.get_casualties(5)
        rem_red, rem_blue = Li.get_remaining(5)
        sim = Li.simulate_battle(5)
        self.assertTrue("LINEAR LAW" in sim)
        self.assertTrue(str(b) in sim)
        self.assertTrue("The battle lasted 5 time units" in sim)
        self.assertTrue("Engagement width: 500" in sim)
        self.assertTrue(f"Red casualties: {cas_red}" in sim)
        self.assertTrue(f"Blue casualties: {cas_blue}" in sim)
        self.assertTrue(f"Red remaining: {rem_red}" in sim)
        self.assertTrue(f"Blue remaining: {rem_blue}" in sim)


if __name__ == '__main__':
    unittest.main()
