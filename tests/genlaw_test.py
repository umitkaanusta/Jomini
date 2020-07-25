# This script tests the class GeneralLaw

import unittest
from jomini.Lanchester import Battle, GeneralLaw


class TestLinearLaw(unittest.TestCase):
    def test_param(self):
        b = Battle()
        self.assertRaises(TypeError, GeneralLaw, "e")
        # engagement_width must be int
        self.assertRaises(TypeError, GeneralLaw, b, "e")
        # engagement_width must be >= 1
        self.assertRaises(ValueError, GeneralLaw, b, 500, -2, 1)
        # engagement_width can not be greater than the size of either side
        self.assertRaises(ValueError, GeneralLaw, b, 35000, 0, 0)
        # p and q must be numeric
        self.assertRaises(TypeError, GeneralLaw, b, 500, "e", "c")
        # p and q must be between 0 and 1 included
        self.assertRaises(ValueError, GeneralLaw, b, 500, 0.1, 5)

    def test_battle(self):
        g = GeneralLaw(Battle(20000, 35000, 0.01, 0.01), 5000, 0.4, 0.3)
        self.assertEqual(g.get_casualty_rates(), (160, 49))
        self.assertEqual(g.get_density(), (1.25e-05, 4.081632653061224e-06))
        self.assertEqual(g.get_casualties(time=6), (960, 294))
        self.assertEqual(g.get_casualties(), (20000, 6125))
        self.assertEqual(g.get_remaining(time=6), (19040, 34706))
        self.assertEqual(g.get_remaining(), (0, 28875))

    def test_simulation(self):
        # tests simulate_battle function
        b = Battle(20000, 35000, 0.01, 0.01)
        g = GeneralLaw(b, 5000, 0.4, 0.3)
        cas_red, cas_blue = g.get_casualties(6)
        rem_red, rem_blue = g.get_remaining(6)
        sim = g.simulate_battle(6)
        self.assertTrue("GENERAL LAW" in sim)
        self.assertTrue(str(b) in sim)
        self.assertTrue("The battle lasted 6 time units" in sim)
        self.assertTrue("Engagement width: 5000" in sim)
        self.assertTrue("p value: 0.4" in sim)
        self.assertTrue("q value: 0.3" in sim)
        self.assertTrue(f"Red casualties: {cas_red}" in sim)
        self.assertTrue(f"Blue casualties: {cas_blue}" in sim)
        self.assertTrue(f"Red remaining: {rem_red}" in sim)
        self.assertTrue(f"Blue remaining: {rem_blue}" in sim)


if __name__ == '__main__':
    unittest.main()
