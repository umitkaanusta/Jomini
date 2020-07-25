# This script tests the classes Lanchester, SquareLaw and LogarithmicLaw
# Since SquareLaw and LogarithmicLaw do not have any tuning parameters, we are testing those two together

import unittest
from jomini.Lanchester import Battle, Lanchester, SquareLaw, LogarithmicLaw


class TestLanchester(unittest.TestCase):
    def test_not_instantiate(self):
        # The class Lanchester is just acting as a parent class to Lanchester laws
        self.assertRaises(RuntimeError, Lanchester)


class TestSquareLaw(unittest.TestCase):
    def test_param(self):
        self.assertRaises(TypeError, SquareLaw, "e")

    def test_battle(self):
        s = SquareLaw(Battle(20000, 35000, 0.01, 0.01))
        self.assertEqual(s.get_casualty_rates(), (350, 200))
        self.assertEqual(s.get_casualties(time=5), (1750, 1000))
        self.assertEqual(s.get_casualties(), (20000, 11600))
        self.assertEqual(s.get_remaining(time=5), (18250, 34000))
        self.assertEqual(s.get_remaining(), (0, 23400))

    def test_simulation(self):
        # tests simulate_battle function
        b = Battle(20000, 35000, 0.01, 0.01)
        s = SquareLaw(b)
        cas_red, cas_blue = s.get_casualties(5)
        rem_red, rem_blue = s.get_remaining(5)
        sim = s.simulate_battle(5)
        self.assertTrue("SQUARE LAW" in sim)
        self.assertTrue(str(b) in sim)
        self.assertTrue("The battle lasted 5 time units" in sim)
        self.assertTrue(f"Red casualties: {cas_red}" in sim)
        self.assertTrue(f"Blue casualties: {cas_blue}" in sim)
        self.assertTrue(f"Red remaining: {rem_red}" in sim)
        self.assertTrue(f"Blue remaining: {rem_blue}" in sim)


class TestLogarithmicLaw(unittest.TestCase):
    def test_param(self):
        self.assertRaises(TypeError, LogarithmicLaw, "e")

    def test_battle(self):
        L = LogarithmicLaw(Battle(80000, 100000, 0.012, 0.005))
        self.assertEqual(L.get_casualty_rates(), (400, 1200))
        self.assertEqual(L.get_casualties(time=6), (2400, 7200))
        self.assertEqual(L.get_casualties(), (33600, 100000))
        self.assertEqual(L.get_remaining(time=6), (77600, 92800))
        self.assertEqual(L.get_remaining(), (46400, 0))

    def test_simulation(self):
        # tests simulate_battle function
        b = Battle(80000, 100000, 0.012, 0.005)
        L = LogarithmicLaw(b)
        cas_red, cas_blue = L.get_casualties(6)
        rem_red, rem_blue = L.get_remaining(6)
        sim = L.simulate_battle(6)
        self.assertTrue("LOGARITHMIC LAW" in sim)
        self.assertTrue(str(b) in sim)
        self.assertTrue("The battle lasted 6 time units" in sim)
        self.assertTrue(f"Red casualties: {cas_red}" in sim)
        self.assertTrue(f"Blue casualties: {cas_blue}" in sim)
        self.assertTrue(f"Red remaining: {rem_red}" in sim)
        self.assertTrue(f"Blue remaining: {rem_blue}" in sim)


if __name__ == '__main__':
    unittest.main()
