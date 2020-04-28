from math import ceil


class Battle:
    def __init__(self, red=20_000, blue=20_000, rho=0.0100, beta=0.0100):
        """
        We assume that battles are two sided
        :param int red: Number of soldiers in the side red
        :param int blue: Number of soldiers in the side blue
        :param float rho: Power coefficient of the red side (default is 0.0100)
        :param float beta: Power coefficient of the blue side (default is 0.0100)
        """
        if not isinstance(red, int) or not isinstance(blue, int):
            raise TypeError("Number of soldiers must be integer")
        if red < 0 or blue < 0:
            raise ValueError("Number of soldiers can not be negative")
        if not isinstance(rho, (int, float)) or not isinstance(beta, (int, float)):
            raise TypeError("The power coefficient must be int or float")
        if rho < 0 or beta < 0:
            raise ValueError("Power coefficient values can not be negative")

        self.red = red
        self.blue = blue
        self.rho = rho
        self.beta = beta

    @property
    def total(self):
        return self.red + self.blue

    @property
    def force_red(self):
        return self.red * self.rho

    @property
    def force_blue(self):
        return self.blue * self.beta

    def __repr__(self):
        return f"Battle({self.red}, {self.rho}, {self.blue}, {self.beta})"

    def __str__(self):
        return f"Battle\n" \
               f"Red side: {self.red} Soldiers | Rho: {self.rho} | Total force: {self.force_red}\n" \
               f"Blue side: {self.blue} Soldiers | Beta: {self.beta} | Total force: {self.force_blue}"


class Lanchester:
    """Shows the general properties & methods of the Linear and Square Law"""
    def __init__(self):
        raise RuntimeError("Lanchester class should not be instantiated")

    def _check_time(self, _time):
        # You really don't need this method in your calculations
        if _time is None:  # If time is none, get the maximum time for the battle to end
            rate_red, rate_blue = self.get_casualty_rates()
            _time = int(ceil(self.battle.red / rate_red)) if rate_red > rate_blue \
                else int(ceil(self.battle.blue / rate_blue))
        if not isinstance(_time, int):
            raise TypeError("time should be int")
        if _time <= 0:
            raise ValueError("time can not be zero or negative")
        return _time

    def get_casualties(self, time=None):
        """ For a battle that takes time t, returns a tuple for the casualties of both sides.
        If time is None, the battle will continue until one side is annihilated. """
        time = self._check_time(time)
        rate_red, rate_blue = self.get_casualty_rates()
        casualties_red = rate_red * time if rate_red * time < self.battle.red else self.battle.red
        casualties_blue = rate_blue * time if rate_blue * time < self.battle.blue else self.battle.blue
        return int(casualties_red), int(casualties_blue)

    def get_remaining(self, time=None):
        """ After the casualties are calculated, returns armies with their new sizes """
        time = self._check_time(time)
        casualties_red, casualties_blue = self.get_casualties(time)
        remaining_red = self.battle.red - casualties_red
        remaining_blue = self.battle.blue - casualties_blue
        return int(remaining_red), int(remaining_blue)

    def _simulate_battle(self, which_model, time=None):
        """ Returns a string showing the simulated battle. If time is None, the battle will continue
         until one side is annihilated. """
        time = self._check_time(time)
        casualties_red, casualties_blue = self.get_casualties(time)
        remaining_red, remaining_blue = self.get_remaining(time)
        return_str = f"----- {which_model} BATTLE RESULTS -----\n" + str(self.battle) + "\n" \
            + f"The battle lasted {time} time units.\n" \
            + f"Red casualties: {casualties_red} | " + f"Blue casualties: {casualties_blue}\n" \
            + f"Red remaining: {remaining_red} | " + f"Blue Remaining: {remaining_blue}\n" + "-" * 60
        return return_str


class LinearLaw(Lanchester):
    def __init__(self, battle, engagement_width):
        """
        Implementing Lanchester's Linear Law
        :param battle: Battle object
        :param int engagement_width: How many units from each army engage at unit time?
        """
        if not isinstance(battle, Battle):
            raise TypeError("battle should be an object of the Battle class")
        if not isinstance(engagement_width, int):
            raise TypeError("engagement_width must be an integer")
        if engagement_width < 1:
            raise ValueError("engagement_width can not be lesser than 1")
        if engagement_width > min(battle.red, battle.blue):
            raise ValueError("engagement_width can not be greater than the size of either side")

        self.battle = battle
        self.engagement_width = engagement_width

    def get_casualty_rates(self):
        """ Returns a tuple representing casualty rates for unit time """
        # Actually Lanchester himself did not even write a formula for the linear law
        density_red = (self.engagement_width / self.battle.red ** 2)
        density_blue = (self.engagement_width / self.battle.blue ** 2)
        rate_red = density_red * self.battle.red * self.battle.blue * self.battle.beta * 100
        rate_blue = density_blue * self.battle.red * self.battle.blue * self.battle.rho * 100
        return int(rate_red), int(rate_blue)

    def simulate_battle(self, time=None):
        return self._simulate_battle(time=time, which_model="LINEAR LAW")


class SquareLaw(Lanchester):
    def __init__(self, battle):
        """
        Implementing Lanchester's Square Law
        :param battle: Battle object
        """
        if not isinstance(battle, Battle):
            raise TypeError("battle should be an object of the Battle class")

        self.battle = battle

    def get_casualty_rates(self):
        """ Returns a tuple representing casualty rates for unit time """
        rate_red = self.battle.red * self.battle.beta
        rate_blue = self.battle.blue * self.battle.rho
        return int(rate_red), int(rate_blue)

    def simulate_battle(self, time=None):
        return self._simulate_battle(time=time, which_model="SQUARE LAW")
