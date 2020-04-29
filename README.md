# Jomini v0.1.2
![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)
![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)

![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)

Jomini creates military simulations by using mathematical combat models. Designed to be helpful for game developers, students, history enthusiasts and -to some extent- scientists. You can mail me at u.kaanusta@gmail.com if you want to contribute.

__To download:__ https://pypi.org/project/Jomini

Documentation will be available in the next update.

# Lanchester Models 101
This package uses the combat models developed by Frederick William Lanchester, a.k.a one of the founding fathers of Operations Research.

- Lanchester Models are deterministic, which means the model will always yield the same result for the same input parameters.
- Lanchester Models view battles as an attrition model, therefore manuevers and sudden changes during the battle can not be represented.
- You might need to do some manual fine-tuning if you are not able to get quality parameters (rho, beta, engagement_width) from a data set.
- ___Despite the downsides, even the primitive models developed by Lanchester himself works wonders with the right parameters.___ 

## Code Sample: Lanchester's Linear Law
- The Linear Law is based on force concentration
- Good for modelling melee battles and unaimed fire (artillery, arquebus, handcannon etc.) 

```python
from jomini.Lanchester import Battle, LinearLaw

# Re-creating the Battle of Cerignola (AD 1503)
# In the actual battle, Spanish(red) lost 500 men while the French(blue) lost 4000 men
# Parameters rho, beta, engagement_width and time are manually fine_tuned
b = Battle(red=6_300, blue=9_000, rho=0.0800, beta=0.0100)
L = LinearLaw(b, engagement_width=100)
print(L.get_casualty_rates()) # Returns casualty rates 
print(L.get_casualties(time=7))
print(L.get_remaining(time=7))
print(L.simulate_battle(time=7))
```

![Linear Law](https://i.imgur.com/yjAUK57.png)

## Code Sample: Lanchester's Square Law
- Given equal power coefficients, the fighting power is proportional to the square of army size.
- Good for modelling aimed fire (e.g Napoleonic line-battles)

```python
from jomini.Lanchester import SquareLaw

# Re-creating the Battle of Gettysburg (1863)
# --Actual battle--
# Union(red) losses: 23.000
# Confederate(blue) losses: 28.000
b = Battle(red=104_000, blue=75_000, rho=0.0180, beta=0.0100)
S = SquareLaw(b)
print(S.get_casualty_rates())
print(S.get_casualties(time=21))
print(S.get_remaining(time=21))
print(S.simulate_battle(time=21))
# What if the battle lasted until one side is annihilated?
print("\n"+ S.simulate_battle()) # time is None by default
```

![Square Law](https://i.imgur.com/oRgkTaq.png)

