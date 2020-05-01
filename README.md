# Jomini v0.1.3
![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

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

## Code Sample: Visiting models used in Jomini
- __Square Law:__ Given equal power coefficients, the fighting power is proportional to the square of army size.
    - Good for modelling aimed fire (e.g Napoleonic line-battles)
- __Logarithmic Law:__ Basically square law at a larger scale, used by Weiss to model the American Civil War 
    - Good for modelling tank combat as well


```python
from jomini.Lanchester import Battle, LinearLaw, SquareLaw, LogarithmicLaw, GeneralLaw

# Simulating a fictitious battle with each of the laws
b = Battle(red=20_000, blue=30_000, rho=0.0150, beta=0.0120)
Linear = LinearLaw(b, engagement_width=500)
Square = SquareLaw(b)
Log = LogarithmicLaw(b)
Generalized = GeneralLaw(b, engagement_width=500, p=0.450, q=0.600)

# If time is not specified, the battle goes on until one side is annihilated.
print(Linear.simulate_battle() + "\n")
print(Square.simulate_battle() + "\n")
print(Log.simulate_battle() + "\n")
print(Generalized.simulate_battle())
```
![Model-1](https://i.imgur.com/Uoz9bz4.png)
![Model-2](https://i.imgur.com/9XlE6aA.png)
