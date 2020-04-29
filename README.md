# Jomini v0.1.2
![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)
![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)]

![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)

Jomini creates military simulations by using mathematical combat models. Designed to be helpful for game developers, students, history enthusiasts and -to some extent- scientists. You can mail me at u.kaanusta@gmail.com if you want to contribute.

__To download:__ https://pypi.org/project/Jomini

Documentation will be available in the next update.

# Lanchester Models 101
This package uses the combat models developed by Frederick William Lanchester, a.k.a one of the founding fathers of Operations Research.

- Lanchester Models are deterministic, which means the model will always yield the same result for the same input parameters.
- Lanchester Models view battles as an attrition model, therefore manuevers and sudden changes during the battle can not be represented.
- You might need to do some manual fine-tuning if you are not able to get quality parameters (rho, beta, engagement_width) from a data set.
- ___Despite all of its downsides, even the primitive models developed by Lanchester himself works wonders with the right parameters.___ 

## Code Sample: Lanchester's Linear Law
- The Linear Law is based on force concentration
- Good for modelling melee battles and unaimed fire (artillery, arquebus, handcannon etc.) 
<script src="https://gist.github.com/umitkaanusta/4c5b9b53fc6db95713a05c0e3e52766b.js"></script>

![Linear Law](https://i.imgur.com/yjAUK57.png)

## Code Sample: Lanchester's Square Law
- Given equal power coefficients, the fighting power is proportional to the square of army size.
- Good for modelling aimed fire (e.g Napoleonic line-battles)
<script src="https://gist.github.com/umitkaanusta/b35cda3d2b9572c827b8ee6f7f959e6c.js"></script>

![Square Law](https://i.imgur.com/oRgkTaq.png)

