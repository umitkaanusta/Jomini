# Jomini

[![CI](https://github.com/umitkaanusta/Jomini/actions/workflows/ci.yml/badge.svg)](https://github.com/umitkaanusta/Jomini/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/Jomini)](https://pypi.org/project/Jomini/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Jomini simulates battles with Lanchester-type combat models, the attrition equations
Frederick W. Lanchester introduced in 1916 and that operations researchers have refined
ever since. It's for game developers, students, history enthusiasts and researchers.

```bash
pip install jomini            # core (numpy + scipy)
pip install "jomini[plot]"    # + matplotlib for result.plot()
pip install "jomini[pandas]"  # + pandas for result.to_pandas()
```

## Quick start

```python
from jomini import Battle, SquareLaw

battle = Battle(red=20_000, blue=30_000)
model = SquareLaw(rho=0.015, beta=0.012)  # rho: red's effectiveness, beta: blue's

result = model.simulate(battle)
print(result)
# Battle lasted 71.73 time units (blue wins by annihilation)
# Red:  20,000 -> 1  (casualties 19,999)
# Blue: 30,000 -> 20,000  (casualties 10,000)

result.winner, result.duration, result.casualties
result.at(10.0)  # exact (red, blue) strengths at t = 10
result.t, result.red  # sampled trajectory as numpy arrays
result.plot()  # needs jomini[plot]
```

Real armies rarely fight to the last man. A `breakpoint` makes a side break once it
has lost that fraction of its starting strength:

```python
model.simulate(battle, breakpoint=0.3)  # both sides break at 30% losses
model.simulate(battle, breakpoint=(0.5, 0.2))  # (red, blue)
model.simulate(battle, t_max=20)  # stop the clock at t = 20
```

## Fitting a model to a real battle

Coefficients are hard to guess, so estimate them from what happened instead. Pass each
side's remaining strength at one or more times:

```python
from jomini import Battle, GeneralLaw, SquareLaw

# Cerignola (1503): 6,300 Spanish (red) lost ~500, 9,000 French (blue) lost ~4,000
cerignola = Battle(red=6_300, blue=9_000)
model = SquareLaw.fit(cerignola, t=1, red=5_800, blue=5_000)  # t=1: "one battle"
model.rho / model.beta  # ~9.3: each Spanish soldier was ~9x as effective

# With a time series you can let the data pick the law, too
GeneralLaw.fit(battle, t=[1, 2, 3, 4], red=[...], blue=[...])  # fits rho, beta, p, q
```

A single observation determines `rho` and `beta` exactly. Fitting `p` and `q` as well needs
observations at two or more times, plus enough attrition for the curves to bend. If losses
are small, the laws all look like straight lines and the exponents can't be told apart.

## 660 historical battles built in

Jomini ships with **CDB90**, the US Army Concepts Analysis Agency's database of 660 land
battles from 1600 to 1982. Each battle includes strengths, losses, duration, commanders,
outcome, coordinates and Wikipedia links, and 610 include a written account of the battle.

```python
from jomini import SquareLaw, cdb90

waterloo = cdb90.get("Waterloo")
print(waterloo)
# Waterloo (1815-06-18, Hundred Days of 1814)
# Attacker: France: 68,265 men, 25,000 casualties
# Defender: Great Britain, Prussia: 137,547 men, 22,500 casualties
# Winner: defender

model = waterloo.fit(SquareLaw)  # attacker = red, defender = blue; time in days
model.rho / model.beta  # ~2.0: each French soldier was about twice as effective

waterloo.narrative  # the full account from the original study
cdb90.search(commander="napoleon", year_from=1805)
cdb90.search(war="world war ii", winner="defender", fittable=True)
cdb90.to_pandas()  # one row per battle, needs jomini[pandas]
```

The data is the cleaned, open edition maintained by Jeffrey Arnold at
[jrnold/CDB90](https://github.com/jrnold/CDB90) (CC0, release 10.8.0), built from the
original CAA/HERO data. `scripts/build_cdb90.py` rebuilds the bundled copy from it.

## Research: which law do real battles follow?

You can't tell the laws apart from one battle, since each can be fitted exactly to a
single pair of losses. You can tell them apart across many battles. `jomini.research`
estimates the law exponent `e = p − q + 1` (2 = square, 1 = linear, 0 = logarithmic). It
picks the `e` at which a side's fighting effectiveness stops depending on how many troops
it brought, following Helmbold's criterion.

```python
from jomini import cdb90
from jomini.research import effectiveness_table, estimate_exponent

print(estimate_exponent(cdb90.battles()))
# e = 0.74 (95% CI 0.57 to 0.89, n = 646); nearest: linear law

ww2 = cdb90.search(war="world war ii")
estimate_exponent(ww2).excludes("square")  # True

effectiveness_table(exponent=1.0)  # per-battle attacker/defender effectiveness (pandas)
```

Across CDB90, the square law, the textbook default, is clearly rejected. You can also
pass `(red_start, red_end, blue_start, blue_end)` tuples from your own data.

### Separating quality from numbers

In CDB90 the side with more troops tends to have worse ones, which drags the plain estimate
down. `adjust_for` controls for CDB90's ratings of each side's advantages (leadership,
training, surprise, air superiority and so on). Effectiveness then only has to be
independent of numbers *at equal quality*:

```python
estimate_exponent(cdb90.battles(), adjust_for="quality")
# e = 0.93 (95% CI 0.81 to 1.06, n = 618); nearest: linear law
```

Once quality is accounted for, historical battles follow the linear law. The combat power
model estimates how much each factor multiplied a side's effectiveness:

```python
from jomini.research import combat_power_model

print(combat_power_model())
# Combat power model (e = 0.94, n = 618, R² = 0.37); attacker effectiveness multipliers:
#   * leadership             x 1.96  (95% CI 1.71 to 2.25)
#   * logistics              x 0.60  (95% CI 0.47 to 0.76)
#   * training               x 1.38  (95% CI 1.18 to 1.61)
#   * initiative             x 1.34  (95% CI 1.12 to 1.60)
#   * technology             x 1.61  (95% CI 1.13 to 2.28)
#   ...
combat_power_model().predict({"leadership": 1, "surprise": 2}, defender_posture="fortified")
```

The ratings were assigned by analysts who already knew the outcome, so read the multipliers
as associations, not causes. The negative logistics effect is real in the data but
unexplained.

## Win probabilities: breakpoints and Monte Carlo

Battles end when one side breaks, long before it is destroyed: in CDB90 the loser lost a
median of 15% of its strength. `jomini.stochastic` models each side's breaking point as a
random variable fitted to history. It's a log-normal fitted by maximum likelihood with
censoring: the loser's losses *are* its breaking point, while the winner's only show that
its own breaking point was higher. Armies broke at lower losses as warfare modernised (a
median of 40% for an attacker in 1700, 18% in 1900), and defenders hold slightly longer.

```python
from jomini import LinearLaw, cdb90
from jomini.stochastic import monte_carlo

waterloo = cdb90.get("Waterloo")
model = waterloo.fit(LinearLaw)
print(monte_carlo(model, waterloo.to_battle(), year=1815, seed=0))
# 2,000 runs: red wins 27.1%, blue wins 73.0%
# ...
```

Options: fixed breaking points (`breakpoints=(0.3, 0.2)`), uncertainty about relative
quality (`effectiveness_sd=...`) and a time limit (`t_max`). Cross-validated on CDB90, the
breakpoint model picks the side that actually broke first 79% of the time, given each
battle's attrition. Its probabilities are well calibrated: predictions of 10%, 30%, 50% and
90% came true 13%, 30%, 56% and 94% of the time.

## AI military historian (MCP server)

Jomini includes an [MCP](https://modelcontextprotocol.io) server, so Claude and other AI
agents can work with the battle data and models directly. Ask "How effective was the
German army against the Soviets?" or "Would Napoleon have won Waterloo with 30% more
men?" and get answers based on the data instead of guesses.

| Tool | What it does |
|---|---|
| `search_battles` | Find battles by name, war, state, commander, year or outcome |
| `get_battle` | Full record of one battle, including the narrative from the original study |
| `fit_battle` | Fit a law to a battle and report each side's relative effectiveness |
| `what_if` | Re-fight a battle with more or fewer troops, or better or worse ones, and compare with history, including each side's win probability |
| `simulate_battle` | Simulate a hypothetical battle from scratch |
| `estimate_law` | Find which law a group of battles follows (by war, state or era) |
| `compare_effectiveness` | How deadly a state's soldiers were against their enemies, battle by battle |
| `combat_power` | Which factors (leadership, surprise, posture, ...) made soldiers deadlier, overall or for one battle |

**Claude Code:**

```bash
claude mcp add jomini -- uvx --from "jomini[mcp]" jomini-mcp
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "jomini": { "command": "uvx", "args": ["--from", "jomini[mcp]", "jomini-mcp"] }
  }
}
```

Until 1.0 is on PyPI, install from GitHub instead:
`uvx --from "jomini[mcp] @ git+https://github.com/umitkaanusta/Jomini" jomini-mcp`.

## The models

Every model is a case of Bracken's generalised Lanchester law:

```
dR/dt = -beta · B^p · R^q
dB/dt = -rho  · R^p · B^q
```

| Model | p | q | Use it for |
|---|---|---|---|
| `SquareLaw` | 1 | 0 | Aimed fire (musketry, Napoleonic line battles, modern direct fire). Fighting strength grows with the *square* of numbers. |
| `LinearLaw` | 1 | 1 | Unaimed or area fire (artillery), and melee seen as many one-on-one duels. |
| `LogarithmicLaw` | 0 | 1 | Losses driven by a side's own size (exposure, disease). Used by Weiss for the American Civil War. |
| `GeneralLaw(rho, beta, p, q)` | any | any | Fitted battles that fall between the classic laws. |

The linear and logarithmic laws only approach zero and never reach it, so a side counts as
destroyed once it drops below `min_strength` (default `1.0`). If you work with
normalised strengths such as `Battle(0.6, 1.0)`, scale it down, for example `min_strength=1e-3`.

## Upgrading from 0.1.x

1.0 is a rewrite. The old versions held each side's casualty rate fixed at its value at the
start of the battle. That turned every law into a straight line and meant hand-tuned scale
factors were needed to get sensible numbers. 1.0 solves the actual differential equations,
so results will differ.

- `Battle(red, blue, rho, beta)` became `Battle(red, blue)` plus `SquareLaw(rho, beta)`, and the same for the other laws.
- `engagement_width` is gone: frontage now goes into the coefficients.
- `get_casualties()`, `get_remaining()` and `simulate_battle()` became `simulate()`, which returns a `BattleResult` (use `.casualties`, `.remaining`, `.at(t)`, `str(result)`).
- Import from `jomini`, not `jomini.Lanchester`.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check && uv run ruff format --check && uv run mypy
```

Releases are published to PyPI from GitHub releases through trusted publishing.

## License

MIT
