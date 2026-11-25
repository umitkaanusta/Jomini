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

## Modern warfare: salvos, mixed forces and learning laws from data

**Salvo exchanges** (Hughes 1995) model fights decided by volleys of missiles or drones
against interceptors, where defences can absorb a whole salvo and striking first decides
the outcome:

```python
from jomini.modern import SalvoModel, SalvoSide

ship = SalvoSide(offense=4, defense=2, staying_power=1)
SalvoModel(ship, ship).simulate(3, 2).winner  # "red"
SalvoModel(ship, ship).simulate(3, 2, first="blue").winner  # "blue": a surprise strike flips it
```

**Mixed forces** fight with a kill-rate matrix between unit types and fire spread over
whatever targets remain:

```python
from jomini.modern import MixedForceLaw

law = MixedForceLaw.from_rates(
    red_kills={"drone": {"tank": 0.02, "infantry": 0.1}, "tank": {"tank": 0.05, "infantry": 0.2}},
    blue_kills={"tank": {"tank": 0.05}, "infantry": {"tank": 0.002, "drone": 0.01}},
)
law.simulate({"drone": 200, "tank": 50}, {"tank": 80, "infantry": 1000}, t_max=50, breakpoint=0.4)
```

**Learning the law from loss time series** (daily casualties, or equipment losses from
trackers like WarSpotting). `fit_power_law` fits each side's exponents with standard
errors, so asymmetric laws show up. `sindy` discovers the equations by sparse regression,
as in recent Russia–Ukraine studies:

```python
from jomini.discovery import fit_power_law, sindy

fit_power_law([(t, red, blue)])
sindy([(t1, red1, blue1), (t2, red2, blue2), ...])
```

Pass several engagements with different starting strengths where you can. Along a single
trajectory, Lanchester laws conserve a quantity (`ρR² − βB²` for the square law) that makes
the law impossible to identify. `sindy` then returns spurious terms, and `fit_power_law`
raises an error instead of guessing.

## More data: Kursk day by day, the Civil War, and Ukraine

`jomini.data` adds three sources beyond CDB90:

```python
from jomini import data

kursk = data.kursk()  # daily, both sides, 5-18 July 1943 (public domain)
kursk.series["soviet"]["tanks_lost"]  # 414 on 12 July, at Prokhorovka
t, german, soviet = kursk.trajectory("combat_power")

data.livermore()  # 63 Civil War battles (Livermore 1900), as HistoricalBattle

gs = data.ukraine_general_staff()  # daily Russian losses claimed by Ukraine (fetched, cached)
gs.daily("tank"), gs.cumulative["personnel"]
data.ukraine_oryx("Russia"), data.ukraine_oryx("Ukraine")  # visually confirmed, both sides
```

| Source | Coverage | Licence | How |
|---|---|---|---|
| Kursk Data Base via Lucas & Turkes (2004) | 14 days × both sides: manpower, casualties, tanks, APCs, artillery | US Government work | bundled |
| Livermore (1900) via jrnold/acw_battle_data | 63 major Civil War battles | public domain; compilation ODC-BY ([cite](https://doi.org/10.6084/m9.figshare.1515995)) | bundled |
| Ukrainian General Staff via PetroIvaniuk | Russian losses, daily since 25 Feb 2022, 14+ categories | MIT | fetched at runtime |
| Oryx via scarnecchia/oryx_data | Visually confirmed equipment losses until Aug 2025 (Ukraine side from Mar 2025) | CC BY 4.0 | fetched at runtime |
| Oryx entries via leedrake5/Russia-Ukraine | Confirmed losses by event date, both sides daily, Feb 2022 to Aug 2023 | MIT | fetched at runtime |

The General Staff series is one side's unverified claims about the other side. Oryx counts
only losses confirmed by photo or video, so it is a lower bound.

Two lessons came out of the new data:

- **Kursk can't pin down a law.** Fitting losses to strengths
  (`fit_power_law_from_losses`) gives exponents around 6, with huge standard errors. Lucas &
  Turkes and Clemens got the same result. Over 14 days German manpower only moved between
  296k and 307k, too little variation to identify the exponents.
- **Battle selection matters as much as numbers.** For the 18 Civil War battles that CDB90
  and Livermore share, the exponent agrees (0.90 vs 0.92). Across each source's full set it
  doesn't (CDB90 1.60, Livermore 0.24). An exponent estimated from one war's battles is only
  as good as the choice of battles.

## Effectiveness that changes over time

Constant coefficients don't describe a long war. `jomini.dynamics` treats relative
effectiveness as a hidden random walk and recovers its path, with uncertainty, from loss
counts. It uses a Kalman filter and RTS smoother, counting noise plus estimated
overdispersion, and maximum-likelihood step sizes. Under the linear law strengths cancel,
so per-period losses are enough.

```python
from jomini import data
from jomini.dynamics import effectiveness_over_time, intensity_over_time

ru = data.ukraine_oryx_events("Russia").resample(28)  # both sides, Feb 2022 to Aug 2023
ua = data.ukraine_oryx_events("Ukraine").resample(28)
path = effectiveness_over_time(ru.daily("Tanks")[1:], ua.daily("Tanks")[1:])
path.ratio(), path.interval(0.9)  # Russia's relative effectiveness, 28-day steps
```

On visually confirmed equipment losses, Russia's relative effectiveness roughly doubled
between 2022 (about 4 pieces of equipment lost per Ukrainian loss) and mid-2023 (about 2).
For tanks the exchange flipped during Ukraine's 2023 counteroffensive, when Ukraine was
attacking prepared defences. At Kursk, German per-soldier effectiveness peaked on 12 July,
the day of Prokhorovka. Oryx counts only *documented* losses, and documentation rates
differ by side and over time, so read these as trends in confirmed losses.

`intensity_over_time` does the same for a single series, such as daily personnel losses
claimed by the General Staff.

## Conflict events and escalation (UCDP GED)

`data.ucdp_ged()` loads the UCDP Georeferenced Event Dataset 26.1: about 418,000 events of
organised violence worldwide from 1989 to 2025, with deaths per side (CC BY 4.0, fetched
once and cached). `jomini.escalation` fits self-exciting (Hawkes) models to daily counts.
They measure how much violence feeds on itself:

```python
from datetime import date

import numpy as np

from jomini import data
from jomini.escalation import fit_hawkes

ged = data.ucdp_ged()
ru = ged.select(conflict="Russia - Ukraine", start=date(2022, 2, 24))
events = ru.daily("events").daily("events")
print(fit_hawkes(events))  # branching ratio ~0.69, half-life ~1 day
```

| Conflict | Branching ratio (spline background) | Half-life |
|---|---|---|
| Russia–Ukraine, 2022–25 | 0.69 | 1.0 days |
| Syria, 2012–16 | 0.56 | 1.3 days |
| Sudan, 2023–25 | 0.23 | 0.1 days |

A war whose intensity drifts (offensives, mobilisation, lulls) looks like contagion to a
model with a constant background rate. In simulations with a drifting background and *no*
excitation, the constant-background fit reported a branching ratio of 0.98 (p < 1e-200).
The default spline background reported 0.00. The spline fit is conservative: it
understates a true branching ratio of 0.5 by about 10–20%.

In UCDP's per-side battle-death estimates for Russia–Ukraine, Russia's relative
effectiveness (`effectiveness_trend(source="ucdp")`) fell from about 0.7 in 2022 to about
0.12 in late 2025, roughly 8 Russian deaths per Ukrainian death. UCDP codes events from
reports that may echo each side's claims, so this is not independent of official figures. Reporting is also asymmetric: part of the falling ratio may reflect Ukrainian losses
being reported less rather than changing effectiveness. UCDP's provisional 2026 data shows
the extreme case, with Ukrainian deaths falling to single digits per month from February
2026 while Russian deaths stay in the thousands.

## The 2026 Iran war

`data.ucdp_candidate()` loads UCDP's provisional events for January–September 2026, which
include the war that began with US and Israeli strikes on Iran on 28 February 2026 and
ended with the 9 April ceasefire. The MCP conflict tools take `dataset="candidate"`.

| | Twelve-Day War (June 2025, GED) | 2026 war (Candidate GED) |
|---|---|---|
| Events (Iran vs Israel/US dyad) | 43 | 459 (410 before the ceasefire) |
| Deaths, best estimate | 1,218 | ~3,860 |
| Iranian military / Israeli & US / civilian | 433 / 1 / 465 | 1,274 / 11 / 1,774 |
| Pattern | decays within days | opening shock (103 events on day 1), event rate halving about every 14 days |

It was a stand-off war of strikes and missile salvos, not ground combat, so Lanchester
exchange ratios don't apply. In both wars UCDP records Israeli deaths mostly as civilian,
far below Israeli reports. Per-wave salvo data for Iranian attacks (an OSINT compilation)
is too sparse to fit a saturation model: only the April 2024 attack has interception
counts.

## Drone swarms against layered air defence

`jomini.airdefense` models a raid flying through bands of defence. It is a one-dimensional
take on Randow et al.'s PDE extension of Lanchester (2026): the raid advects inbound and
diffuses. Each layer's shots follow a saturating (Holling type II) curve and draw on a
finite magazine:

```python
from jomini.airdefense import generic_defense

defense = generic_defense()            # illustrative: interceptors, mobile guns, missiles
print(defense.raid(100, attacker_unit_cost=50_000))
sizes, leak_share, elasticity = defense.leakage_curve([25, 50, 100, 200, 400])
camp = defense.campaign([150] * 10, restock={"missiles": 3})   # magazine depletion
```

**Proposition.** If each attacker's chance of being shot down can only fall as more
attackers are present (capacity limits, magazines), the leak *fraction* can't fall as raids
grow: leakers rise at least in proportion (elasticity ≥ 1). The module docstring has the
proof sketch, and a property test checks it on the spatial model.

**Ukraine's data violate it.** `data.ukraine_air_raids()` fetches the Ukrainian Air
Force's nightly reports (Ivaniuk, Kaggle, CC BY-NC-SA 4.0: fetched, never bundled).
`saturation_test` compares nights within the same month:

| Nights (≥5 launched) | Log-odds stopped per log raid size (95% CI) | Leak elasticity |
|---|---|---|
| Shahed/Geran, Sep 2022 – Aug 2024 | −0.19 (−0.41 to +0.03) | 0.84 |
| Shahed/Geran, Sep 2024 – Sep 2025 | **+0.46** (+0.36 to +0.57) | 0.59 |
| Shahed/Geran, Oct 2025 – Nov 2026 | **+0.51** (+0.43 to +0.59) | 0.56 |
| Cruise missiles, 2022–2026 | **+0.35** (+0.14 to +0.56) | 0.69 |
| Ballistic missiles, 2022–2026 | +0.59 (−0.06 to +1.25) | 0.86 |

Since autumn 2024, doubling a drone raid within the same month has multiplied the number
getting through by only about 1.5, not 2. No fixed-capacity defence produces that. The
change coincides with mass Russian decoy launches and Ukraine's build-up of mobile fire
groups and interceptor drones. Candidate explanations, which these data can't separate:

- defences surged on warning of big raids (`Layer.surge`: the model needs a surge of
  about 1 to match);
- decoys making up more of bigger raids;
- routing;
- the defender's own reporting.

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
| `salvo_exchange` | Missile, drone-swarm or naval exchanges with Hughes' salvo model |
| `list_datasets` | Every dataset's coverage, licence, access and caveats |
| `cite` | BibTeX for datasets, methods and Jomini itself |
| `get_loss_series` | Loss time series: Kursk daily, Ukrainian General Staff claims, Oryx |
| `effectiveness_trend` | How relative effectiveness changed over time (Kalman-smoothed, with bands) |
| `loss_intensity` | Smoothed loss rate behind a noisy series |
| `fit_law_to_series` | Fit each side's attrition law to per-period strengths and losses |
| `breakpoint_distribution` | How much a side loses before it breaks, for a given year |
| `conflict_events` | Monthly events and deaths for any conflict or country, 1989–2025 (UCDP) |
| `escalation_model` | Self-exciting (Hawkes) fit: how strongly violence feeds on itself |

Every tool result carries a `provenance` block listing its sources and methods with
`cite_keys`. Pass those keys to `cite` (or `jomini.catalog.bibtex`) to get BibTeX. Each
dataset's caveats (e.g. "one side's claims", "documented losses only") are part of the
catalogue, and agents are instructed to pass them on.

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

## Research workflow: provenance and citation

```python
from jomini import catalog

catalog.DATASETS["kursk"].caveats  # read before drawing conclusions
print(catalog.bibtex(["cdb90", "law_estimation", "breakpoints"]))  # data + methods + Jomini
```

`jomini.catalog` records each dataset's coverage, unit, access (bundled or fetched),
licence, caveats and references, plus the references behind each method. DOIs and page
ranges appear only where they were checked; verify entries against the publisher before
formal citation.

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
