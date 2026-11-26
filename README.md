# Jomini

[![CI](https://github.com/umitkaanusta/Jomini/actions/workflows/ci.yml/badge.svg)](https://github.com/umitkaanusta/Jomini/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Jomini simulates drone raids against layered air defence and fits the model to what
happens over Ukraine each night. It also measures how violence feeds on itself in a war,
tracks how each side's effectiveness changes over time, and holds data on 660 historical
battles. You can use it from Python, or give it to Claude as a set of tools.

The package on PyPI is out of date. I'll update it soon. Until then, clone the repository,
or run Jomini as an MCP server straight from GitHub (see [below](#use-it-from-claude)).

## Demo: a Shahed raid on Ukraine

### 1. Run a raid

```python
from jomini.airdefense import ukraine_defense

defense = ukraine_defense()  # fitted to Ukraine's nightly reports, Sep 2024 to Nov 2026
print(defense.raid(125))  # a typical night
# 125 launched, 19.3 leaked (15.4%); interceptors 82 kills/130 shots,
# mobile guns 14 kills/44 shots, missiles 10 kills/11 shots

defense.raid(30).leak_fraction  # 0.24
defense.raid(500).leak_fraction  # 0.088
```

Send four times as many drones and a smaller share gets through.

You can build your own defence and fight it night after night:

```python
from jomini.airdefense import AirDefense, Layer

drones = Layer("interceptor drones", 300, 150, pk=0.6, fire_rate=60, engage_rate=2, magazine=300)
guns = Layer("mobile guns", 150, 40, pk=0.3, fire_rate=100, engage_rate=2)
patriot = Layer("Patriot", 40, 0, pk=0.85, fire_rate=20, engage_rate=2, magazine=60)
defense = AirDefense((drones, guns, patriot))

defense.raid(300, launch_window=2.0)  # 300 drones launched over two hours: 58% get through

camp = defense.campaign([300] * 10, restock={"Patriot": 4, "interceptor drones": 30})
camp.empty_after("Patriot")  # 3: the Patriots run out of missiles after the fourth night
camp.leak_fraction  # 69% get through on the first night, 83% from the seventh
```

### 2. The equations

The drones fly in from 300 km out, through three rings of defence, toward the target:

```
target <-- missiles --|-- mobile guns --|-- interceptors --|<-- raid
  0 km              40 km            150 km             300 km
```

Let $u(x, t)$ be the number of drones per kilometre at distance $x$ from the target. They
fly in at speed $v$ and spread out as they go:

```math
\frac{\partial u}{\partial t} = v\,\frac{\partial u}{\partial x} + D\,\frac{\partial^2 u}{\partial x^2} - \sum_k h_k(t)\,\mathbf{1}[x \in \text{ring } k]\,u
```

Ring $k$ sees $N_k$ drones and fires $s_k$ shots an hour. Each drone in the ring is killed
at rate $h_k$:

```math
N_k(t) = \int_{\text{ring } k} u\,dx, \qquad
s_k = \frac{F_k\,e_k\,N_k}{F_k + e_k\,N_k}, \qquad
h_k = \frac{p_k\,s_k}{N_k}
```

Every shot comes out of the ring's stock, and the ring falls silent when the stock is gone:

```math
\frac{dM_k}{dt} = -s_k
```

A ring can be reinforced when a big raid is coming. For a raid of $R$ drones, its firing
rates are multiplied by

```math
F_k,\ e_k \;\to\; \left(\frac{R}{100}\right)^{\sigma} F_k,\ \left(\frac{R}{100}\right)^{\sigma} e_k
```

Drones that reach $x = 0$ get through.

| Symbol | Meaning |
|---|---|
| $v$ | Drone speed (185 km/h) |
| $D$ | How fast the raid spreads out (50 km²/h) |
| $p_k$ | Chance that one shot kills |
| $e_k$ | How often a lone drone is shot at (shots per hour) |
| $F_k$ | The most shots the ring can fire in an hour |
| $M_k$ | Shots left in stock |
| $\sigma$ | Surge: how much the defence is reinforced for big raids |

This follows Randow et al. (2026), cut down to one dimension.

### 3. What it means

The shots formula has two regimes.

- **Few drones.** The ring has time for all of them: $s_k \approx e_k N_k$, so every drone
  runs the same risk, $h_k = p_k e_k$. Numbers don't help the attacker.
- **Many drones.** The ring fires as fast as it can: $s_k \approx F_k$, so each drone's
  risk falls as $h_k = p_k F_k / N_k$. Numbers swamp the defence. This is saturation.

That leads to a firm rule. If a drone's risk can only fall as more drones arrive (crews
that can't keep up, missiles that run out), the share that gets through can only rise
with the size of the raid. The proof is in the module docstring, and a property test
checks it on the model. Only surge ($\sigma > 0$) can make big raids leak a smaller share.

With no limit on firing and no stock to run out, the model has a closed form. A drone
spends $w_k / v$ hours crossing a ring $w_k$ km wide, so the share that gets through is

```math
\text{share through} = \exp\!\left[-\lambda \left(\frac{R}{100}\right)^{\sigma}\right],
\qquad \lambda = \sum_k \frac{p_k\, e_k\, w_k}{v}
```

For the Ukraine fit, $\lambda = 1.79$ and $\sigma = 0.19$. This matches the full simulation
to three decimal places.

![Two raids flying through the three rings of the Ukraine-fitted defence: 24% of a 30-drone raid gets through, 9% of a 500-drone raid](docs/raid.gif)

Two raids against the defence fitted to Ukraine. A quarter of the small raid gets through.
Of the big raid, less than a tenth does.

### 4. Fitted to Ukraine's data

The Ukrainian Air Force reports each night how many drones were launched and how many were
stopped. `data.ukraine_air_raids()` fetches these reports, compiled by Petro Ivaniuk on
Kaggle (CC BY-NC-SA 4.0, so Jomini fetches them at run time and never ships them). The fit
uses the 778 nights with at least five Shahed-type drones launched, from September 2024 to
November 2026. It groups nights by raid size compared with the typical raid that month
(125 drones), so a month of heavy raids doesn't get confused with a big raid.

| Parameter | Estimate |
|---|---|
| Kill chance per shot, $p_k$ (interceptors, guns, missiles) | 0.63, 0.32, 0.90 |
| Surge, $\sigma$ | 0.19 |
| Firing capacity, $F_k$ | never reached on a typical night (set to unlimited) |
| Stock, $M_k$ | doesn't run out within a night |
| Overall lethality, $\lambda$ | 1.79 |

| Raid size | Got through (data) | Model |
|---|---|---|
| ¼ of typical | 16.5% | 23.5% |
| ½ | 19.7% | 19.2% |
| ¾ | 18.3% | 17.2% |
| typical | 16.1% | 15.5% |
| 1⅓× | 13.1% | 13.9% |
| 2× | 11.2% | 11.9% |
| 4× | 8.5% | 8.5% |

The model is off by 1.5 percentage points on average. It overstates leakage on the
smallest raids.

A surge of 0.19 means that when a raid doubles, each drone is shot at about 14% more
often. Before autumn 2024 there is no sign of this. `saturation_test` checks it directly
against nights in the same month:

| Nights (5+ launched) | Change in log-odds of being stopped when the raid doubles (95% CI) |
|---|---|
| Shahed, Sep 2022 to Aug 2024 | −0.13 (−0.28 to +0.02) |
| Shahed, Sep 2024 to Sep 2025 | **+0.32** (+0.25 to +0.40) |
| Shahed, Oct 2025 to Nov 2026 | **+0.35** (+0.30 to +0.41) |
| Cruise missiles, 2022 to 2026 | **+0.24** (+0.10 to +0.39) |
| Ballistic missiles, 2022 to 2026 | +0.41 (−0.04 to +0.87) |

What the fit can't tell you:

- The reports give only launched and stopped. They don't say which ring made the kill, so
  the data fix only the overall lethality $\lambda$ and the surge $\sigma$. The ring
  ranges, costs and the split of kills between rings are placeholders.
- These are Ukraine's own figures. Russia began launching decoys in large numbers in
  autumn 2024, and decoys that never arrive count as stopped.
- The fit can't tell real surge apart from decoys making up more of bigger raids, from
  route planning, or from how the numbers are reported.

`scripts/fit_ukraine_defense.py` reproduces the fit, and `scripts/make_raid_gif.py` draws the
animation.

## Other models

- **Escalation** (`jomini.escalation`): measures how much violence in a war feeds on
  itself, from the UCDP event data (about 418,000 events since 1989). In Russia–Ukraine,
  each event sets off 0.69 more on average; in Sudan, 0.23.
- **Effectiveness over time** (`jomini.dynamics`): tracks how many losses each side takes
  per enemy loss, period by period, with error bands, from Oryx's photo-confirmed
  equipment losses.
- **Historical battles** (`jomini.cdb90`, `jomini.research`, `jomini.stochastic`): 660
  land battles from 1600 to 1982, each with strengths, losses, commanders and a written
  account. Jomini fits attrition models to them, tests which factors made armies deadlier,
  and estimates each side's chance of winning from how much loss armies took before they
  broke.
- **Missile exchanges** (`jomini.modern`): Hughes' salvo model for fleets and missile duels.
- **Data loaders** (`jomini.data`): Ukrainian General Staff loss claims, Oryx, UCDP, daily
  data for Kursk 1943, and 63 American Civil War battles.

## Use it from Claude

Jomini runs as an [MCP](https://modelcontextprotocol.io) server, so Claude and other AI
agents can use the models and data directly. Ask "What happens if Russia launches 800
Shaheds tonight?" or "Is violence in Sudan feeding on itself?" and the answer comes from
the data, not a guess.

**Claude Code:**

```bash
claude mcp add jomini -- uvx --from "jomini[mcp] @ git+https://github.com/umitkaanusta/Jomini" jomini-mcp
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "jomini": {
      "command": "uvx",
      "args": ["--from", "jomini[mcp] @ git+https://github.com/umitkaanusta/Jomini", "jomini-mcp"]
    }
  }
}
```

| Tool | What it does |
|---|---|
| `air_defense_raid` | Run drone raids against the Ukraine-fitted defence, or one with fixed capacity, for one night or many |
| `air_defense_saturation_test` | Test Ukraine's nightly reports for saturation, by weapon and period |
| `conflict_events` | Monthly events and deaths for any conflict or country since 1989 |
| `escalation_model` | How strongly violence in a conflict feeds on itself |
| `effectiveness_trend` | How each side's effectiveness changed over time |
| `get_loss_series` | Loss time series: Kursk, Ukrainian General Staff, Oryx |
| `loss_intensity` | The smoothed loss rate behind a noisy series |
| `search_battles`, `get_battle` | Find a historical battle and read its full record |
| `fit_battle`, `what_if`, `simulate_battle` | Measure each side's effectiveness in a battle, and refight it with changes |
| `compare_effectiveness`, `combat_power` | How deadly an army was, and which factors made it so |
| `estimate_law`, `fit_law_to_series` | Fit attrition models to groups of battles or to loss series |
| `breakpoint_distribution` | How much loss an army takes before it breaks |
| `salvo_exchange` | Missile and drone exchanges with Hughes' salvo model |
| `list_datasets`, `cite` | Every dataset's coverage, licence and caveats, and BibTeX |

Every result lists its sources and methods. Pass their keys to `cite` for BibTeX. Each
dataset's caveats, such as "one side's claims", travel with the results.

## Sources and citation

```python
from jomini import catalog

catalog.DATASETS["ukraine_air_raids"].caveats  # read these before drawing conclusions
print(catalog.bibtex(["ukraine_air_raids", "air_defense"]))
```

`jomini.catalog` records each dataset's coverage, licence, caveats and references, and
the papers behind each method. Check entries against the publisher before you cite them
formally.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check && uv run ruff format --check && uv run mypy
```

## License

MIT
