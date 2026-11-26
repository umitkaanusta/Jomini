# Jomini

[![CI](https://github.com/umitkaanusta/Jomini/actions/workflows/ci.yml/badge.svg)](https://github.com/umitkaanusta/Jomini/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Simulate battles from the musket to the drone.**

Jomini provides models of attrition and escalation, and lets you fit them to historical
battles and, where data exists, to wars still being fought.

It also runs as an [MCP server for your AI agents](#use-it-from-claude). The Python package
on PyPI is out of date; I'll update it once my PyPI account is unblocked.

We don't celebrate or glorify war or any other violence. Jomini is for studying how wars
are fought today and how they were fought in the past.

## Simulating a drone raid

![Two raids flying through three rings of air defence: 24% of a 30-drone raid gets through, 9% of a 500-drone raid](docs/raid.gif)

Two drone raids, one of 30 drones and one of 500, fly at a city by night. Three rings of
defence stand in their way: interceptor drones far out, mobile gun teams closer in, and
surface-to-air missiles around the city.

The simulation uses a model of drone swarms against layered air defence by Charles
Randow, Ryan Allaire and Antonio Mastroberardino (2026). It treats the swarm as a cloud
of drones that flows toward the target and thins out as each ring fires into it.

We fitted the model to 778 nights of Ukrainian Air Force reports, from September 2024 to
November 2026. The fitted defence stops three drones in four from the small raid, and
more than nine in ten from the big one.

<details>
<summary>The equations and the fit</summary>

**The equations**

Each ring is one kind of equipment, with its own range, chance to kill, rate of fire,
stock of ammunition and cost per shot. In the demo:

```
city <-- missiles --|-- mobile guns --|-- interceptor drones --|<-- raid
0 km              40 km            150 km                   300 km
```

Let <i>u</i>(<i>x</i>, <i>t</i>) be the number of drones per kilometre at distance <i>x</i> from the city. They
fly in at speed <i>v</i> and spread out as they go. Ring <i>k</i> kills them at rate <i>h</i><sub>k</sub>:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/eq/raid-dark.svg">
  <img alt="du/dt = v du/dx + D d2u/dx2 - sum over rings of h_k u" src="docs/eq/raid-light.svg">
</picture>

Ring <i>k</i> has <i>N</i><sub>k</sub> drones inside it and fires <i>s</i><sub>k</sub> shots an hour, until its stock <i>M</i><sub>k</sub>
runs out:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/eq/rings-dark.svg">
  <img alt="N_k = drones in ring k; s_k = F_k e_k N_k / (F_k + e_k N_k); h_k = p_k s_k / N_k; dM_k/dt = -s_k" src="docs/eq/rings-light.svg">
</picture>

When a big raid is coming, the defence can be reinforced. For a raid of <i>R</i> drones, <i>F</i><sub>k</sub>
and <i>e</i><sub>k</sub> are multiplied by (<i>R</i>/100)<sup>σ</sup>. Drones that reach <i>x</i> = 0 get through.

| Symbol | Meaning |
|---|---|
| <i>v</i> | Drone speed (185 km/h) |
| <i>D</i> | How fast the raid spreads out (50 km²/h) |
| <i>p</i><sub>k</sub> | Chance that one shot kills |
| <i>e</i><sub>k</sub> | How often a lone drone is shot at (shots per hour) |
| <i>F</i><sub>k</sub> | The most shots the ring can fire in an hour |
| <i>M</i><sub>k</sub> | Shots left in stock |
| σ | Surge: how much the defence is reinforced for big raids |

**What it means**

- **Few drones:** every drone runs the same risk, <i>h</i><sub>k</sub> = <i>p</i><sub>k</sub> <i>e</i><sub>k</sub>. Numbers don't help the
  attacker.
- **Many drones:** the ring fires flat out, so each drone's risk falls as <i>p</i><sub>k</sub> <i>F</i><sub>k</sub>/<i>N</i><sub>k</sub>.
  Numbers swamp the defence.

So a defence that can't reinforce lets a bigger share through as raids grow. If big raids
leak a smaller share, the defence must be reinforcing (σ > 0).

**Fitted to Ukraine's data**

Each night the Ukrainian Air Force reports how many drones were launched and how many
were stopped. Jomini downloads these reports, compiled by Petro Ivaniuk, when you use
them. The fit uses 778 nights with at least five drones launched, and compares each night
with the typical raid that month (125 drones).

| Parameter | Estimate |
|---|---|
| Chance one shot kills (drones, guns, missiles) | 0.63, 0.32, 0.90 |
| Surge, σ | 0.19: when a raid doubles, each drone is shot at 14% more often |
| Most shots per hour, <i>F</i><sub>k</sub> | never reached on a typical night |
| Stock, <i>M</i><sub>k</sub> | doesn't run out within a night |

| Raid size | Got through (data) | Model |
|---|---|---|
| ¼ of typical | 16.5% | 23.5% |
| ½ | 19.7% | 19.2% |
| typical | 16.1% | 15.5% |
| 2× | 11.2% | 11.9% |
| 4× | 8.5% | 8.5% |

**Why do small raids get through more often?** A big raid is spotted early, and more of
the defence turns out for it: more gun teams, more interceptors in the air. A small raid
meets whatever is on duty that night. In the model this is the surge: when a raid
doubles, each drone is shot at about 14% more often.

**Reproduce it**

```bash
uv sync --all-extras
uv run python scripts/fit_ukraine_defense.py
uv run python scripts/make_raid_gif.py
```

</details>

## A missile and drone strike

![A hundred ballistic missiles alone: 6 get through. The same missiles behind 150 drones: 33 get through](docs/strike.gif)

A hundred ballistic missiles fly at a city. In the top run they come alone. In the bottom
run, 150 drones are sent first, timed to reach the city just before the missiles, as in
the mass strikes of the Iran–Israel wars. Three rings defend the city: fighters that hunt
drones far out, an upper tier that shoots only at missiles, and a lower tier near the city
that shoots at both.

The model is the drone raid model, extended so that drones and missiles share the
defence. A ring splits its fire between the attackers inside it in proportion to their
numbers, a rule from C. S. Holling's work on predators and prey (1959).

Alone, 6 missiles in 100 get through. Behind the drones, 33 do: the drones use up the
lower tier's interceptors before the missiles arrive.

<details>
<summary>The equations and the numbers</summary>

Each wave <i>j</i> (drones, missiles) has its own density <i>u</i><sub>j</sub>(<i>x</i>, <i>t</i>) and speed <i>v</i><sub>j</sub>, and
follows the same equation as the drone raid. Ring <i>k</i> fires at all the waves it can
engage and splits its shots between them:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/eq/strike-dark.svg">
  <img alt="s_kj = F_k e_k N_kj / (F_k + e_k sum_i N_ki); h_kj = p_kj s_kj / N_kj; dM_k/dt = -sum_j s_kj" src="docs/eq/strike-light.svg">
</picture>

<i>N</i><sub>kj</sub> is the number of wave <i>j</i> inside ring <i>k</i>, and <i>p</i><sub>kj</sub> is the ring's chance of
killing one of them with one shot (0 if it can't engage them). Every shot, at a drone or
a missile, comes out of the same stock <i>M</i><sub>k</sub>.

| Ring | Range | Engages | Chance one shot kills | Interceptors |
|---|---|---|---|---|
| Fighters | 300–120 km | drones | 0.6 | unlimited |
| Upper tier | 120–40 km | missiles | 0.8 | 70 |
| Lower tier | 40–0 km | both | 0.7 | 90 |

The drones fly at 185 km/h and arrive over an hour. The missiles fly at 7,000 km/h and
arrive within six minutes, two hours after the first drones.

These numbers are made up to show the mechanism. There is too little public data on
single strikes to fit them.

**Reproduce it**

```bash
uv sync --all-extras
uv run python scripts/make_strike_gif.py
```

</details>

## Waterloo, with a bigger French army

![Waterloo as fought: the French break around 7 pm and Napoleon wins 27 refights in 100. With half as many men again he wins 37](docs/waterloo.gif)

Waterloo, 18 June 1815. Napoleon attacked with 68,000 men. Against him stood 138,000
British, Dutch and Prussian troops, counted together. The top run replays the battle as
fought. The bottom run gives Napoleon half as many men again: 102,000.

Losses follow Frederick Lanchester's equations of attrition (1916), fitted to the real
casualties. Each army breaks when its losses reach a level learned from 660 battles
fought between 1600 and 1982, following Robert Helmbold's work on breaking points
(1971).

As fought, the French break at about seven in the evening, close to when the Imperial
Guard's attack failed on the day. Napoleon wins 27 refights in 100. With half as many men
again he wins 37. More men help, but he still loses most of the time.

<details>
<summary>The equations and the numbers</summary>

<i>R</i> and <i>B</i> are the French and allied armies. Each side's losses grow with both armies'
sizes:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/eq/waterloo-dark.svg">
  <img alt="dR/dt = -beta R B; dB/dt = -rho R B" src="docs/eq/waterloo-light.svg">
</picture>

Fitted to the real casualties (25,000 French, 22,500 allied in half a day), a French
soldier was 0.9 times as deadly as an allied one (ρ/β).

An army breaks once it has lost a share <i>b</i> of its men, where log <i>b</i> is normally
distributed. Fitted to 660 battles, the median for 1815 is 25% for an attacker and 27%
for a defender. Each refight draws new breaking points for both sides.

This uses the linear form of Lanchester's equations, which fits the 660 battles best.
Under the square form, where numbers count for more, the bigger French army would win
53 refights in 100.

**Reproduce it**

```bash
uv sync --all-extras
uv run python scripts/make_waterloo_gif.py
```

</details>

## What it lacks

- **Space.** No maps, terrain or routes. The raid model knows only how far each drone is
  from its target.
- **Everything but attrition.** No manoeuvre, morale, command, supply or weather, except
  that a side can break after heavy losses.
- **Real orders of battle.** Ranges, weapon numbers and costs are placeholders.
- **Decoys and jamming.** They show up only as drones that never arrive.
- **Forecasts.** The fits describe what happened. They don't predict what will.

## Other models

- **Escalation:** how much violence in a war feeds on itself, from UCDP's 418,000 events
  since 1989. In Russia–Ukraine each event sets off 0.69 more on average; in Sudan, 0.23.
- **Effectiveness over time:** how many losses each side takes for each enemy loss, period
  by period, from Oryx's photo-confirmed equipment losses.
- **Historical battles:** 660 land battles from 1600 to 1982 with strengths, losses,
  commanders and a written account of each. Fit attrition models to them, find what made
  armies deadlier, and estimate each side's chance of winning.
- **Missile exchanges:** Hughes' salvo model for fleets and missile duels.

## Use it from Claude

Jomini can run as an MCP server, so Claude and other AI agents can use the models and data
directly.

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
| `air_defense_raid` | Run drone raids against the Ukraine-fitted defence or one you design, for one night or many |
| `air_defense_strike` | Run mixed strikes of drones and missiles against rings that share their interceptors |
| `conflict_events` | Monthly events and deaths for any conflict or country since 1989 |
| `escalation_model` | How strongly violence in a conflict feeds on itself |
| `effectiveness_trend` | How each side's effectiveness changed over time |
| `get_loss_series`, `loss_intensity` | Daily and smoothed losses: Kursk, Ukrainian General Staff, Oryx |
| `search_battles`, `get_battle` | Find a historical battle and read its full record |
| `fit_battle`, `what_if`, `simulate_battle` | Measure each side in a battle, and refight it with changes |
| `compare_effectiveness`, `combat_power` | How deadly an army was, and what made it so |
| `estimate_law`, `fit_law_to_series` | Fit attrition models to groups of battles or to loss series |
| `breakpoint_distribution` | How much loss an army takes before it breaks |
| `salvo_exchange` | Missile and drone exchanges |
| `list_datasets`, `cite` | Every dataset's coverage, licence and caveats, and BibTeX |

Every answer lists the sources it used, and `cite` turns them into BibTeX.

## Sources

| Dataset | Licence | |
|---|---|---|
| CDB90: CAA Database of Battles | CC0 1.0 (data); MIT (code) | ships with Jomini |
| Battle of Kursk, southern front, daily | Public domain (U.S. Government work, 17 U.S.C. 105) | ships with Jomini |
| American Civil War battles (Livermore 1900) | Public domain (1900); transcription ODC-BY 1.0 | ships with Jomini |
| UCDP Georeferenced Event Dataset 26.1 | CC BY 4.0 | downloaded when used |
| UCDP Candidate GED (provisional, current year) | CC BY 4.0 | downloaded when used |
| Russian losses claimed by Ukraine's General Staff | MIT (compilation) | downloaded when used |
| Oryx visually confirmed equipment losses (weekly) | CC BY 4.0 (compilation) | downloaded when used |
| Oryx confirmed losses by event date (both sides) | MIT (compilation) | downloaded when used |
| Russian drone and missile raids on Ukraine (nightly) | CC BY-NC-SA 4.0 (non-commercial; not bundled) | downloaded when used |

<details>
<summary>Papers and books</summary>

- Lanchester, Frederick W. (1916). *Aircraft in Warfare: The Dawn of the Fourth Arm*. Constable.
- Bracken, Jerome (1995). *Lanchester Models of the Ardennes Campaign*. Naval Research Logistics.
- Helmbold, Robert L. (1964). *Some Observations on the Use of Lanchester's Theory for Prediction*. Operations Research.
- Helmbold, Robert L. (1971). *Decision in Battle: Breakpoint Hypotheses and Engagement Termination Data*. RAND Corporation.
- Hartley, Dean S., III (1995). *A Mathematical Model of Attrition Data*. Naval Research Logistics.
- Fricker, Ronald D., Jr. (1998). *Attrition Models of the Ardennes Campaign*. Naval Research Logistics.
- Lucas, Thomas W.; Turkes, T. (2004). *Fitting Lanchester Equations to the Battles of Kursk and Ardennes*. Naval Research Logistics.
- Hughes, Wayne P., Jr. (1995). *A Salvo Model of Warships in Missile Combat Used to Evaluate Their Staying Power*. Naval Research Logistics.
- Dupuy, Trevor N. (1979). *Numbers, Predictions and War: Using History to Evaluate Combat Factors and Predict the Outcome of Battles*. Bobbs-Merrill.
- Rotte, Ralph; Schmidt, Christoph M. (2003). *On the Production of Victory: Empirical Determinants of Battlefield Success in Modern War*. Defence and Peace Economics.
- Brunton, Steven L.; Proctor, Joshua L.; Kutz, J. Nathan (2016). *Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems*. Proceedings of the National Academy of Sciences. https://doi.org/10.1073/pnas.1517384113
- Rauch, H. E.; Tung, F.; Striebel, C. T. (1965). *Maximum Likelihood Estimates of Linear Dynamic Systems*. AIAA Journal.
- (2024). *Analyzing Russia–Ukraine War Patterns Based on Lanchester Model Using SINDy Algorithm*. Mathematics. https://doi.org/10.3390/math12060851
- Hawkes, Alan G. (1971). *Spectra of Some Self-Exciting and Mutually Exciting Point Processes*. Biometrika.
- Lewis, Erik; Mohler, George; Brantingham, P. Jeffrey; Bertozzi, Andrea L. (2012). *Self-Exciting Point Process Models of Civilian Deaths in Iraq*. Security Journal.
- Browning, Raiha; Patten, Hamish; Rousseau, Judith; Mengersen, Kerrie (2024). *Bayesian Spatiotemporal Modelling of Political Violence and Conflict Events Using Discrete-Time Hawkes Processes*. arXiv:2408.14940.
- Sundberg, Ralph; Melander, Erik (2013). *Introducing the UCDP Georeferenced Event Dataset*. Journal of Peace Research.
- Davies, Shawn; Pettersson, Therése; Öberg, Magnus (2026). *Organized Violence 1989–2025, and Violent Political Protests*. Journal of Peace Research. https://doi.org/10.1093/jopres/xjag046
- U.S. Army Concepts Analysis Agency (1991). *Database of Battles, Version 1990 (CDB90)*. U.S. Army Concepts Analysis Agency.
- Arnold, Jeffrey B. (2026). *CDB90 Battle Dataset (release 10.8.0)*. https://github.com/jrnold/CDB90. https://doi.org/10.5281/zenodo.23928303
- Livermore, Thomas L. (1900). *Numbers and Losses in the Civil War in America, 1861–65*. Houghton, Mifflin and Company.
- Arnold, Jeffrey B. (n.d.). *American Civil War Battle Data*. https://github.com/jrnold/acw_battle_data. https://doi.org/10.6084/m9.figshare.1515995
- Ivaniuk, Petro (n.d.). *2022 Ukraine–Russia War Dataset*. https://github.com/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset.
- Oryx (n.d.). *Attack on Europe: Documenting Equipment Losses During the Russian Invasion of Ukraine*. https://www.oryxspioenkop.com/.
- Scarnecchia, Daniel (n.d.). *oryx_data*. https://github.com/scarnecchia/oryx_data.
- Drake, Lee (n.d.). *Russia-Ukraine*. https://github.com/leedrake5/Russia-Ukraine.
- Randow, Charles; Allaire, Ryan H.; Mastroberardino, Antonio (2026). *Modeling Drone Swarms with a PDE Extension of Lanchester's Laws*. The Journal of Defense Modeling and Simulation. https://doi.org/10.1177/15485129261436073
- Holling, C. S. (1959). *Some Characteristics of Simple Types of Predation and Parasitism*. The Canadian Entomologist.
- Ivaniuk, Petro (n.d.). *Massive Missile Attacks on Ukraine*. https://www.kaggle.com/datasets/piterfm/massive-missile-attacks-on-ukraine.

</details>

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check && uv run ruff format --check && uv run mypy
```

## License

MIT
