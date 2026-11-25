# Jomini-Bench

Can an LLM forecast historical battles, and do Jomini's tools help it?

Each item is one of 149 held-out decisive CDB90 battles, described only by what was known
before it was fought: strengths, equipment, the defender's posture, frontage and era. The
model gives the attacker's win probability and each side's casualties.

| Axis | Values | Measures |
|---|---|---|
| Variant | `named` (battle, date, combatants) / `anon` (no names or places, decade only) | how much of the score comes from remembering famous battles |
| Condition | `plain` / `tools` (Jomini's simulation and research tools; the lookup tools are withheld because they reveal outcomes) | whether quantitative tools help |

Metrics: accuracy, Brier score and log loss on the winner, and the median
`|log(predicted / actual)|` casualty error (0.69 means typically off by a factor of 2).

## Baselines (no API calls, fitted on the 453 training battles)

| Baseline | Accuracy | Brier | Casualty log error |
|---|---|---|---|
| Base rate (attacker wins 65%) | 0.644 | 0.229 | 0.889 |
| Logistic regression (force ratio, posture, era) | 0.624 | 0.225 | 0.681 |
| Jomini (linear law, expected quality, breakpoint Monte Carlo) | 0.604 | 0.230 | **0.674** |

What's known before a battle carries almost no signal about who wins. An LLM that beats
these baselines on `anon` items is finding something the models miss. Extra gains on
`named` items mostly reflect memory.

## Run

```bash
uv run --group bench python bench/jomini_bench.py items
uv run --group bench python bench/jomini_bench.py baselines
uv run --group bench python bench/jomini_bench.py run --variant anon --condition tools --n 20
uv run --group bench python bench/jomini_bench.py score bench/results/*.jsonl
```

`run` calls the Claude API and costs money (default model `claude-opus-5-5`, effort
`medium`). Use `--n` for a small pilot first. Results are written as JSONL to
`bench/results/`.
