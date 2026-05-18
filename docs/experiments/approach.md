# How experiments work

Parameter tuning is easiest when you treat the model like a system under test.

## Principles

1. **Change one knob at a time** (or use a small grid).
2. Run **multiple trials** when randomness > 0.
3. Log everything: prompt, parameters, output, token usage.
4. Evaluate per use-case with a metric aligned to business needs.

## A/B/C sweeps

We recommend:

- Sweep temperature with `top_p=1`
- Then sweep top_p with a fixed temperature
- Then add penalties only if repetition is a confirmed issue

This repo provides scripts under `scripts/`.

--8<-- "_abbreviations.md"
