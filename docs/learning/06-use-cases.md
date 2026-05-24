# Experiment 6: Real Use Cases

## The Question

**How do you translate parameter intuition into task-specific settings?**

Different tasks need different output distributions:

- Classification needs correctness and low entropy.
- Summarization needs fidelity with moderate flexibility.
- Code generation needs diversity with guard rails.
- Creative writing needs high novelty without total chaos.

## Setup: Four Task-Shaped Logit Distributions

```python
# Classification (high confidence)
logits_class = np.array([4.5, 0.3, 0.1, -0.2, -0.5, -1.0, -1.2, -1.5, -2.0, -2.5], dtype=float)
tokens_class = ['INTENT_APPROVE', 'INTENT_REJECT', 'INTENT_REVIEW', 'TONE_FORMAL', 'TONE_CASUAL', 'ACTION_ESCALATE', 'ACTION_DELAY', 'ACTION_AUDIT', 'SAFETY_HOLD', 'SAFETY_BLOCK']

# Summarization (moderate confidence)
logits_sum = np.array([2.1, 1.9, 1.7, 1.3, 0.8, 0.5, 0.2, -0.2, -0.5, -1.0], dtype=float)
tokens_sum = ['BENEFIT', 'COST', 'TIMELINE', 'STAKEHOLDER', 'RISK', 'OPPORTUNITY', 'CONSTRAINT', 'DETAIL', 'TANGENT', 'NOISE']

# Code generation (many plausible continuations)
logits_code = np.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], dtype=float)
tokens_code = ['FOR_LOOP', 'IF_STMT', 'FUNC_DEF', 'CLASS_DEF', 'IMPORT', 'RETURN', 'ASSIGNMENT', 'CALL', 'COMMENT', 'DECORATOR']

# Creative writing (broad exploration desired)
logits_creative = np.array([1.5, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2], dtype=float)
tokens_creative = ['BRAVE', 'QUIRKY', 'DARK', 'HUMOROUS', 'POIGNANT', 'SARCASTIC', 'TENDER', 'WHIMSICAL', 'TRAGIC', 'MYSTICAL']
```

## Your Experiment: Choose Parameters per Task

Start by predicting entropy and output behavior, then run:

```python
use_cases = [
    ('Classification', tokens_class, logits_class, [
        dict(name='Classification (low entropy)', temperature=0.2, top_k=0, top_p=1.0),
    ]),
    ('Summarization', tokens_sum, logits_sum, [
        dict(name='Summarization (moderate entropy)', temperature=0.6, top_k=0, top_p=0.95),
    ]),
    ('Code', tokens_code, logits_code, [
        dict(name='Code (diverse but bounded)', temperature=0.8, top_k=30, top_p=1.0),
    ]),
    ('Creative', tokens_creative, logits_creative, [
        dict(name='Creative (high entropy)', temperature=1.1, top_k=0, top_p=0.85),
    ]),
]
```

Run:

```bash
python3 notebooks/sampling_simulator.py
```

## What to Watch For

1. **Entropy progression** should generally move from low (classification) to high (creative).
2. **Top-token dominance** should be strongest for classification.
3. **Viable token count** should increase with creativity level.

Typical range targets:

- Classification: entropy ~0.5-1.0
- Summarization: entropy ~1.5-2.0
- Code: entropy ~2.0-2.5
- Creative: entropy ~2.0-2.8

## Core Principle

There is no single best setting; there is only a setting aligned to your task objective.

| Task | Desired behavior | Typical strategy |
|------|------------------|------------------|
| Classification | Single correct output | T~0.1-0.2, low/no sampling filters |
| Factual QA / RAG | Accurate, grounded | T=0.2-0.4, top-p around 0.9-1.0 |
| Summarization | Faithful + fluent | T=0.4-0.6, top-p around 0.9-0.95 |
| Code generation | Correct + flexible | T=0.2-0.8, top-k or top-p guard rails |
| Creative writing | Novel and varied | T=0.9-1.1, top-p around 0.9-0.95 |
| Brainstorming | Maximum exploration | T around 1.2, broad candidate set |
| Chat/dialog | Coherent natural variety | T around 0.7, top-p around 0.9 |

## Calibration Heuristic

1. Start with **temperature** (deterministic vs creative goal).
2. Add **top-p** (`0.9` is a common baseline; reduce for precision).
3. Add **top-k** when you need a hard rank cutoff.
4. Add **repetition penalty** only if looping appears (start near `1.1`).
5. Validate with your task metric (accuracy, ROUGE, pass@k, etc.).

## Entropy-Driven Tuning Cheat Sheet

- **Need lower entropy:** decrease temperature, tighten top-p.
- **Need higher entropy:** increase temperature, loosen or disable filters.

Entropy bands:

- **0-1.0:** extraction/classification style behavior
- **1.5-2.0:** balanced summarization/QA behavior
- **2.2+:** exploratory creative behavior

## Real-World Checklist

- [ ] Define your task success metric first
- [ ] Start from a baseline (`temperature=0.7`, `top_p=1.0`, no penalties)
- [ ] Sweep one variable at a time
- [ ] Log prompts, settings, outputs, and metrics
- [ ] Compare against a control configuration
- [ ] Iterate toward your target behavior

## Next Steps

- Move to cloud testing: [Setup Overview](../setup/overview.md)
- Continue local exploration: [Interactive Playgrounds](../experiments/simulator-guide.md)
- Use broader reference docs: [Parameter Map](../parameters/parameter-map.md)

## Optional Challenge

Design your own use case:

1. Create a logit distribution that matches your task uncertainty.
2. Predict settings before running.
3. Run and inspect entropy/output quality.
4. Iterate based on observed behavior.

## What You Have Accomplished

You now understand:

- Temperature and distribution shaping
- Top-p nucleus filtering
- Top-k rank filtering
- Parameter interactions and ordering
- Repetition-control tradeoffs
- Task-specific tuning strategy

> This intuition transfers across providers and model families.

--8<-- "_abbreviations.md"

