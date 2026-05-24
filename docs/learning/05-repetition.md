# Experiment 5: Repetition Penalties

## The Question

**How do repetition penalties work, and what do they actually change?**

They are designed to reduce loops like "the the the" or repeated phrases, but they do not hard-ban repetition.

## Core Idea

A repetition penalty modifies logits of recently seen tokens before sampling.

- `penalty = 1.0`: no change
- `penalty > 1.0`: discourage repeated tokens

Common multiplicative rule:

```python
if token in recent_tokens:
    if logit > 0:
        logit /= penalty
    else:
        logit *= penalty
```

This asymmetry avoids over-penalizing already-unlikely tokens.

## Setup

Use sequence generation (not single-step sampling):

```python
logits = np.array([2.2, 1.8, 1.4, 0.9, 0.2, 0.1, -0.3, -0.6, -0.8, -1.0], dtype=float)
tokens = ['approve', 'reject', 'review', 'escalate', 'delay', 'audit', 'optimize', 'notify', 'assign', 'close']

# Example sequence settings
sequence_length = 20
lookback = 8
```

## Your Experiment: Penalty Strength Sweep

Run with fixed decoding and varying repetition penalty:

```python
settings = [
    dict(name='No penalty', temperature=0.8, top_k=0, top_p=1.0, repetition_penalty=1.0, sequence_len=8),
    dict(name='Light penalty (1.1x)', temperature=0.8, top_k=0, top_p=1.0, repetition_penalty=1.1, sequence_len=8),
    dict(name='Moderate penalty (1.3x)', temperature=0.8, top_k=0, top_p=1.0, repetition_penalty=1.3, sequence_len=8),
    dict(name='Heavy penalty (1.5x)', temperature=0.8, top_k=0, top_p=1.0, repetition_penalty=1.5, sequence_len=8),
    dict(name='Extreme penalty (2.0x)', temperature=0.8, top_k=0, top_p=1.0, repetition_penalty=2.0, sequence_len=8),
]
```

Run:

```bash
python3 notebooks/sampling_simulator.py
```

## What to Watch For

1. **Token diversity** usually increases as penalty rises.
2. **Entropy** often increases as repeated high-probability tokens are pushed down.
3. **Looping behavior** should decrease, especially for short repeated streaks.

Example trend:

```text
=== No penalty ===
Entropy: 1.234
approve       0.287
reject        0.201
...

=== Heavy penalty (1.5x) ===
Entropy: 2.012
approve       0.156
reject        0.155
review        0.143
...
```

## Important Caveats

- Penalties **discourage**, not forbid.
- Too strong penalties can hurt coherence by suppressing necessary words.
- Lookback window matters: tokens outside the window are no longer penalized.

## Practical Strength Guide

| Penalty | Behavior |
|---------|----------|
| 1.0 | Off |
| 1.1-1.2 | Mild discouragement |
| 1.3-1.5 | Noticeable anti-loop effect |
| > 1.8 | Can become unnatural/incoherent |

## Frequency vs Presence Penalty (API Mapping)

Many APIs expose additive penalties instead of multiplicative ones:

| Type | Typical effect |
|------|----------------|
| **Presence penalty** | Penalize any token that has appeared at least once |
| **Frequency penalty** | Penalize tokens more as they appear more times |

The intuition is similar: increase novelty pressure, reduce repetitive loops.

## Optional: Measure Repeats Directly

```python
def count_adjacent_repeats(generated_ids):
    repeats = 0
    for i in range(1, len(generated_ids)):
        if generated_ids[i] == generated_ids[i - 1]:
            repeats += 1
    return repeats
```

Then compare repeat counts across penalty settings.

## Task-Level Guidance

| Use case | Penalty | Lookback | Why |
|----------|---------|----------|-----|
| Summarization | 1.0 | N/A | Repetition may be acceptable for key terms |
| Code generation | 1.1 | 32 | Prevent loops while keeping syntax patterns |
| Chatbot | 1.2 | 64 | Improve conversational variety |
| Story generation | 1.15 | 128 | Encourage novelty without derailing style |

## Takeaway

> **Repetition penalties are anti-loop pressure, not a hard rule.** Start small and increase only when you observe looping.

👉 Next: **[Experiment 6: Real Use Cases](06-use-cases.md)**

--8<-- "_abbreviations.md"
