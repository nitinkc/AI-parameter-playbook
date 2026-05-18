# Experiment 2: Top-p (Nucleus Sampling) Sweep

## The Question

**What does top-p *actually* do? How does it differ from temperature?**

- Temperature changes *all* probabilities uniformly (stretches the distribution)
- Top-p does something **different**: it *filters out* low-probability tokens entirely
- But the cutoff is **dynamic** based on certainty—how?

## What You'll Do

Keep temperature fixed and vary `top_p`, watching how many tokens "survive" the nucleus filter.

## The Setup

Same fake logits as Experiment 1:

```python
logits = np.array([2.2, 1.8, 1.4, 0.9, 0.2, 0.1, -0.3, -0.6, -0.8, -1.0], dtype=float)
```

## Your Experiment: Isolate Top-p

Fix `temperature=1.0` and run with these `top_p` values:

| Run | Temperature | Top-p | Top-k | What to observe |
|-----|-------------|-------|-------|-----------------|
| 1   | 1.0         | 0.2   | 0     | Very tight nucleus; maybe only 1-2 tokens |
| 2   | 1.0         | 0.5   | 0     | ~50% cumulative probability; ~3-4 tokens |
| 3   | 1.0         | 0.8   | 0     | ~80% of mass; ~6-7 tokens |
| 4   | 1.0         | 0.95  | 0     | Nearly all tokens, just removing tail |
| 5   | 1.0         | 1.0   | 0     | No filtering; no tokens removed |

## How to Run

Replace the settings:

```python
settings = [
    dict(name='Top-p=0.2 (very tight)',      temperature=1.0, top_k=0, top_p=0.2),
    dict(name='Top-p=0.5 (medium)',          temperature=1.0, top_k=0, top_p=0.5),
    dict(name='Top-p=0.8 (loose)',           temperature=1.0, top_k=0, top_p=0.8),
    dict(name='Top-p=0.95 (nearly all)',     temperature=1.0, top_k=0, top_p=0.95),
    dict(name='Top-p=1.0 (no filter)',       temperature=1.0, top_k=0, top_p=1.0),
]
```

Run:

```bash
python3 experiments/local/sampling_simulator.py
```

## What to Watch For

As you decrease `top_p` from 1.0 → 0.2, observe:

1. **How many tokens have non-zero probability?**
   - At p=0.2, maybe only 1-2 tokens survive
   - At p=1.0, all 10 tokens have a chance

   *Why?* Top-p filters by cumulative probability. Lower p = stricter filter = fewer tokens.

2. **Entropy trend**: Does it decrease as p decreases?
   - Yes! Fewer candidates = lower entropy = more predictable

3. **Do the probabilities of *surviving* tokens change?**
   - Yes. When a token survives the filter, it gets renormalized (rescaled up so the nucleus totals 100%)
   
   Example:
   - Before filter: tokens have probabilities [0.30, 0.25, 0.20, 0.15, ...]
   - After top-p=0.75: only first 3 survive; they renormalize to [0.40, 0.33, 0.27]

Example output:

```
=== Top-p=0.2 (very tight) ===
Entropy: 0.721
approve       0.667
reject        0.333
review        0.000
escalate      0.000
...

=== Top-p=0.8 (loose) ===
Entropy: 2.234
approve       0.240
reject        0.198
review        0.167
escalate      0.145
delay         0.102
audit         0.078
optimize      0.050
notify        0.015
...
```

## The Insight

**Top-p selects a *dynamic nucleus* of tokens.**

Unlike top-k (which always keeps exactly k tokens regardless of confidence), top-p adjusts based on the model's certainty:

- When the model is **confident** (one token much higher prob): nucleus is small
- When the model is **uncertain** (many tokens similar prob): nucleus is larger

This is often called "adaptive" or "self-adjusting" filtering.

## Math Behind It

1. Compute softmax probabilities from logits
2. Sort tokens by probability (descending)
3. Accumulate probabilities: token 1 = 0.30, token 1+2 = 0.55, token 1+2+3 = 0.75, ...
4. Keep tokens until cumulative sum ≥ p
5. Renormalize: divide all survivors by their sum (so they sum to 1)
6. Discard the rest (set their probability to 0)

## Real-World Implication

- **Safety-critical tasks** (medical diagnosis, financial advice): Use top-p=0.5 or lower (strict nucleus)
- **Creative tasks** (ideation, content generation): Use top-p=0.9+ (loose nucleus)
- **Balanced tasks**: top-p=0.75 to 0.85 is a common default

## Key Difference from Temperature

| Aspect | Temperature | Top-p |
|--------|-------------|-------|
| Mechanism | Scales all logits uniformly | Filters + renormalizes |
| Effect | Makes all tokens more/less likely *proportionally* | **Creates** a candidate set |
| Dynamic? | No—fixed for a run | Yes—adapts to confidence |
| Order | Preserves ranking (most likely stays most likely) | Preserves ranking within nucleus |

## Next Step

You now understand:
- Temperature = stretches distribution
- Top-p = selects a nucleus

Ready to see how they interact? Move to:

👉 **[Experiment 3: Top-k Sweep →](03-top-k.md)**

---

## Optional: Dive Deeper

Try varying **both** temperature and top-p to see the interaction:

```python
settings = [
    dict(name='T=0.5, p=0.5',  temperature=0.5, top_k=0, top_p=0.5),
    dict(name='T=1.0, p=0.5',  temperature=1.0, top_k=0, top_p=0.5),
    dict(name='T=1.5, p=0.5',  temperature=1.5, top_k=0, top_p=0.5),
]
```

Notice: Temperature affects *how confident* the distribution is about its nucleus; top-p controls *which tokens* get a shot.

--8<-- "_abbreviations.md"
