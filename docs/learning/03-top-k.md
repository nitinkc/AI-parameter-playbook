# Experiment 3: Top-k Sweep

## The Question

**What does top-k do? How is it different from top-p?**

- Top-p selects a **cumulative** set (dynamic size)
- Top-k selects the **top k ranked** tokens (fixed size)
- When should you use one vs. the other?

## What You'll Do

Keep temperature and top-p fixed at their "neutral" values, then vary `top_k` to see how a hard rank-based cutoff affects the distribution.

## The Setup

Same logits:

```python
logits = np.array([2.2, 1.8, 1.4, 0.9, 0.2, 0.1, -0.3, -0.6, -0.8, -1.0], dtype=float)
```

If we rank by logit value:
1. 'approve' (2.2)
2. 'reject' (1.8)
3. 'review' (1.4)
4. 'escalate' (0.9)
5. 'delay' (0.2)
6. 'audit' (0.1)
7. 'optimize' (-0.3)
8. 'notify' (-0.6)
9. 'assign' (-0.8)
10. 'close' (-1.0)

## Your Experiment: Isolate Top-k

Fix `temperature=1.0`, `top_p=1.0` (no nucleus filter), and vary `top_k`:

| Run | Top-k | What to expect |
|-----|-------|-----------------|
| 1   | 1     | Only 'approve' |
| 2   | 3     | Only top 3: approve, reject, review |
| 3   | 5     | Top 5: approve, reject, review, escalate, delay |
| 4   | 10    | All tokens (no cutoff) |
| 5   | 0     | All tokens (k=0 means no limit) |

## How to Run

```python
settings = [
    dict(name='Top-k=1 (greedy)',       temperature=1.0, top_k=1, top_p=1.0),
    dict(name='Top-k=3',                temperature=1.0, top_k=3, top_p=1.0),
    dict(name='Top-k=5',                temperature=1.0, top_k=5, top_p=1.0),
    dict(name='Top-k=10 (all)',         temperature=1.0, top_k=10, top_p=1.0),
    dict(name='Top-k=0 (no limit)',     temperature=1.0, top_k=0, top_p=1.0),
]
```

Run:

```bash
python3 experiments/local/sampling_simulator.py
```

## What to Watch For

1. **Number of tokens with non-zero probability:**
   - k=1: Only 1 token ever chosen
   - k=3: Only 3 tokens ever chosen (by rank)
   - k=10: All 10 tokens have a chance

2. **Entropy:**
   - k=1: Entropy = 0 (completely deterministic)
   - k=3: Low entropy
   - k=10: Same as k=0 or no filter (full distribution entropy)

3. **Rank preservation:**
   - Top-k always keeps the *highest-ranked* tokens by logit value
   - Unlike top-p, rank cutoff is absolute (not based on probability mass)

Example output:

```
=== Top-k=1 (greedy) ===
Entropy: 0.0
approve       1.000
reject        0.000
review        0.000
...

=== Top-k=5 ===
Entropy: 1.612
approve       0.358
reject        0.268
review        0.186
escalate      0.117
delay         0.071
audit         0.000
...
```

## The Insight

**Top-k is a *ranked* filter, not a *probability* filter.**

It says: "Keep the top k tokens by logit rank, discard the rest."

### Top-k vs. Top-p at a glance

| Aspect | Top-k | Top-p |
|--------|-------|-------|
| Selection rule | Rank-based (keep top k) | Probability-based (keep cumulative ≥ p) |
| Size | Fixed (always k tokens) | Variable (adapts to confidence) |
| Typical use | Broad diversity, prevent tail tokens | Adaptive safety |
| Examples | k=5 in chat, k=20 for code | p=0.9 in creative, p=0.75 in safety |

## Real-World Implication

- **Code generation**: Often uses top-k=20..40 to prevent garbage tokens but allow reasonable diversity
- **Chat**: Might use top-k=5 to keep responses focused  
- **Classification**: Usually avoids top-k entirely, wants to see all options

## Why Both Exist

1. **Top-k is simpler mentally**: "What are my top 5 choices?"
2. **Top-p is more adaptive**: "Give me options until I've covered 90% confidence"

Many modern systems use **both**: apply top-k first (remove obvious garbage), then top-p (adaptive nucleus).

## Next Step

You now understand:
- Temperature = stretches distribution
- Top-p = selects a dynamic nucleus  
- Top-k = selects a fixed-size ranked set

Ready to see what happens when you combine them?

👉 **[Experiment 4: Combined Filters →](04-combined.md)**

---

## Optional: Dive Deeper

Try this experiment: compare how top-k and top-p behave differently with the same logits:

```python
# At what top-p value do we include the 5th token?
# At what top-k value do we include exactly the top 5?

# This shows that top_k=5 != top_p=(where 5th token enters)
# because top-p is probabilistic, not rank-based
```

Hint: Run a temperature=1.0 baseline, note which 5 tokens have the highest probability mass, then compute what top_p threshold includes those exact 5. Compare that to top_k=5 (which includes 5 by *rank*, not probability).

--8<-- "_abbreviations.md"
