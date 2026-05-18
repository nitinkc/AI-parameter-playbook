# Experiment 4: Combined Filters (The Real World)

## The Question

**What happens when you combine temperature + top-p + top-k together? Do they conflict or cooperate?**

So far, you've isolated each parameter. But real systems use **multiple filters together**. Does order matter? Can you predict the outcome?

## What You'll Do

Apply all three filters in realistic combinations and watch how they interact.

## The Setup

Same logits:

```python
logits = np.array([2.2, 1.8, 1.4, 0.9, 0.2, 0.1, -0.3, -0.6, -0.8, -1.0], dtype=float)
tokens = ['approve', 'reject', 'review', 'escalate', 'delay', 'audit', 'optimize', 'notify', 'assign', 'close']
```

## The Filter Pipeline (Order Matters!)

The simulator applies filters in this order:

```
Raw logits
  ↓
[Temperature scaling: logits / T]
  ↓
[Top-k: keep top k by rank]
  ↓
[Top-p: keep cumulative ≥ p]
  ↓
[Softmax: convert to probabilities]
  ↓
[Sample]
```

**Key insight:** This ordering is standard. Most models apply filters in roughly this sequence.

## Your Experiment: 5 Realistic Scenarios

Run these settings:

### Scenario 1: Safety-critical (Classification/RAG)

```python
dict(name='Safety: T=0.3, k=0, p=1.0', 
     temperature=0.3, top_k=0, top_p=1.0)
```

**Rationale:** Low temperature keeps outputs consistent; no extra filters needed.

### Scenario 2: Balanced (Summarization)

```python
dict(name='Balanced: T=0.7, k=0, p=1.0', 
     temperature=0.7, top_k=0, top_p=1.0)
```

**Rationale:** Moderate temperature allows reasonable variation without hallucination.

### Scenario 3: Creative (Brainstorming)

```python
dict(name='Creative: T=1.2, k=0, p=0.9', 
     temperature=1.2, top_k=0, top_p=0.9)
```

**Rationale:** Higher temperature + nucleus sampling = diverse but coherent.

### Scenario 4: Diverse + Safe (Top-k guard rail)

```python
dict(name='Diverse+Safe: T=0.9, k=10, p=0.95', 
     temperature=0.9, top_k=10, top_p=0.95)
```

**Rationale:** Top-k prevents weird far-tail tokens; top-p adapts nucleus; moderate temperature.

### Scenario 5: Broad but Bounded (Code generation)

```python
dict(name='Broad: T=0.8, k=40, p=1.0', 
     temperature=0.8, top_k=40, top_p=1.0)
```

**Rationale:** Large top-k allows many plausible options, temperature control instead of filtering.

## How to Run

```python
settings = [
    dict(name='Safety: T=0.3, k=0, p=1.0', temperature=0.3, top_k=0, top_p=1.0),
    dict(name='Balanced: T=0.7, k=0, p=1.0', temperature=0.7, top_k=0, top_p=1.0),
    dict(name='Creative: T=1.2, k=0, p=0.9', temperature=1.2, top_k=0, top_p=0.9),
    dict(name='Diverse+Safe: T=0.9, k=10, p=0.95', temperature=0.9, top_k=10, top_p=0.95),
    dict(name='Broad: T=0.8, k=40, p=1.0', temperature=0.8, top_k=40, top_p=1.0),
]
```

Run:

```bash
python3 experiments/local/sampling_simulator.py
```

## What to Watch For

1. **Entropy comparison**: Which scenario has the lowest? Highest?
   - Safety should be lowest
   - Creative should be highest

2. **Distribution shape**: 
   - Safety: Very peaked (1-2 tokens dominate)
   - Creative: More spread out
   - Broad: Flatter (many tokens viable)

3. **How many tokens survive to sampling?**
   - Safety: Maybe 3-4 viable tokens
   - Broad: Most of the 10 tokens

4. **Interaction effect**: 
   - Does top-k at k=10 change anything vs. k=0 when top-p=0.95 is already restricting the nucleus?
   - Watch: Sometimes k doesn't help if p is already tight.

Example output comparison:

```
=== Safety: T=0.3 ===
Entropy: 0.614
approve       0.896
reject        0.087
review        0.013
escalate      0.003
...

=== Creative: T=1.2, p=0.9 ===
Entropy: 1.892
approve       0.201
reject        0.189
review        0.176
escalate      0.143
delay         0.151
audit         0.107
optimize      0.032
notify        0.001
...

=== Broad: T=0.8, k=40 ===
Entropy: 2.156
approve       0.287
reject        0.233
review        0.162
escalate      0.105
delay         0.063
audit         0.044
optimize      0.069
notify        0.021
assign        0.013
close         0.003
```

## The Insight: Filters Cooperate, Usually

1. **Top-k + top-p together**: 
   - Top-k prevents ultra-tail garbage
   - Top-p adapts within that safer set
   - They **cooperate** (both restrict, no real conflict)

2. **Temperature + filters together**:
   - Temperature controls the *shape* of the distribution
   - Filters control the *size* of the candidate set
   - Temperature acts first, so it affects what opportunities filters have

3. **Order matters**:
   If you applied filters *before* temperature:
   - You'd filter the logits directly
   - Temperature would then stretch the already-filtered logits
   - This can produce different results!

## Real-World Implication

This is why there's **no one magic setting**:

- **Safety-critical + want speed**: T=0.2, no filters needed (entropy ~0.6)
- **Want balance + natural diversity**: T=0.7 to 0.8 with top-p=0.95 (entropy ~1.8)
- **Want creativity but not chaos**: T=1.0 to 1.2 with top-p=0.85 (entropy ~2.0)
- **Want maximum controlled diversity**: T=0.8 to 0.9 with k=40, p=1.0 (entropy ~2.2)

## Next Step

You understand:
- Single parameters and their effects
- How multiple parameters interact
- Real-world scenario combinations

Ready to tackle penalties?

👉 **[Experiment 5: Repetition Penalties →](05-repetition.md)**

---

## Optional: Dive Deeper

Try graphing entropy vs. temperature for multiple top-p values:

```python
import matplotlib.pyplot as plt

temps = np.linspace(0.1, 2.0, 10)
for p in [0.5, 0.8, 1.0]:
    entropies = []
    for t in temps:
        out = run_experiment(tokens, logits, temperature=t, top_p=p, n_samples=10000)
        entropies.append(out['entropy'])
    plt.plot(temps, entropies, marker='o', label=f'top_p={p}')

plt.xlabel('Temperature')
plt.ylabel('Entropy')
plt.title('Temperature Effect on Entropy (by top_p setting)')
plt.legend()
plt.grid(True)
plt.show()
```

This shows you that temperature behavior **changes** depending on top_p. Neat!

--8<-- "_abbreviations.md"
