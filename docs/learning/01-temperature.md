# Experiment 1: Temperature Sweep

## The Question

**How does temperature affect the distribution of token probabilities?**

- When temperature is **low**, does the model "lock in" to one answer?
- When temperature is **high**, does it explore more options?
- Is there a sweet spot?

## What You'll Do

You'll run the same logits through the sampling pipeline with **different temperatures**, then watch how the probability distribution changes.

## The Setup

Open `experiments/local/sampling_simulator.py` and look for this section:

```python
if __name__ == '__main__':
    tokens = ['approve', 'reject', 'review', 'escalate', 'delay', 'audit', 'optimize', 'notify', 'assign', 'close']
    logits = np.array([2.2, 1.8, 1.4, 0.9, 0.2, 0.1, -0.3, -0.6, -0.8, -1.0], dtype=float)
```

These fake logits represent a case where the model leans toward 'approve' but has uncertainty below that.

## Your Experiment: Isolate Temperature

Run the simulator **6 times** with these settings (only changing temperature):

| Run | Temperature | Top-p | Top-k | What to observe |
|-----|-------------|-------|-------|-----------------|
| 1   | 0.2         | 1.0   | 0     | Very narrow, peaked distribution |
| 2   | 0.5         | 1.0   | 0     | Narrower than baseline |
| 3   | 1.0         | 1.0   | 0     | "Baseline" distribution (unchanged logits) |
| 4   | 1.5         | 1.0   | 0     | Flatter; more tokens have viable probability |
| 5   | 2.0         | 1.0   | 0     | Very flat; even low-prob tokens get sampled |
| 6   | 0.1         | 1.0   | 0     | Extreme: almost always 'approve' |

## How to Run

Modify the `settings` list in the simulator to include these temperatures:

```python
settings = [
    dict(name='T=0.1 (extreme determinist)', temperature=0.1, top_k=0, top_p=1.0),
    dict(name='T=0.2 (very det.)',          temperature=0.2, top_k=0, top_p=1.0),
    dict(name='T=0.5 (det.)',                temperature=0.5, top_k=0, top_p=1.0),
    dict(name='T=1.0 (baseline)',            temperature=1.0, top_k=0, top_p=1.0),
    dict(name='T=1.5 (creative)',            temperature=1.5, top_k=0, top_p=1.0),
    dict(name='T=2.0 (very creative)',       temperature=2.0, top_k=0, top_p=1.0),
]
```

Then run:

```bash
python3 experiments/local/sampling_simulator.py
```

## What to Watch For

As you increase temperature from 0.1 → 2.0, observe:

1. **Entropy**: Does it increase? *(It should—higher temp = flatter distribution = more uncertainty)*

2. **Top token probability**: 
   - At T=0.1, 'approve' should get ~95%+
   - At T=2.0, 'approve' might drop to ~30-40%

3. **Number of viable tokens**: 
   - At T=0.1, maybe only 2-3 tokens have non-trivial probability
   - At T=2.0, maybe 7-8 tokens all get sampled regularly

Example output:

```
=== T=0.1 (extreme determinist) ===
Entropy: 0.412
approve       0.967
reject        0.023
review        0.005
escalate      0.002
...

=== T=2.0 (very creative) ===
Entropy: 2.891
optimize      0.152
approve       0.138
notify        0.134
assign        0.130
reject        0.129
...
```

## The Insight

**Temperature is a probability distribution *stretcher*.** 

- Low T: Squeezes the distribution → a few tokens dominate
- High T: Spreads the distribution → many tokens compete equally
- The *order* (which token is most likely) doesn't change, only the *gap between them*

## Math Behind It

Temperature scales the logits before softmax:

$$P(token_i) = \frac{e^{logit_i / T}}{\sum_j e^{logit_j / T}}$$

- Divide by **small T** (e.g., 0.1) → logits get bigger spread → softmax peaks sharply
- Divide by **large T** (e.g., 2.0) → logits get smaller spread → softmax flattens

## Real-World Implication

- **Summarization** (wants consistency): Use T=0.1 to 0.3
- **Brainstorming** (wants variety): Use T=0.8 to 1.5
- **Classification** (wants correctness): Use T=0.0 to 0.2

## Next Step

Once you've run this experiment and seen the pattern, you're ready for:

👉 **[Experiment 2: Top-p (Nucleus Sampling) →](02-top-p.md)**

---

## Optional: Dive Deeper

Want to add visualization? After running the simulator, you can plot entropy vs. temperature:

```python
import matplotlib.pyplot as plt

temps = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
entropies = []

for t in temps:
    out = run_experiment(tokens, logits, temperature=t, n_samples=10000)
    entropies.append(out['entropy'])

plt.plot(temps, entropies, marker='o', label='Entropy')
plt.xlabel('Temperature')
plt.ylabel('Entropy (bits)')
plt.title('How Temperature Affects Distribution Spread')
plt.grid(True)
plt.legend()
plt.show()
```

This visually shows the relationship.

--8<-- "_abbreviations.md"
