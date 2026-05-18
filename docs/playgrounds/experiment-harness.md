# Interactive Playgrounds: Run Experiments Locally

Welcome to the hands-on part. This section contains **ready-to-run Python scripts** you can modify and experiment with locally.

## Overview

We provide:

1. **`experiments/local/sampling_simulator.py`** – Pure Python (NumPy) simulator of the decoding pipeline
2. **`experiment_harness.py`** – Local experiment runner to test parameter sweeps without APIs

Both require **zero cloud credentials** and run completely offline.

## Quick Start

### 1. Sampling Simulator (5 minutes)

The simulator runs the token selection pipeline with fake logits, showing how parameters change probability distributions.

```bash
# One-time setup
pip install -r requirements-experiments.txt

# Run the default experiments
python3 experiments/local/sampling_simulator.py
```

You'll see output like:

```
=== Very deterministic (temp=0.2) ===
Entropy: 1.234
approve       0.851
reject        0.095
...
```

**What it does:**
- Reads pre-defined logits (next-token odds from a fake model)
- Applies temperature, top-k, top-p, penalties
- Samples thousands of times to estimate the distribution
- Shows entropy and probability of each token

**Files:**
- `experiments/local/sampling_simulator.py` – Main script

### 2. Experiment Harness (coming soon)

A local experiment runner that:
- Can load real LLM outputs or continue using simulated distributions
- Runs parameter sweeps (compare T=0.2,0.5,0.8... in one go)
- Logs results to CSV/JSON for analysis

## Modifying the Simulator for Your Experiments

### Add custom logits

Edit `experiments/local/sampling_simulator.py`:

```python
if __name__ == '__main__':
    # Change the tokens and logits to match your use case
    tokens = ['your_token_1', 'your_token_2', ...]
    logits = np.array([1.5, 1.2, 0.8, ...], dtype=float)
    
    settings = [
        dict(name='Your setting 1', temperature=0.5, top_k=0, top_p=1.0),
        dict(name='Your setting 2', temperature=1.0, top_k=10, top_p=0.9),
    ]
```

### Add custom experiments

Add a new settings dict:

```python
settings = [
    # ... existing settings ...
    dict(name='My custom experiment', 
         temperature=0.7, 
         top_k=5, 
         top_p=0.85,
         min_p=0.05,
         typical_p=0.9),
]
```

The simulator will run it and show results.

### Visualize results

After running, plot entropy vs. temperature:

```python
# At the bottom of experiments/local/sampling_simulator.py, after the main loop:

import matplotlib.pyplot as plt

temps = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
entropies = []

for t in temps:
    out = run_experiment(tokens, logits, temperature=t, n_samples=10000)
    entropies.append(out['entropy'])

plt.plot(temps, entropies, marker='o')
plt.xlabel('Temperature')
plt.ylabel('Entropy')
plt.title('Entropy vs Temperature')
plt.grid(True)
plt.show()
```

## Advanced: Custom Penalty Experiments

Most use-case work involves testing penalty strategies. The simulator includes:

- `apply_repetition_penalty()` – Penalizes tokens seen in recent window
- `filter_min_p()` – Filters by relative probability (advanced)
- `filter_typical_p()` – Filters by information content (advanced)

Example: Test different penalty levels:

```python
settings = [
    dict(name='No penalty', repetition_penalty=1.0, sequence_len=0),
    dict(name='Mild (1.1x)', repetition_penalty=1.1, sequence_len=64),
    dict(name='Moderate (1.3x)', repetition_penalty=1.3, sequence_len=64),
    dict(name='Heavy (1.5x)', repetition_penalty=1.5, sequence_len=64),
]
```

## What's Next?

### Follow the Learning Path
The 6-experiment learning path in the previous section gives you a guided progression:
1. **Temperature** (how it stretches distributions)
2. **Top-p** (dynamic filtering)
3. **Top-k** (ranked filtering)
4. **Combined** (all three together)
5. **Penalties** (repetition discouragement)
6. **Use cases** (tuning for real tasks)

Each experiment has a dedicated guide.

### Move to Cloud (Optional)
Once you've built intuition locally, test on real models:
- **Azure OpenAI** → [Setup & run cloud experiments](../setup/overview.md)
- **OpenAI API, local LLMs** → Same parameter concepts

### Explore Advanced Topics
- **Determinism debugging** → Parameter `seed` for reproducibility
- **Structured outputs** → JSON mode and function calling
- **Cost optimization** → `max_tokens` and stopping criteria

---

## Troubleshooting

**Q: Script crashes with "numpy not found"**  
A: Install it: `pip install numpy`

**Q: How do I save results?**  
A: Modify the script to write to CSV:
```python
import csv
with open('results.csv', 'w') as f:
    writer = csv.writer(f)
    for t, p in out['top_tokens']:
        writer.writerow([name, t, p])
```

**Q: Can I use real model logits instead of fake ones?**  
A:  Yes! If you have a real model's logits (usually available via API with `logprobs=True`), pass them directly to `run_experiment()`.

---

👉 **Next: [Back to Learning Path](../learning/overview.md) or [Jump to Experiment 1](../learning/01-temperature.md)**

--8<-- "_abbreviations.md"
