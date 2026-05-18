# Local Sampling Simulator Guide

## What is This?

A **free, offline simulator** that teaches you exactly what LLM sampling parameters do by:

1. Starting with fake logits (pretend next-token odds)
2. Applying parameter transformations (temperature, filters, penalties)
3. Sampling thousands of times to estimate the distribution
4. Showing results: entropy, top tokens, survival rate

**Key idea:** Parameter math is identical in all models. Learn it here, apply it everywhere.

---

## What It Is (and Isn't)

### ✅ What It Covers

- **Temperature scaling** (logits ÷ T)
- **Softmax → probabilities**
- **Top-k truncation** (keep top k ranked tokens)
- **Top-p (nucleus) truncation** (keep smallest set with cumulative prob ≥ p)
- **Min-p filtering** (keep prob ≥ min_p × max_prob)
- **Typical-p filtering** (keep tokens with "typical" information content)
- **Repetition penalties** (discourage tokens seen recently)

### ❌ What It Doesn't Do

- Run an actual transformer model
- Produce realistic text (uses fake logits)
- Account for vocabulary size  (uses 10 tokens for simplicity)

**Think of it as:** "If a model believed these were the next-token odds, how would parameter X change behavior?"

---

## The Decoding Pipeline (Visual)

This is the standard token selection process. Any LLM uses something like this:

```
Raw logits (e.g., [2.2, 1.8, 1.4, ...])
        ↓
[Optional: Repetition penalty applied to recent tokens]
        ↓
[Temperature scaling: logits / T]
        ↓
[Filtering: top-k, then top-p, then min-p, then typical-p]
        ↓
[Softmax: convert to probabilities]
        ↓
[Sampling: pick a token randomly according to probabilities]
        ↓
[Next token selected]
```

Most systems apply filters in roughly this order. The order matters!

---

## Key Concepts (Intuition + Math)

### Temperature

Rescales logits before softmax:

$$P(token_i) = \frac{e^{logit_i / T}}{\sum_j e^{logit_j / T}}$$

- **Lower T (< 1)** → sharpens distribution → deterministic
- **Higher T (> 1)** → flattens distribution → random

### Top-k

Keeps only the **k highest-ranked tokens** (by logit), discards rest.

- Fixed-size candidate set
- Always removes lowest-probability tail
- Good for preventing garbage tokens

### Top-p (Nucleus Sampling)

Keeps the **smallest set of tokens** whose cumulative probability ≥ p.

- Dynamic-size candidate set (adapts to model confidence)
- Balances exploration and coherence
- Common default: p ≈ 0.9 or 0.95

### Min-p

Keeps tokens with probability ≥ min_p × max_prob_at_this_step.

- Adapts like top-p but using a relative threshold
- Useful in open-source backends

### Typical-p

Selects tokens with **typical information content** (close to distribution entropy).

- Different from max-probability focus
- Less common but sometimes better for certain tasks

### Repetition Penalty

Scales down logits of tokens seen recently (within a window).

- Discourages (not forbids) repetition
- Window size matters (look back 8, 32, 64, 128 tokens?)
- Penalty strength: 1.0=none, 1.1=mild, 1.5=heavy

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- `numpy` (required)
- `matplotlib` (optional, for plotting)

### Install

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-experiments.txt
```

### Run the Default Experiments

```bash
python3 experiments/local/sampling_simulator.py
```

You'll see output like:

```
=== Very deterministic (T=0.2) ===
Entropy: 0.412
approve       0.967
reject        0.023
review        0.005
escalate      0.002
delay         0.000
...
```

---

## The Simulator Code

Below is the complete, ready-to-run simulator at `experiments/local/sampling_simulator.py`:

### Core Math Functions

```python
import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    '''Numerically stable softmax.'''
    x = logits - np.max(logits)
    e = np.exp(x)
    return e / np.sum(e)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    '''Scale logits by temperature (logits / T).'''
    if temperature <= 0:
        raise ValueError('temperature must be > 0')
    return logits / temperature


def filter_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    '''Keep only the top-k logits; set others to -inf.'''
    if k is None or k <= 0 or k >= logits.size:
        return logits
    idx = np.argpartition(logits, -k)[-k:]
    mask = np.full_like(logits, fill_value=False, dtype=bool)
    mask[idx] = True
    out = logits.copy()
    out[~mask] = -np.inf
    return out


def filter_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    '''Nucleus filtering: keep smallest set with cumulative prob >= p.'''
    if p is None or p >= 1.0:
        return logits
    if p <= 0:
        raise ValueError('top_p must be in (0,1]')

    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])

    keep = order[cum <= p]
    if keep.size == 0:
        keep = order[:1]

    mask = np.zeros_like(probs, dtype=bool)
    mask[keep] = True
    out = logits.copy()
    out[~mask] = -np.inf
    return out


def filter_min_p(logits: np.ndarray, min_p: float) -> np.ndarray:
    '''Min-p filtering: keep tokens with prob >= min_p * max_prob.'''
    if min_p is None or min_p <= 0:
        return logits
    if min_p > 1:
        raise ValueError('min_p must be in (0,1]')

    probs = softmax(logits)
    thr = min_p * np.max(probs)
    mask = probs >= thr

    out = logits.copy()
    out[~mask] = -np.inf
    return out


def filter_typical_p(logits: np.ndarray, typical_p: float) -> np.ndarray:
    '''Typical-p filtering.
    
    Keeps tokens with information content close to distribution entropy.
    '''
    if typical_p is None or typical_p >= 1.0:
        return logits
    if typical_p <= 0:
        raise ValueError('typical_p must be in (0,1]')

    probs = softmax(logits)
    surprise = -np.log(np.clip(probs, 1e-30, 1.0))
    H = float(np.sum(probs * surprise))

    order = np.argsort(np.abs(surprise - H))
    cum = np.cumsum(probs[order])

    keep = order[cum <= typical_p]
    if keep.size == 0:
        keep = order[:1]

    mask = np.zeros_like(probs, dtype=bool)
    mask[keep] = True

    out = logits.copy()
    out[~mask] = -np.inf
    return out


def apply_repetition_penalty(logits: np.ndarray, generated_ids: list[int], penalty: float) -> np.ndarray:
    '''Simple repetition penalty: downscale logits for already-seen tokens.'''
    if penalty is None or penalty == 1.0:
        return logits
    out = logits.copy()
    for tid in set(generated_ids):
        if out[tid] > 0:
            out[tid] /= penalty
        else:
            out[tid] *= penalty
    return out
```

### Sampling & Metrics

```python
def sample_next(logits: np.ndarray, rng: np.random.Generator) -> int:
    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))


def entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-30, 1.0)
    return float(-np.sum(p * np.log(p)))


def run_experiment(
    token_names,
    base_logits,
    n_samples=5000,
    seed=123,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    min_p=None,
    typical_p=1.0,
    repetition_penalty=1.0,
    sequence_len=0,
):
    '''Run parameter sweep experiment.'''
    rng = np.random.default_rng(seed)

    counts = np.zeros(len(base_logits), dtype=int)
    generated = []

    for _ in range(n_samples):
        logits = base_logits.copy()

        # Optional: penalize tokens that appear in the recent window
        if sequence_len > 0:
            logits = apply_repetition_penalty(logits, generated[-sequence_len:], repetition_penalty)

        # Temperature scaling
        logits = apply_temperature(logits, temperature)

        # Filters (order matters)
        logits = filter_top_k(logits, top_k)
        logits = filter_top_p(logits, top_p)
        logits = filter_min_p(logits, min_p)
        logits = filter_typical_p(logits, typical_p)

        tid = sample_next(logits, rng)
        counts[tid] += 1
        generated.append(tid)

    probs_emp = counts / counts.sum()

    return {
        'counts': counts,
        'probs_emp': probs_emp,
        'entropy': entropy(probs_emp),
        'top_tokens': sorted(
            [(token_names[i], probs_emp[i]) for i in range(len(token_names))],
            key=lambda x: -x[1]
        )[:10],
    }
```

### Main Demo

```python
if __name__ == '__main__':
    tokens = ['approve', 'reject', 'review', 'escalate', 'delay', 'audit', 'optimize', 'notify', 'assign', 'close']

    # Example base logits (hand-made): first few tokens more likely
    logits = np.array([2.2, 1.8, 1.4, 0.9, 0.2, 0.1, -0.3, -0.6, -0.8, -1.0], dtype=float)

    settings = [
        dict(name='Very deterministic (T=0.2)', temperature=0.2, top_k=0, top_p=1.0),
        dict(name='Balanced (T=0.8)', temperature=0.8, top_k=0, top_p=0.95),
        dict(name='More creative (T=1.3)', temperature=1.3, top_k=50, top_p=1.0),
        dict(name='Tighter nucleus (T=1.0, p=0.6)', temperature=1.0, top_k=0, top_p=0.6),
        dict(name='Min-p filter (p=0.1)', temperature=1.0, top_k=0, top_p=1.0, min_p=0.1),
        dict(name='Typical-p (p=0.2)', temperature=1.0, top_k=0, top_p=1.0, typical_p=0.2),
    ]

    for s in settings:
        name = s.pop('name')
        out = run_experiment(tokens, logits, n_samples=10000, seed=42, **s)
        print(f"\n=== {name} ===")
        print('Entropy:', round(out['entropy'], 3))
        for t, p in out['top_tokens']:
            print(f"{t:10s}  {p:.3f}")
```

---

## Interpreting Results

### Entropy

**What is it?** Average "surprise" in the distribution. Higher = more random, lower = more deterministic.

- **Entropy ≈ 0** → Nearly deterministic (T=0.1, or top-k=1)
- **Entropy ≈ 1.0** → Fairly peaked (safety-critical tasks)
- **Entropy ≈ 2.0+** → Flat (creative tasks, code generation)

**How to read it:**
- Low entropy = model "confident" in a few top options
- High entropy = model "uncertain," many options equally plausible

### Top Token Probability

**What is it?** How likely is the most probable token?

- Top token prob ~90%+ → Highly deterministic (classification)
- Top token prob ~30-50% → Balanced (summarization)
- Top token prob ~15-25% → Creative (writing, brainstorming)

### Surviving Token Count

**What is it?** How many tokens have non-zero probability after all filters?

- With top-k=1: Only 1 token
- With top-k=5: Only 5 tokens
- With top-p=0.5: ~2-4 tokens (varies by confidence)
- With no filters + high temp: Most/all tokens

---

## Recommended Experiment Sequence

### Experiment 1: Temperature Sweep

Fix `top_p=1.0`, `top_k=0`, vary temperature:

```python
settings = [
    dict(name='T=0.1', temperature=0.1, top_k=0, top_p=1.0),
    dict(name='T=0.2', temperature=0.2, top_k=0, top_p=1.0),
    dict(name='T=0.5', temperature=0.5, top_k=0, top_p=1.0),
    dict(name='T=1.0', temperature=1.0, top_k=0, top_p=1.0),
    dict(name='T=1.5', temperature=1.5, top_k=0, top_p=1.0),
    dict(name='T=2.0', temperature=2.0, top_k=0, top_p=1.0),
]
```

**Observe:** Entropy increases linearly. Top token probability decreases.

### Experiment 2: Top-p Sweep

Fix `temperature=1.0`, `top_k=0`, vary `top_p`:

```python
settings = [
    dict(name='p=0.2', temperature=1.0, top_k=0, top_p=0.2),
    dict(name='p=0.5', temperature=1.0, top_k=0, top_p=0.5),
    dict(name='p=0.8', temperature=1.0, top_k=0, top_p=0.8),
    dict(name='p=0.95', temperature=1.0, top_k=0, top_p=0.95),
    dict(name='p=1.0', temperature=1.0, top_k=0, top_p=1.0),
]
```

**Observe:** Lower p = fewer tokens survive = lower entropy.

### Experiment 3: Top-k Sweep

Fix `temperature=1.0`, `top_p=1.0`, vary `top_k`:

```python
settings = [
    dict(name='k=1', temperature=1.0, top_k=1, top_p=1.0),
    dict(name='k=5', temperature=1.0, top_k=5, top_p=1.0),
    dict(name='k=20', temperature=1.0, top_k=20, top_p=1.0),
    dict(name='k=0 (no limit)', temperature=1.0, top_k=0, top_p=1.0),
]
```

**Observe:** k=1 → entropy=0 (only 1 token). Higher k → higher entropy.

---

## Real-World Parameter Presets

| Task | Temperature | Top-p | Top-k | Notes |
|------|-------------|-------|-------|-------|
| Classification | 0.0-0.2 | 1.0 | 0 | Deterministic |
| Extraction | 0.1-0.3 | 1.0 | 0 | Safe, consistent |
| Summarization | 0.6-0.8 | 0.95 | 0 | Balance fidelity + variety |
| RAG/Q&A | 0.2-0.4 | 1.0 | 0 | Grounded, low hallucination |
| Code generation | 0.8 | 1.0 | 30 | Diverse but bounded |
| Brainstorming | 0.9-1.2 | 0.85 | 0 | Creative |
| Story writing | 1.0-1.3 | 0.9 | 0 | High creativity |

---

## Optional: Visualization

Plot entropy vs. temperature:

```python
import matplotlib.pyplot as plt

temps = np.linspace(0.1, 2.0, 20)
entropies = []

for t in temps:
    out = run_experiment(tokens, logits, temperature=t, n_samples=5000)
    entropies.append(out['entropy'])

plt.figure(figsize=(10, 6))
plt.plot(temps, entropies, marker='o', linewidth=2)
plt.xlabel('Temperature', fontsize=12)
plt.ylabel('Entropy (bits)', fontsize=12)
plt.title('How Temperature Affects Distribution Spread', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Next Steps

- **Follow the guided learning path:** Start with [Experiment 1: Temperature](../learning/01-temperature.md)
- **Read parameter documentation:** [Parameter Reference](../parameters/parameter-map.md)
- **Move to cloud:** [Azure OpenAI Setup](../setup/overview.md) (when ready)

---

## FAQ

**Q: Can I modify the simulator to use my own logits?**  
A: Yes! Replace `logits = np.array([...])` with your own values.

**Q: Will this teach me how a real model works?**  
A: No—this teaches you parameter *math*, which is universal. But real models have vocabulary of 50k+ tokens and produce coherent text. The parameter principles are identical.

**Q: Which parameters matter most?**  
A: In order: (1) Temperature, (2) Top-p, (3) Penalties. Top-k is less common in modern systems.

**Q: Can I use this with other APIs (OpenAI, Anthropic)?**  
A: The parameter math is the same everywhere. Learn it here, apply it there.

---

👉 **Ready to start?** [Go to Experiment 1: Temperature →](../learning/01-temperature.md)

--8<-- "_abbreviations.md"
