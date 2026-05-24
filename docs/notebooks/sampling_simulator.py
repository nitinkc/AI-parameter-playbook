"""
Local Sampling Simulator: Understand LLM Parameters Without API Calls

This simulator demonstrates how decoding parameters (temperature, top-k, top-p, etc.)
affect token selection probabilities. It uses fake logits to isolate parameter behavior.

Usage:
    python sampling_simulator.py

Output:
    Entropy and top-token probabilities for each parameter configuration.
"""

import numpy as np


# ============================================================================
# Core parameter functions
# ============================================================================

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax conversion."""
    x = logits - np.max(logits)
    e = np.exp(x)
    return e / np.sum(e)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by temperature (logits / T)."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def filter_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep only the top-k logits; set others to -inf."""
    if k is None or k <= 0 or k >= logits.size:
        return logits
    idx = np.argpartition(logits, -k)[-k:]
    mask = np.full_like(logits, fill_value=False, dtype=bool)
    mask[idx] = True
    out = logits.copy()
    out[~mask] = -np.inf
    return out


def filter_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filtering: keep smallest set with cumulative prob >= p."""
    if p is None or p >= 1.0:
        return logits
    if p <= 0:
        raise ValueError("top_p must be in (0,1]")

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
    """Min-p filtering: keep tokens with prob >= min_p * max_prob."""
    if min_p is None or min_p <= 0:
        return logits
    if min_p > 1:
        raise ValueError("min_p must be in (0,1]")

    probs = softmax(logits)
    thr = min_p * np.max(probs)
    mask = probs >= thr

    out = logits.copy()
    out[~mask] = -np.inf
    return out


def filter_typical_p(logits: np.ndarray, typical_p: float) -> np.ndarray:
    """Typical-p filtering (keep tokens with typical information content)."""
    if typical_p is None or typical_p >= 1.0:
        return logits
    if typical_p <= 0:
        raise ValueError("typical_p must be in (0,1]")

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


def apply_repetition_penalty(logits: np.ndarray, generated_ids: list, penalty: float) -> np.ndarray:
    """Repetition penalty: downscale logits for already-seen tokens."""
    if penalty is None or penalty == 1.0:
        return logits
    out = logits.copy()
    for tid in set(generated_ids):
        if out[tid] > 0:
            out[tid] /= penalty
        else:
            out[tid] *= penalty
    return out


# ============================================================================
# Sampling and metrics
# ============================================================================

def sample_next(logits: np.ndarray, rng: np.random.Generator) -> int:
    """Sample next token from logits."""
    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))


def entropy(probs: np.ndarray) -> float:
    """Calculate entropy (average surprise) of a probability distribution."""
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
    """Run a parameter sweep experiment."""
    rng = np.random.default_rng(seed)

    counts = np.zeros(len(base_logits), dtype=int)
    generated = []

    for _ in range(n_samples):
        logits = base_logits.copy()

        # Optional: penalize tokens that appear in the recent window
        if sequence_len > 0 and len(generated) > 0:
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
        "counts": counts,
        "probs_emp": probs_emp,
        "entropy": entropy(probs_emp),
        "top_tokens": sorted(
            [(token_names[i], probs_emp[i]) for i in range(len(token_names))],
            key=lambda x: -x[1],
        )[:10],
    }


# ============================================================================
# Main demo
# ============================================================================

if __name__ == "__main__":
    # Define tokens (representing possible next outputs)
    tokens = [
        "approve",
        "reject",
        "review",
        "escalate",
        "delay",
        "audit",
        "optimize",
        "notify",
        "assign",
        "close",
    ]

    # Fake logits: a distribution where 'approve' is favored
    logits = np.array(
        [2.2, 1.8, 1.4, 0.9, 0.2, 0.1, -0.3, -0.6, -0.8, -1.0], dtype=float
    )

    print("=" * 70)
    print("LOCAL SAMPLING SIMULATOR: Understanding LLM Parameters")
    print("=" * 70)
    print("\nBase logits (what a model might output before sampling):")
    for t, l in zip(tokens, logits):
        print(f"  {t:15s}  {l:6.2f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Temperature Sweep")
    print("=" * 70)
    print("\nHow does temperature affect randomness?")

    settings_temp = [
        dict(name="Extremely deterministic (T=0.1)", temperature=0.1, top_k=0, top_p=1.0),
        dict(name="Very deterministic (T=0.2)", temperature=0.2, top_k=0, top_p=1.0),
        dict(name="Deterministic (T=0.5)", temperature=0.5, top_k=0, top_p=1.0),
        dict(name="Baseline (T=1.0)", temperature=1.0, top_k=0, top_p=1.0),
        dict(name="Creative (T=1.5)", temperature=1.5, top_k=0, top_p=1.0),
        dict(name="Very creative (T=2.0)", temperature=2.0, top_k=0, top_p=1.0),
    ]

    for s in settings_temp:
        name = s.pop("name")
        out = run_experiment(tokens, logits, n_samples=10000, seed=42, **s)
        print(f"\n--- {name} ---")
        print(f"Entropy: {out['entropy']:.3f}")
        for t, p in out["top_tokens"][:5]:
            print(f"  {t:15s}  {p:.3f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Top-p (Nucleus) Sweep")
    print("=" * 70)
    print("\nHow does top-p restrict the candidate set?")

    settings_topp = [
        dict(name="Tight nucleus (p=0.2)", temperature=1.0, top_k=0, top_p=0.2),
        dict(name="Medium nucleus (p=0.5)", temperature=1.0, top_k=0, top_p=0.5),
        dict(name="Loose nucleus (p=0.8)", temperature=1.0, top_k=0, top_p=0.8),
        dict(name="Nearly all (p=0.95)", temperature=1.0, top_k=0, top_p=0.95),
        dict(name="No filter (p=1.0)", temperature=1.0, top_k=0, top_p=1.0),
    ]

    for s in settings_topp:
        name = s.pop("name")
        out = run_experiment(tokens, logits, n_samples=10000, seed=42, **s)
        print(f"\n--- {name} ---")
        print(f"Entropy: {out['entropy']:.3f}")
        for t, p in out["top_tokens"][:5]:
            print(f"  {t:15s}  {p:.3f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Top-k Sweep")
    print("=" * 70)
    print("\nHow does top-k limit the candidate set by rank?")

    settings_topk = [
        dict(name="Greedy (k=1)", temperature=1.0, top_k=1, top_p=1.0),
        dict(name="Small set (k=3)", temperature=1.0, top_k=3, top_p=1.0),
        dict(name="Medium set (k=5)", temperature=1.0, top_k=5, top_p=1.0),
        dict(name="Large set (k=10)", temperature=1.0, top_k=10, top_p=1.0),
        dict(name="No limit (k=0)", temperature=1.0, top_k=0, top_p=1.0),
    ]

    for s in settings_topk:
        name = s.pop("name")
        out = run_experiment(tokens, logits, n_samples=10000, seed=42, **s)
        print(f"\n--- {name} ---")
        print(f"Entropy: {out['entropy']:.3f}")
        for t, p in out["top_tokens"][:5]:
            print(f"  {t:15s}  {p:.3f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Combined (Real-World Scenarios)")
    print("=" * 70)
    print("\nHow do multiple parameters interact?")

    settings_combined = [
        dict(
            name="Safety-critical (T=0.3, no filters)",
            temperature=0.3,
            top_k=0,
            top_p=1.0,
        ),
        dict(
            name="Balanced (T=0.7, p=0.95)",
            temperature=0.7,
            top_k=0,
            top_p=0.95,
        ),
        dict(
            name="Creative (T=1.2, p=0.9)",
            temperature=1.2,
            top_k=0,
            top_p=0.9,
        ),
        dict(
            name="Diverse but safe (T=0.9, k=10, p=0.95)",
            temperature=0.9,
            top_k=10,
            top_p=0.95,
        ),
        dict(
            name="Broad diversity (T=0.8, k=40)",
            temperature=0.8,
            top_k=40,
            top_p=1.0,
        ),
    ]

    for s in settings_combined:
        name = s.pop("name")
        out = run_experiment(tokens, logits, n_samples=10000, seed=42, **s)
        print(f"\n--- {name} ---")
        print(f"Entropy: {out['entropy']:.3f}")
        for t, p in out["top_tokens"][:5]:
            print(f"  {t:15s}  {p:.3f}")

    print("\n" + "=" * 70)
    print("TAKEAWAYS")
    print("=" * 70)
    print("""
1. Temperature: Controls smoothness of the distribution
   - Low T (0.1-0.3): Deterministic, peaked
   - High T (1.2+): Flat, diverse

2. Top-p: Selects a dynamic nucleus of tokens
   - Low p (0.2-0.5): Few tokens survive
   - High p (0.9+): Most tokens viable

3. Top-k: Keeps a fixed-size ranked set
   - k=1: Only the most likely token
   - k=10+: Many tokens viable

4. Entropy: Measures randomness
   - E ≈ 0: Deterministic
   - E ≈ 1.0: Moderate
   - E ≈ 2.0+: High diversity

Next steps:
  - Read the Learning Path: docs/learning/00-overview.md
  - Try Experiment 1: docs/learning/01-temperature.md
  - Explore: docs/playgrounds/simulator-guide.md
""")


