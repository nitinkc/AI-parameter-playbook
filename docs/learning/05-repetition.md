# Experiment 5: Repetition Penalties

## The Question

**How do repetition penalties work? What do they actually penalize?**

You've probably encountered it: an LLM gets stuck and repeats the same thing over and over ("the the the..." or "I'm sorry I'm sorry I'm sorry...").

Penalties try to fix this, but:
- Do they prevent repetition entirely?
- Do they just make it *less likely*?
- What counts as "repetition" (exact phrase? single token)?

## What You'll Do

Simulate a multi-token generation scenario where tokens are penalized for appearing recently in the sequence. Watch how penalties change token selection over time.

## The Setup

Instead of sampling once, we'll sample **20 tokens in a row**, applying a repetition penalty to tokens that appeared in the last 8 tokens.

```python
logits = np.array([2.2, 1.8, 1.4, 0.9, 0.2, 0.1, -0.3, -0.6, -0.8, -1.0], dtype=float)
tokens = ['approve', 'reject', 'review', 'escalate', 'delay', 'audit', 'optimize', 'notify', 'assign', 'close']

# Simulate what happens when we generate a 20-token sequence
sequence_len = 8  # Look back this many tokens
```

## Your Experiment: Penalty Impact

Look in the simulator for the extended experiment function and run:

### Scenario 1: No penalty

```python
dict(name='No penalty',
     temperature=0.8, top_k=0, top_p=1.0, 
     repetition_penalty=1.0,  # 1.0 = no effect
     sequence_len=8)
```

### Scenario 2: Light penalty

```python
dict(name='Light penalty (1.1x)',
     temperature=0.8, top_k=0, top_p=1.0, 
     repetition_penalty=1.1,
     sequence_len=8)
```

### Scenario 3: Moderate penalty

```python
dict(name='Moderate penalty (1.3x)',
     temperature=0.8, top_k=0, top_p=1.0, 
     repetition_penalty=1.3,
     sequence_len=8)
```

### Scenario 4: Heavy penalty

```python
dict(name='Heavy penalty (1.5x)',
     temperature=0.8, top_k=0, top_p=1.0, 
     repetition_penalty=1.5,
     sequence_len=8)
```

### Scenario 5: Extreme penalty

```python
dict(name='Extreme penalty (2.0x)',
     temperature=0.8, top_k=0, top_p=1.0, 
     repetition_penalty=2.0,
     sequence_len=8)
```

## How to Run

Modify the simulator to use the sequence-aware experiment. If it's not already there, add this to `experiments/local/sampling_simulator.py`:

```python
def run_sequence_experiment(
    token_names,
    base_logits,
    n_sequences=100,       # How many 20-token sequences to generate
    sequence_length=20,    # Length of each sequence
    seed=123,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    lookback=8,           # How many recent tokens to penalize
):
    rng = np.random.default_rng(seed)
    token_counts = np.zeros(len(base_logits), dtype=int)
    repetition_counts = {}  # Track how often each token repeats

    for _ in range(n_sequences):
        generated = []
        for step in range(sequence_length):
            logits = base_logits.copy()
            
            # Apply repetition penalty to recently-generated tokens
            if len(generated) > 0:
                recent_window = generated[-lookback:]
                for tid in set(recent_window):
                    if logits[tid] > 0:
                        logits[tid] /= repetition_penalty
                    else:
                        logits[tid] *= repetition_penalty
            
            # Apply other filters
            logits = apply_temperature(logits, temperature)
            logits = filter_top_k(logits, top_k)
            logits = filter_top_p(logits, top_p)
            
            tid = sample_next(logits, rng)
            generated.append(tid)
            token_counts[tid] += 1
    
    # Count repetitions (same token chosen twice in a row)
    total_repetitions = 0
    for _ in range(n_sequences):
        generated = []  # Would need to regenerate properly in real code
        # For demo purposes, just return token distribution

    probs_emp = token_counts / token_counts.sum()
    
    return {
        'counts': token_counts,
        'probs_emp': probs_emp,
        'entropy': entropy(probs_emp),
        'top_tokens': sorted(
            [(token_names[i], probs_emp[i]) for i in range(len(token_names))],
            key=lambda x: -x[1]
        )[:10],
    }
```

Run the suite of penalties:

```bash
python3 experiments/local/sampling_simulator.py
```

## What to Watch For

1. **Token diversity**:
   - No penalty: Might generate "approve reject approve approve reject approve..." (repeated)
   - Heavy penalty: More like "approve reject review escalate delay audit optimize..."

2. **Entropy trend**:
   - As you increase penalty, does entropy go up?
   - Why? (Hint: it forces the model to spread tokens more evenly)

3. **Distribution smoothing**:
   - No penalty: High-prob tokens still dominate
   - Heavy penalty: Distribution flattens (entropy increases)

Example hypothetical output:

```
=== No penalty ===
Entropy: 1.234
approve       0.287
reject        0.201
review        0.156
escalate      0.098
delay         0.076
audit         0.053
optimize      0.045
notify         0.032
assign         0.041
close          0.011

=== Heavy penalty (1.5x) ===
Entropy: 2.012
approve       0.156
reject        0.155
review        0.143
escalate      0.128
delay         0.119
audit         0.107
optimize      0.094
notify         0.051
assign         0.028
close          0.019
```

## The Insight: Penalties Don't Prevent, They Discourage

**Key realization:** Repetition penalties don't *forbid* repetition; they *reduce the logit* of already-seen tokens.

This means:
1. ✅ Repetition is still *possible* (the penalty doesn't make it impossible)
2. ✅ More likely to see variety (but might miss intentional repetition like "very very important")
3. ✅ The window matters (tokens outside the lookback aren't penalized)

### How it works mathematically

For each token in the recent window:
- If `logit > 0`: divide by `repetition_penalty` (makes it *less* likely)
- If `logit < 0`: multiply by `repetition_penalty` (makes it *more* likely, i.e., "less negative")

This asymmetry is because:
- High logits: want to temper high-probability tokens
- Low logits: want to avoid double-penalizing low-probability tokens

## Real-World Implication

| Use case | Penalty | Lookback | Reason |
|----------|---------|----------|--------|
| Summarization | 1.0 (none) | N/A | Summaries should focus on key points, even if repeated |
| Code generation | 1.1 (mild) | 32 | Avoid infinite loops but allow intentional repetition like `for i in range(i in range...)` |
| Chatbot | 1.2 (moderate) | 64 | Encourage variety but not too aggressive |
| Story generation | 1.15 (light) | 128 | Balance variety with narrative consistency |

**Note:** Most models allow `frequency_penalty` and `presence_penalty` (OpenAI) or similar in their APIs. These are more sophisticated than the simple "repetition_penalty" here, but the intuition is similar.

## Next Step

You now understand:
- How individual parameters work (temperature, top-p, top- k)
- How they combine
- How penalties discourage repetition

Ready to apply this knowledge to real use cases?

👉 **[Experiment 6: Real Use Cases →](06-use-cases.md)**

---

## Optional: Dive Deeper

Try tracking actual repetitions in a long sequence:

```python
def count_repeats(generated_tokens, lookback=8):
    """Count how many times a token is selected twice in succession."""
    repeats = 0
    for i in range(1, len(generated_tokens)):
        if generated_tokens[i] == generated_tokens[i-1]:
            repeats += 1
    return repeats

# Then compare:
# - No penalty:  ~10 repeats in 1000 tokens
# - Light penalty (1.1x): ~5 repeats
# - Heavy penalty (1.5x): ~1-2 repeats
```

This gives you an intuition for *how much* the penalty actually helps.

--8<-- "_abbreviations.md"
