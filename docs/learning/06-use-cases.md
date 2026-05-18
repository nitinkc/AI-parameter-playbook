# Experiment 6: Real Use Cases

## The Question

**Now that you understand the parameters, how do you actually tune them for real tasks?**

Different tasks have different needs:
- **Classification**: Wants correctness, not creativity
- **Summarization**: Wants consistency and fidelity
- **Code generation**: Wants diversity while avoiding syntax errors
- **Creative writing**: Wants variety and novelty

How do you translate this into parameter choices?

## What You'll Do

Design parameter sets for 4 different enterprise use cases using what you've learned. Then compare them to see the pattern.

## The Setup

You'll use **4 different logit distributions** to simulate what the model "believes" for each task:

### Logit Set 1: Classification (High Confidence)

```python
# Model is very confident in the answer—one class way ahead
logits_class = np.array([4.5, 0.3, 0.1, -0.2, -0.5, -1.0, -1.2, -1.5, -2.0, -2.5], dtype=float)
tokens_class = ['INTENT_APPROVE', 'INTENT_REJECT', 'INTENT_REVIEW', 'TONE_FORMAL', 
                'TONE_CASUAL', 'ACTION_ESCALATE', 'ACTION_DELAY', 'ACTION_AUDIT', 
                'SAFETY_HOLD', 'SAFETY_BLOCK']
```

Why? Classifications often have one clear answer; the model should confidently pick it.

### Logit Set 2: Summarization (Balanced Confidence)

```python
# Model has moderate confidence—several plausible summaries exist
logits_sum = np.array([2.1, 1.9, 1.7, 1.3, 0.8, 0.5, 0.2, -0.2, -0.5, -1.0], dtype=float)
tokens_sum = ['BENEFIT', 'COST', 'TIMELINE', 'STAKEHOLDER', 'RISK', 
              'OPPORTUNITY', 'CONSTRAINT', 'DETAIL', 'TANGENT', 'NOISE']
```

Why? Summarization balances fidelity (stay close to source) with conciseness (be brief).

### Logit Set 3: Code Generation (Uncertain)

```python
# Many plausible next tokens (for loop, if statement, function, class, etc.)
logits_code = np.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], dtype=float)
tokens_code = ['FOR_LOOP', 'IF_STMT', 'FUNC_DEF', 'CLASS_DEF', 'IMPORT', 
               'RETURN', 'ASSIGNMENT', 'CALL', 'COMMENT', 'DECORATOR']
```

Why? Code has many syntactically valid paths; the model shouldn't lock in too early.

### Logit Set 4: Creative (Uncertain, Requires Diversity)

```python
# Many plausible continuations; we want variety
logits_creative = np.array([1.5, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2], dtype=float)
tokens_creative = ['BRAVE', 'QUIRKY', 'DARK', 'HUMOROUS', 'POIGNANT', 
                   'SARCASTIC', 'TENDER', 'WHIMSICAL', 'TRAGIC', 'MYSTICAL']
```

Why? Creative writing benefits from exploration; we want high entropy.

## Your Experiment: Design Parameters for Each Use Case

Using what you learned, design parameter sets for each use case and predict the entropy before running:

### Use Case 1: Classification

**Your hypothesis:**
- Temperature: ?
- Top-p: ?
- Top-k: ?
- Expected entropy: ? (low, high, medium?)

**Rationale:** Classification is safety-critical. We want the model to pick the most likely class confidently.

**Suggested settings:**

```python
dict(name='Classification (low entropy)',
     temperature=0.2, top_k=0, top_p=1.0)
```

### Use Case 2: Summarization

**Your hypothesis:**
- Temperature: ?
- Top-p: ?
- Top-k: ?
- Expected entropy: ? 

**Rationale:** Summarization needs fidelity (accurate) but some variety (multiple valid summaries). Not as strict as classification, not as creative as writing.

**Suggested settings:**

```python
dict(name='Summarization (moderate entropy)',
     temperature=0.6, top_k=0, top_p=0.95)
```

### Use Case 3: Code Generation

**Your hypothesis:**
- Temperature: ?
- Top-p: ?
- Top-k: ?
- Expected entropy: ?

**Rationale:** Code benefits from diversity (many syntactically valid implementations) but can't be too chaotic (weird tokens → bugs).

**Suggested settings:**

```python
dict(name='Code (diverse but bounded)',
     temperature=0.8, top_k=30, top_p=1.0)
```

### Use Case 4: Creative Writing

**Your hypothesis:**
- Temperature: ?
- Top-p: ?
- Top-k: ?
- Expected entropy: ?

**Rationale:** Creative writing rewards novelty. We want high entropy but not complete randomness.

**Suggested settings:**

```python
dict(name='Creative (high entropy)',
     temperature=1.1, top_k=0, top_p=0.85)
```

## How to Run

Create a new experiment script or add to the simulator:

```python
if __name__ == '__main__':
    use_cases = [
        ('Classification', tokens_class, logits_class, [
            dict(name='Classification (low entropy)', temperature=0.2, top_k=0, top_p=1.0),
        ]),
        ('Summarization', tokens_sum, logits_sum, [
            dict(name='Summarization (moderate)', temperature=0.6, top_k=0, top_p=0.95),
        ]),
        ('Code', tokens_code, logits_code, [
            dict(name='Code (diverse)', temperature=0.8, top_k=30, top_p=1.0),
        ]),
        ('Creative', tokens_creative, logits_creative, [
            dict(name='Creative (high)', temperature=1.1, top_k=0, top_p=0.85),
        ]),
    ]

    for use_case_name, tokens, logits, settings in use_cases:
        print(f"\n\n{'='*60}")
        print(f"USE CASE: {use_case_name}")
        print(f"{'='*60}")
        
        for s in settings:
            name = s.pop('name')
            out = run_experiment(tokens, logits, n_samples=10000, seed=42, **s)
            print(f"\n--- {name} ---")
            print(f"Entropy: {out['entropy']:.3f}")
            for t, p in out['top_tokens'][:5]:
                print(f"  {t:20s}  {p:.3f}")
```

## What to Watch For

1. **Entropy progression:**
   - Classification: Should be lowest (~0.5 to 1.0)
   - Summarization: Medium (~1.5 to 2.0)
   - Code: Higher (~2.0 to 2.5)
   - Creative: Highest (~2.0 to 2.8)

2. **Top token dominance:**
   - Classification: Top token should get 80%+ probability
   - Summarization: Top token ~30-40%
   - Code: Top token ~20-30%
   - Creative: More even distribution

3. **How many viable tokens?**
   - Classification: 1-3 tokens get meaningful probability
   - Summarization: 4-6 tokens
   - Code: 7-10 tokens
   - Creative: Nearly all tokens

Example output:

```
============================================================
USE CASE: Classification
============================================================

--- Classification (low entropy) ---
Entropy: 0.712
INTENT_APPROVE       0.891
INTENT_REJECT        0.089
INTENT_REVIEW        0.015
TONE_FORMAL          0.003
TONE_CASUAL          0.001


============================================================
USE CASE: Summarization
============================================================

--- Summarization (moderate) ---
Entropy: 1.623
BENEFIT              0.281
COST                 0.243
TIMELINE             0.198
STAKEHOLDER          0.159
RISK                 0.073


============================================================
USE CASE: Code
============================================================

--- Code (diverse) ---
Entropy: 2.342
FOR_LOOP             0.156
IF_STMT              0.143
FUNC_DEF             0.131
CLASS_DEF            0.119
IMPORT               0.108


============================================================
USE CASE: Creative
============================================================

--- Creative (high) ---
Entropy: 2.511
BRAVE                0.154
QUIRKY               0.142
DARK                 0.138
HUMOROUS             0.133
POIGNANT             0.119
```

## The Insight: Entropy Tells the Story

1. **Low entropy** (0-1.0) = High confidence = Use for extraction, classification
2. **Medium entropy** (1.5-2.0) = Balanced = Use for summarization, grounded QA
3. **High entropy** (2.2+) = Exploratory = Use for creative, ideation

You can **engineer this** by tuning parameters:
- **Lower entropy?** Reduce temperature, add top-p filtering
- **Higher entropy?** Increase temperature, use larger top-k

## Real-World Checklist

When you move to a real model (Azure OpenAI, OpenAI API, local LLM), use this checklist:

- [ ] **Understand your task entropy goal** (is it safety-critical or creative?)
- [ ] **Start with a baseline** (temperature=0.7, top_p=1.0, no penalties)
- [ ] **Measure 1 metric that matters** (accuracy for classification, ROUGE for summarization, etc.)
- [ ] **Sweep parameters systematically** (change temperature first, then top-p)
- [ ] **Log everything** (prompts, outputs, parameters, metrics)
- [ ] **Iterate** (tune toward your goal)

## Next Steps

### Option A: Move to Cloud
You now understand parameters deeply. Ready to test on real models?

👉 **[Azure OpenAI Setup →](../setup/overview.md)**

### Option B: Dive Deeper Locally
Want to explore more parameter combinations or edge cases?

👉 **[Interactive Playgrounds →](../playgrounds/simulator-guide.md)**

### Option C: Learn by Doing
Want to pick a real use case and design a parameter sweep?

👉 **[Parameter Reference →](../parameters/parameter-map.md)**

---

## Optional Challenge: Design Your Own

Think of a task you care about (your own use case). Design a logit distribution that represents model uncertainty on that task. Then:

1. Predict what parameters you'd use
2. Run the simulator with those parameters
3. Check if the entropy makes sense
4. Iterate

This is exactly what ML engineers do in production!

---

## What You've Accomplished

Congratulations! You've completed the 6-experiment learning path. You now understand:

✅ Temperature and how it reshapes distributions  
✅ Top-p (nucleus) and dynamic filtering  
✅ Top-k and ranked filtering  
✅ How parameters combine and interact  
✅ How penalties discourage repetition  
✅ How to tune for different use cases  

**This knowledge transfers to every LLM you encounter.**

Next: Pick your next adventure!

--8<-- "_abbreviations.md"
