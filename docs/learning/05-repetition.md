# Experiment 5: Repetition Penalties — A Step-by-Step Learning Guide

> **What this file is**: A beginner-friendly, ground-up explanation of Experiment 5 (Repetition Penalties).  
> **Who it's for**: Anyone who completed Experiments 1–4 and wants to understand repetition penalties — including the math, the failure modes, and why they're nothing like top-k or top-p.  
> **What you'll get**: The ability to diagnose repetition problems in real LLM output, choose the right penalty strength and lookback window, and understand the difference between multiplicative, presence, and frequency penalties.

---

## Before You Start: A New Kind of Problem

The first four experiments all dealt with a single-step sampling question: given a probability distribution, how do we pick one token? Temperature, top-k, and top-p are all about *shaping that distribution at a single moment*.

Repetition penalty is different. It's the first parameter in this series that cares about **history** — what tokens the model has already generated. It answers a different question:

> Not "how do I pick a token?" but "how do I avoid picking tokens I've already picked too many times?"

This makes it fundamentally a **sequence-level** control, not a token-level one. You need to understand this shift in framing before the mechanics will make sense.

---

## The Core Intuition: Pressure, Not Prohibition

Before touching any math, understand the key word in the experiment description:

> "Repetition penalties **discourage**, not forbid."

This is the most important distinction. Top-k and top-p can create hard zeros — tokens that literally cannot be sampled. Repetition penalty never creates a hard zero (unless the penalty is infinite, which no system uses). It only *moves* the probability of seen tokens downward — and the other unseen tokens absorb the redistributed probability.

Think of it like a tax:
- No tax (penalty=1.0): every token pays the same rate, seen tokens have no disadvantage
- Light tax (penalty=1.1): seen tokens pay a 10% surcharge, slight disadvantage
- Heavy tax (penalty=2.0): seen tokens pay a 100% surcharge, significant disadvantage
- Prohibition (penalty=∞): seen tokens are banned entirely — but no real system does this

The model can still generate "approve" after it has already generated "approve." It just becomes less likely to do so. How much less likely depends on the penalty value and the original probability of the token.

---

## Step 1: Understand the Multiplicative Penalty Formula

The formula from the experiment:

```python
if token in recent_tokens:
    if logit > 0:
        logit /= penalty
    else:
        logit *= penalty
```

This needs careful unpacking. Let's go through each part.

### Part 1: "if token in recent_tokens"

The penalty only applies to tokens that have appeared in a lookback window — typically the last N tokens of the generated sequence. Tokens outside the window are treated normally.

If `lookback = 8`, only the last 8 generated tokens are tracked. If "approve" appeared at position 3 but you're now generating position 12, "approve" is no longer penalized (it left the lookback window).

This is a design choice: tokens you used *very recently* are penalized; tokens you used *long ago* are not. The window size controls "how long is the model's anti-repetition memory."

### Part 2: "if logit > 0: logit /= penalty"

For positive logits — tokens the model already likes — dividing by the penalty makes the logit smaller, which softmax then translates to a lower probability.

Example: `logit = 2.2`, `penalty = 1.3`
```
Penalized logit = 2.2 / 1.3 = 1.692
```

The logit dropped from 2.2 to 1.692. Not to zero — just lower.

### Part 3: "else: logit *= penalty"

For negative logits — tokens the model already disfavors — multiplying by the penalty makes the logit *more negative*, pushing probability even lower.

Example: `logit = -0.6`, `penalty = 1.3`
```
Penalized logit = -0.6 * 1.3 = -0.78
```

The logit dropped from -0.6 to -0.78. More negative = even less likely after softmax.

### Why This Asymmetry?

Without the positive/negative split, you'd apply the same operation to all logits. But dividing a *negative* logit by a penalty > 1 would make it *less negative* (closer to zero) — which would *increase* its probability. That's the opposite of what you want.

Example of what would go wrong without the split:
```
logit = -0.6, penalty = 1.3
Wrong operation: -0.6 / 1.3 = -0.46  ← less negative, HIGHER probability!
Correct operation: -0.6 * 1.3 = -0.78  ← more negative, lower probability
```

The asymmetry ensures that the penalty *always discourages* seen tokens, regardless of the sign of their logit.

---

## Step 2: Trace the Penalty for One Token at Three Penalty Strengths

Let's track what happens to "approve" (logit=2.2) when it has already appeared in the lookback window.

### Setup

Original distribution at T=0.8 (as in the experiment settings):
```
Scaled logits (÷0.8):    approve: 2.75, reject: 2.25, review: 1.75, ...
Probabilities via softmax:
  approve:   ~0.44  (44%)
  reject:    ~0.26  (26%)
  review:    ~0.15  (15%)
  escalate:  ~0.07  (7%)
  delay:     ~0.03  (3%)
  ...
```

### Penalty = 1.0 (No penalty)

No change. Approve stays at logit 2.75 (post-temperature). Probability ~44%.

### Penalty = 1.3 (Moderate)

Approve's logit after temperature = 2.75 (positive, so divide):
```
2.75 / 1.3 = 2.115
```

New distribution after penalty applied to approve:
- approve: logit 2.115 → lower probability
- all other tokens: unchanged logits

After softmax with approve's logit reduced:
```
approve probability drops from ~44% to ~30%
All other tokens' probabilities increase proportionally
```

The "missing" probability from approve (about 14 percentage points) gets redistributed across all other tokens. The tokens near approve in rank (reject, review) benefit most in absolute terms, but every surviving token benefits somewhat.

### Penalty = 2.0 (Extreme)

Approve's logit after temperature = 2.75:
```
2.75 / 2.0 = 1.375
```

Approve's logit has nearly halved. After softmax:
```
approve probability drops from ~44% to roughly ~15-18%
reject, review, and others all increase significantly
```

At penalty=2.0, approve is still the most likely token, but it's no longer dominant. The model now genuinely considers reject, review, escalate as real alternatives.

---

## Step 3: Understand the Lookback Window

The lookback window is the set of recent tokens that are subject to the penalty. This is a parameter that often gets ignored, but it's as important as the penalty strength.

### Visualizing the Window

Suppose `lookback = 8` and the model has generated this sequence:

```
Position:    1       2        3        4        5        6        7        8        9
Token:    approve  reject  approve   review  approve  review  approve  reject   [next]
```

At position 9, the lookback window covers positions 1–8. Tokens in the window:

```
In window: approve (appears at pos 1, 3, 5, 7 → 4 times), reject (pos 2, 8 → 2 times), review (pos 4, 6 → 2 times)
Not in window: escalate, delay, audit, optimize, notify, assign, close
```

The penalty applies to approve, reject, and review. The other 7 tokens are unaffected — they have their normal probabilities.

### What Happens When Tokens Leave the Window

As generation continues, older tokens drop out of the window. At position 10:
- Window covers positions 2–9
- Position 1 (approve) is now outside the window
- Approve is still penalized because it also appears at positions 3, 5, 7, and 9

But at position 17 (if approve hasn't appeared since position 9):
- Window covers positions 9–16
- Approve's last appearance was at position 9, still barely in the window

At position 18:
- Window covers positions 10–17
- Approve's last appearance (position 9) is now outside the window
- Approve is no longer penalized — it gets its full original probability back

### The Lookback Window Controls "Forgetting"

A short window (e.g., 8): the model forgets recent tokens quickly. Good for preventing immediate adjacent repetitions ("the the the") but won't help with phrases that repeat every 10 tokens.

A long window (e.g., 128): the model remembers a much longer history. Better for preventing paragraph-level repetitions but at the risk of suppressing legitimate reuse of important words (technical terms, names that genuinely need to appear repeatedly).

---

## Step 4: The Five Penalty Strengths — What Each Produces

### Penalty = 1.0 (No penalty)

Distribution is entirely shaped by temperature (T=0.8 in this experiment). No anti-repetition pressure.

Behavior you'll see: The model's natural tendencies dominate. If "approve" has logit 2.2, it will appear frequently. In a short sequence of 8 tokens, you'll likely see "approve" 2–3 times.

Entropy ≈ 1.2–1.5 (just T=0.8 effect).

When "no penalty" is actually fine:
- Summarization: technical terms legitimately recur ("the protein binds... the binding process... the protein's binding affinity")
- List generation: list structure words ("first", "second", "third") appear repeatedly by design
- Code generation: syntax keywords ("return", "for", "if") are supposed to recur

---

### Penalty = 1.1 (Light, 10% surcharge)

Mild discouragement. If "approve" appeared once and had 44% probability, it now might have ~40%.

Behavior: Nearly identical to no penalty for most generations. You'd need to run hundreds of samples to statistically detect the difference. Helps slightly with the most extreme cases where the dominant token has 80%+ probability.

Entropy: Barely higher than no penalty.

When to use: As a conservative default "just in case" measure. Won't noticeably change output but provides a small safety margin against the most pathological loops.

---

### Penalty = 1.3 (Moderate)

Meaningful discouragement. "Approve" with logit 2.2 (positive) gets divided: 2.2/1.3 = 1.69. At T=0.8: 2.75 → 2.115.

Behavior: Noticeably more variety. You'll see approve less, but it still wins often. The second and third tokens (reject, review) get picked more than they would without penalty.

Entropy: Meaningfully higher. In the experiment output, expect entropy to jump noticeably compared to no penalty.

When to use: Standard setting for chatbots and dialogue systems. Reduces repetitive phrasing without degrading coherence.

---

### Penalty = 1.5 (Heavy)

Significant discouragement. "Approve" at 2.75 (post T) → 2.75/1.5 = 1.833.

Behavior: Approve still wins sometimes, but the race between approve, reject, and review is now genuinely competitive. You'll see much more variety in short sequences. The "winner" of each generation step changes more frequently.

Entropy: High. The experiment's example output shows approve at ~15.6% — it went from being dominant (44%) to being just one of many roughly-equal options.

When to use: Story generation, creative writing where you want linguistic variety. Also useful if you've observed looping in production and need to break it.

---

### Penalty = 2.0 (Extreme)

Very strong discouragement. "Approve" at 2.75 → 2.75/2.0 = 1.375.

Behavior: Approve's probability has roughly halved. The model now actively seeks alternatives. You'll see a much wider range of tokens in the output. But here's the risk: even *necessary* repetitions get penalized. If the correct output truly requires saying "approve" multiple times (e.g., "The system should approve all, and if approved, report as approved"), the model will try to rephrase awkwardly.

Entropy: Very high, often higher than the "unconstrained" entropy you'd see from temperature alone. The penalty can temporarily make the distribution *more uniform* than even a high temperature would produce.

When to use: Rarely in production. Useful for stress testing, or when you've verified that your specific task has zero tolerance for any repetition (e.g., generating a list of strictly unique items, where each item must be different from all previous ones).

---

## Step 5: Frequency vs Presence Penalty — The Two API Flavors

Many APIs (including OpenAI's) expose two different penalty types. They sound similar but have meaningfully different behavior.

### Presence Penalty

```
Rule: If a token has appeared AT ALL in the lookback window,
      apply a fixed logit deduction.

Effect: binary — either penalized (appeared) or not (hasn't appeared)

Formula (additive version):
  if token in seen_tokens:
      logit -= presence_penalty_value
```

**What it produces**: Equal pressure on all seen tokens, regardless of how many times they've appeared. Token appearing once = same penalty as token appearing ten times. The penalty is about *existence* in the window, not *frequency*.

**When it matters**: When you want to discourage *any* repetition of words you've already used, but you don't care whether a word appeared once or many times. Good for diversity of vocabulary.

### Frequency Penalty

```
Rule: The more times a token has appeared, the larger its penalty.

Effect: proportional to count — tokens that repeat many times get penalized more

Formula (additive version):
  if token in seen_tokens:
      logit -= frequency_penalty_value * count_of_appearances
```

**What it produces**: Escalating pressure. A token appearing once gets a small penalty. The same token appearing five times gets 5x the penalty. The model is progressively discouraged from continuing a repetition pattern once it starts.

**When it matters**: When you want to allow occasional reuse of common words but strongly prevent runaway repetition loops. Good for breaking "the the the the" patterns because the penalty compounds with each additional repetition.

### Side-by-Side Comparison

Imagine the lookback window contains: [approve, reject, approve, approve, review, approve]

"Approve" has appeared 4 times. "Reject" has appeared 1 time. "Review" has appeared 1 time.

**With presence penalty = 0.5 (additive):**
```
approve: logit -= 0.5  (appeared, flat deduction)
reject:  logit -= 0.5  (appeared, same deduction despite fewer occurrences)
review:  logit -= 0.5  (appeared, same deduction)
Others:  no change
```
All seen tokens penalized equally. Approve gets no extra penalty for appearing 4 times.

**With frequency penalty = 0.5 (additive):**
```
approve: logit -= 0.5 * 4 = 2.0  (4 appearances, heavy penalty)
reject:  logit -= 0.5 * 1 = 0.5  (1 appearance, light penalty)
review:  logit -= 0.5 * 1 = 0.5  (1 appearance, light penalty)
Others:  no change
```
Approve gets 4x more penalty than reject. The more it looped, the harder it gets penalized.

### Which to Use When

| Situation | Use Presence Penalty | Use Frequency Penalty |
|-----------|---------------------|----------------------|
| Generate diverse vocabulary | ✓ | |
| Break runaway loops | | ✓ |
| Even treatment of all seen tokens | ✓ | |
| Escalating pressure on repeat offenders | | ✓ |
| Output is a list (each item must be unique) | ✓ | |
| Dialogue (some words legitimately repeat) | | ✓ |

---

## Step 6: Read the Graphs — Panel by Panel

The graph `exp5_repetition.png` has 9 panels across 3 rows. Here is how to read each one.

---

### Row 1, Panels 1–3: "Before and After Penalty" bar charts

Three bar charts showing the distribution before and after applying the penalty to "approve" and another token, for three penalty strengths (1.0, 1.3, 2.0).

**What you're looking at:**
- X-axis: token index (0–14)
- Y-axis: probability
- Dark/grey bars: original distribution (before penalty)
- Colored bars: distribution after penalty applied
- Red-shaded vertical bands at the penalized token positions
- Annotation "penalized" with an arrow pointing to the penalized token's post-penalty bar

**How to read them:**

Panel 1 (penalty=1.0): The two sets of bars are identical. No change. This is your visual baseline.

Panel 2 (penalty=1.3): The penalized tokens' bars (under the red shading) are shorter in the colored vs grey version. All other tokens' bars are slightly taller in the colored version. The redistribution is visible if you look carefully — the gain is spread across many tokens, so each gains a little.

Panel 3 (penalty=2.0): The penalized tokens' bars have dropped substantially. The surrounding tokens have noticeably taller bars in the colored version. The redistribution is clearly visible.

**What to look for:**

1. How much did the penalized token's bar shrink? That's the direct effect.
2. Did the token maintain its rank? (It should still be the tallest bar overall at penalty=1.3 and probably at 2.0 too — just shorter.)
3. Which tokens benefited most from the redistribution? They're the ones that grew the most.

---

### Row 2, Left + Middle: "Penalized Token Probability vs Penalty Strength"

A line chart showing how the probability of each penalized token changes as penalty increases from 1.0 to 3.0.

**What you're looking at:**
- X-axis: penalty value (1.0 to 3.0)
- Y-axis: probability of the penalized token after applying that penalty
- Pink line: approve (high-ranked token, positive logit)
- Yellow line: audit or another token (lower-ranked token)
- Horizontal dashed lines: the original probabilities (before any penalty)
- Vertical dotted line at penalty=1.0: the baseline (no penalty)

**How to read it:**

Both lines start at the left (penalty=1.0) at their original probability values. As you move right (higher penalty), both lines drop.

The pink line (approve, high probability) drops faster in absolute terms — it has more probability to lose because it starts higher. But it takes a large penalty to make it fall below the second-ranked token.

The yellow line (lower-probability token) drops more slowly in absolute terms but may drop faster as a fraction of its original value — because its logit is smaller, dividing by the penalty has a proportionally larger effect on its already-compressed logit.

**The "crossover point"**: If you extend the lines far enough, you might see them cross — meaning the originally-higher token gets penalized below the originally-lower token. This crossover indicates the penalty has become strong enough to invert the natural ranking. This is usually the "too strong" zone.

**The flattening zone**: Both lines eventually flatten and stop dropping much even as penalty increases further. This is because at very high penalties, the penalized tokens' logits become so negative that they're already near-zero probability — further penalty has diminishing returns.

---

### Row 2, Right: "Logit Reduction for Penalized Token"

A line chart showing the logit change (raw logit shift, not probability shift) for the penalized token.

**What you're looking at:**
- X-axis: penalty value
- Y-axis: change in logit value (penalty=1.0 is the reference; values are negative because the logit is being reduced)
- Purple line with shaded area below
- Horizontal dashed line at zero: the no-penalty baseline

**How to read it:**

The line starts at zero (penalty=1.0, no change) and drops steadily as penalty increases. The drop is not linear — it's faster at low penalty values (going from 1.0 to 1.5 drops the logit more than going from 2.5 to 3.0).

**Why the logit change matters more than the probability change:**

Logit changes are what the model "feels" internally. A logit drop of 1.0 always means the same thing in terms of relative preference, regardless of where the logit started. A probability drop from 40% to 30% means something different if the token started at 41% vs if it started at 80%.

The logit reduction chart shows the raw mechanism. The probability chart (row 2, left) shows the consequence. Both together give you the complete picture.

---

### Row 3, Panels 1–2: "Presence Penalty" and "Frequency Penalty" comparison charts

Two grouped bar charts, one for each penalty type, showing how three tokens are affected at three penalty strengths (0.5, 1.0, 1.5).

**What you're looking at:**
- X-axis: three tokens, labeled with how many times each appeared in the window (e.g., "Token 0 (seen 1x)", "Token 1 (seen 3x)", "Token 2 (seen 5x)")
- Y-axis: probability after applying the penalty
- Three bars per token: one for each penalty strength (lighter shade = weaker penalty, darker = stronger)

**How to read the Presence Penalty panel:**

All three tokens get the same penalty deduction regardless of how many times they appeared. The bars for "seen 1x", "seen 3x", and "seen 5x" tokens all drop by the same amount for a given penalty strength.

At penalty=1.5 (darkest bars): all three tokens' bars are at roughly the same height relative to their starting values. Equal treatment.

**How to read the Frequency Penalty panel:**

The token appearing 5x gets the largest penalty (longest downward shift). The token appearing 1x gets the smallest penalty. The bars show staggered heights — "seen 5x" is shortest, "seen 1x" is tallest, at any given penalty strength.

At penalty=1.5: the token seen 5x has a dramatically lower bar than the token seen 1x. The penalty has compounded.

**The visual comparison:**

The key visual difference between the two panels: in the Presence panel, bars within each group are nearly the same height (equal treatment). In the Frequency panel, bars within each group form a clear staircase pattern (escalating treatment). That staircase pattern is the "frequency" in frequency penalty.

---

### Row 3, Right: "Repetition Penalty Guide" table

A reference table showing penalty values and their effects.

**How to read it:**

Each row is a penalty value range. The effect column describes what you'd observe in practice. The last row (>1.8) is highlighted in red to signal that this is the danger zone.

Use this as a quick lookup when calibrating penalty in the field.

---

## Step 7: How to Read the Sample Output Text

```
=== No penalty ===
Entropy: 1.234
approve       0.287
reject        0.201
review        0.143
escalate      0.099
delay         0.067
...
```

**Reading guide:**

| What you see | What it means |
|---|---|
| `Entropy: 1.234` | Moderate — a few tokens dominate but others are viable |
| `approve 0.287` | 28.7% — highest probability token at T=0.8 |
| Regular decline down the list | Natural distribution shape from temperature |
| No hard zeros | No top-k or top-p filtering in this setting |

---

```
=== Heavy penalty (1.5x) ===
Entropy: 2.012
approve       0.156
reject        0.155
review        0.143
escalate      0.134
delay         0.121
audit         0.109
...
```

**Reading guide:**

| What you see | What it means |
|---|---|
| `Entropy: 2.012` | Much higher — distribution is now much flatter |
| `approve 0.156` | Down from 28.7% — penalty significantly reduced it |
| `reject 0.155` | Nearly tied with approve — tokens are close to equal |
| `review 0.143` | Still declining but slowly — near-uniform distribution |
| Very small drops between tokens | Penalty flattened the distribution |

Notice the approve-reject gap: at no penalty, approve had 8.6 percentage points more than reject (28.7% vs 20.1%). At heavy penalty, the gap is 0.1 percentage points (15.6% vs 15.5%). The penalty erased nearly all of approve's advantage — not because approve is bad, but because it appeared in the lookback window.

---

## Step 8: The "Too Strong" Failure Mode — How Coherence Breaks Down

Here is the scenario that demonstrates why penalty > 1.8 is marked as dangerous:

Imagine generating a customer service response. The model needs to say "the customer" multiple times:

> "Thank you for contacting us. The customer's order has been flagged. The customer will receive a refund. Please tell the customer..."

At penalty=1.3: "the" and "customer" get mild discouragement after first use. The model might slightly vary phrasing — "the order has been flagged. You will receive a refund. Please advise the customer..." — which is natural and actually better writing.

At penalty=1.8: "the" has appeared many times and faces a heavy penalty. The model starts actively avoiding it. Output degrades: "Your order has been flagged. A refund will arrive. Please advise regarding..." — grammatically awkward, stilted.

At penalty=2.5: Even common function words like "a", "is", "the", "to" get heavily penalized after their first use. The model can no longer form coherent sentences because the most common words in any language are always appearing in the lookback window.

> **The lesson**: Repetition penalty doesn't know which repetitions are intentional and which are looping failures. It applies the same pressure to "approve approve approve" (bad loop) and "the the the" (also bad) as it does to "The president of the company said the policy..." (perfectly natural reuse of "the"). High penalties create unnatural avoidance of necessary repetitions.

---

## Step 9: The Lookback Window in Practice

Three tasks, three different lookback window choices:

### Code Generation (lookback = 32)

```python
for i in range(n):       # 'for' at position 1
    result = process(i)  # 'i' at position 4
    if result:           # 'if' at position 7
        for j in range(m):   # 'for' again at position 10
```

If `for` has a long lookback window, it gets penalized when it appears at position 10. But `for` is syntactically necessary here — it's not a loop caused by the model getting confused. Short-to-medium lookback (32 tokens) strikes a balance: prevents the model from writing `for for for` in a broken loop while still allowing `for` to appear legitimately a few lines later.

### Story Generation (lookback = 128)

In a narrative, you want characters' names, key objects, and thematic words to recur across paragraphs. But you don't want the model to repeat entire phrases. A long lookback window (128 tokens) catches paragraph-level repetition while allowing tokens from several paragraphs ago to be used again.

### Chatbot (lookback = 64)

Conversational responses tend to be 50–150 tokens long. A 64-token lookback keeps the current response diverse internally while still allowing common words to recur naturally across turns.

---

## Step 10: The "Adjacent Repeats" Diagnostic

The experiment suggests this diagnostic code:

```python
def count_adjacent_repeats(generated_ids):
    repeats = 0
    for i in range(1, len(generated_ids)):
        if generated_ids[i] == generated_ids[i - 1]:
            repeats += 1
    return repeats
```

This counts how often consecutive tokens are identical — the "the the the" type of loop.

**How to use it to calibrate penalty:**

1. Run your model with penalty=1.0 and count adjacent repeats over 1000 samples
2. Try penalty=1.1 — did adjacent repeats drop significantly?
3. Try penalty=1.3 — how much further did repeats drop?
4. Plot penalty vs adjacent-repeat count
5. Find the "elbow" — where adding more penalty stops reducing repeats. That's your sweet spot.

**What "elbow" means here:**

Going from 1.0 to 1.1 might cut adjacent repeats in half. Going from 1.1 to 1.3 might cut them in half again. Going from 1.5 to 1.8 might reduce them by only 5%. The 1.5→1.8 range is diminishing returns — you're not getting fewer loops but you are getting more incoherence. The elbow at ~1.5 is where you stop.

---

## Step 11: Common Misconceptions — Cleared Up

**Misconception 1: "Repetition penalty stops the model from repeating tokens"**

No. It discourages repetition. The model can still output the same token multiple times. At penalty=2.0, if "approve" genuinely has the highest logit by a wide margin, it can still win the sampling lottery multiple times in a row — just less often than without the penalty.

**Misconception 2: "Higher penalty = better output quality"**

Only up to a point. Too high a penalty forces the model to avoid words it needs to use naturally. The relationship between penalty and quality is an inverted U — quality improves as you add penalty (reducing loops) then falls as penalty becomes excessive (forcing unnatural avoidance). The peak of the U depends on your task.

**Misconception 3: "Repetition penalty and temperature do the same thing"**

They're completely different mechanisms. Temperature changes all tokens' probabilities globally (reshaping the whole distribution). Repetition penalty changes only the specific tokens that have appeared in the lookback window, leaving all other tokens unchanged. They also operate at different levels: temperature is a single-step operation; repetition penalty is a history-dependent operation.

**Misconception 4: "Presence and frequency penalty are just different names for the same thing"**

No. Presence penalty is binary — any seen token gets a flat deduction. Frequency penalty is proportional — tokens seen more times get larger deductions. For a token seen 5 times, frequency penalty applies 5x the pressure of presence penalty. Their behavior diverges significantly once any token appears more than once.

**Misconception 5: "I should always use repetition penalty"**

Only if your task has a repetition problem. For tasks like code generation, mathematics, or factual extraction, many tokens legitimately need to appear multiple times. Adding repetition penalty to these tasks can actively hurt output quality by discouraging necessary repetitions. Always diagnose first: if you're not seeing loops, don't add penalty.

---

## Step 12: Practical Decision Guide

```
Step 1: Do you observe looping or repetitive output?
  NO  → Don't add repetition penalty; it won't help and may hurt
  YES → Continue to Step 2

Step 2: What type of repetition are you seeing?
  Adjacent ("the the the"):
    → Start with frequency penalty, strength 1.1–1.3
    → Short lookback (16–32 tokens) is enough
  Phrase-level (same sentence repeating every 5 tokens):
    → Moderate penalty (1.3–1.5), lookback 32–64
  Paragraph-level (same idea repeated in different words):
    → Light presence penalty (1.1–1.2), long lookback (128+)
    → Also consider checking temperature — too-low temperature causes this type

Step 3: Start conservative, measure, increase only if needed
  → Penalty 1.1 first
  → Measure adjacent repeats and output quality
  → If loops persist, increase to 1.3
  → If quality degrades at any point, back down one step

Step 4: Never exceed 1.5 for production without manual review
  → Test outputs at 1.5+ carefully before deploying
  → Watch for: awkward phrasing, missing function words, stilted tone
```

### Task-Level Settings Reference

| Task | Penalty | Lookback | Type | Notes |
|------|---------|----------|------|-------|
| Classification | 1.0 | N/A | N/A | No repetition issue; penalty not needed |
| Factual QA | 1.0–1.1 | 32 | Presence | Key terms legitimately repeat |
| Summarization | 1.0–1.1 | 64 | Frequency | Important nouns must recur; gentle penalty |
| Code generation | 1.1 | 32 | Frequency | Syntax tokens repeat by design; light touch |
| Chatbot | 1.2 | 64 | Frequency | Conversational variety, allow common words |
| Creative writing | 1.15–1.2 | 128 | Presence | Encourage vocabulary diversity across paragraphs |
| Story generation (long) | 1.2–1.3 | 128–256 | Frequency | Longer memory, moderate loop prevention |
| Brainstorming | 1.3–1.5 | 64 | Presence | Want distinct ideas, strong diversity pressure |

---

## Quick Reference: Graph Interpretation Cheat Sheet

When you open `exp5_repetition.png`, scan in this order:

```
1. Look at the before/after bar charts (row 1)
   → Find the red-shaded bands — those are the penalized tokens
   → Compare colored vs grey bars: how much did penalized tokens shrink?
   → Did they maintain their rank (still tallest), or did other tokens surpass them?
   → Which tokens grew the most from redistribution?

2. Look at the "Penalized Token Probability" line chart (row 2, left+middle)
   → Read the starting probability for each line (at penalty=1.0)
   → How steeply does each line drop?
   → Is there a crossover point where one line falls below another?
   → Where does each line flatten (diminishing returns zone)?

3. Look at the "Logit Reduction" chart (row 2, right)
   → The logit chart shows the internal mechanism; probability chart shows the result
   → Fast early drop + slow later flattening = diminishing returns pattern
   → The shape of this curve tells you how "efficient" each increment of penalty is

4. Look at the Presence vs Frequency comparison (row 3, panels 1–2)
   → In the Presence panel: are the bars within each token group roughly equal?
   → In the Frequency panel: does the staircase pattern appear (more appearances = shorter bar)?
   → The degree of staircase steepness shows how aggressively frequency penalty escalates

5. Look at the guide table (row 3, right)
   → Check which penalty range your current setting falls into
   → Is the red warning row visible? (>1.8 zone)
```

---

## Summary

| Concept | One-sentence explanation |
|---------|--------------------------|
| Repetition penalty | Modifies logits of recently-seen tokens to make them less likely (not impossible) to be sampled again |
| Penalty = 1.0 | No change — the baseline, as if the parameter doesn't exist |
| Positive logit rule | Positive logits get divided by the penalty, making them smaller |
| Negative logit rule | Negative logits get multiplied by the penalty, making them more negative |
| Why asymmetry | Without it, dividing a negative logit would make it less negative, accidentally increasing its probability |
| Lookback window | Only tokens within the last N positions are penalized; older tokens return to normal |
| Presence penalty | Binary — any seen token gets the same flat deduction |
| Frequency penalty | Proportional — tokens seen more times get larger deductions |
| Too strong | Penalties above 1.5–1.8 start penalizing necessary words, causing incoherence |
| Discourages not forbids | Even heavily penalized tokens can still be sampled; it's pressure, not prohibition |

> **Final takeaway**: Repetition penalty is a diagnostic tool, not a default setting. Run without it first. If you observe loops, add it conservatively starting at 1.1 and measure both loop frequency and output coherence. The goal is the minimum penalty that breaks the loop — not the maximum penalty your system will accept.

---

*Part of the 6-Experiment LLM Parameter Learning Path*  
*Previous: Experiment 4 — Combined Filters*  
*Next: Experiment 6 — Real Use Cases*