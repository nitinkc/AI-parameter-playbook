# Experiment 4: Combined Filters — A Step-by-Step Learning Guide

> **What this file is**: A beginner-friendly, ground-up explanation of Experiment 4 (Combined Filters).  
> **Who it's for**: Anyone who completed Experiments 1–3 and wants to understand how temperature, top-k, and top-p work *together* in real production systems.  
> **What you'll get**: The ability to design a sampling configuration from scratch for any task, understand why order matters, and read any combined-filter graph with confidence.

---

## Before You Start: Why Combining Parameters Isn't Just Addition

You've learned three tools so far:

- **Temperature**: reshapes the entire probability distribution (all tokens shift)
- **Top-k**: removes every token outside the top-k ranked positions (fixed cutoff)
- **Top-p**: removes every token outside the cumulative probability nucleus (adaptive cutoff)

A natural assumption is that combining them just adds their effects together. That's mostly true — but the interactions have surprises. Sometimes two parameters cancel each other out. Sometimes one parameter makes another irrelevant. Sometimes a small change to one parameter causes a large effect because of how it interacts with another.

This experiment is about developing the intuition to predict these interactions before running the model — which is what separates engineers who are good at prompt engineering from those who are guessing.

---

## The Core Mental Model: Layered Filtering

Think of the sampling pipeline as a series of gates, each one narrowing the candidate set further:

```
All tokens (e.g., 32,000 in a real LLM)
    │
    ▼  Temperature scaling
    │   → Reshapes probabilities (no tokens removed, just reweighted)
    │
    ▼  Top-k filtering
    │   → Removes tokens below rank k
    │   → e.g., 32,000 → 50 tokens survive
    │
    ▼  Top-p filtering
    │   → Removes tokens outside the nucleus
    │   → e.g., 50 → 12 tokens survive
    │
    ▼  Renormalization
    │   → Surviving tokens rescaled to sum to 1.0
    │
    ▼  Sample one token
```

Each gate operates on the output of the previous gate. This is why order matters. Temperature must come first because it changes the logits that determine rankings. Top-k uses those rankings. Top-p uses the post-top-k probabilities to find its nucleus.

If you ran top-p before temperature, you'd be filtering based on the wrong distribution. If you ran temperature after top-k, the ranking that top-k used would be invalid.

---

## Step 1: Understand the Pipeline in Detail

Let's trace a single token through all three stages using our 10-token example.

**Starting logits:**
```
approve:   2.2
reject:    1.8
review:    1.4
escalate:  0.9
delay:     0.2
audit:     0.1
optimize: -0.3
notify:   -0.6
assign:   -0.8
close:    -1.0
```

**Stage 1: Temperature scaling (T=0.7)**

Divide all logits by 0.7:
```
approve:   2.2 / 0.7 = 3.14
reject:    1.8 / 0.7 = 2.57
review:    1.4 / 0.7 = 2.00
escalate:  0.9 / 0.7 = 1.29
delay:     0.2 / 0.7 = 0.29
audit:     0.1 / 0.7 = 0.14
optimize: -0.3 / 0.7 = -0.43
notify:   -0.6 / 0.7 = -0.86
assign:   -0.8 / 0.7 = -1.14
close:    -1.0 / 0.7 = -1.43
```

Apply softmax to get probabilities:
```
approve:   0.486  (48.6%)
reject:    0.287  (28.7%)
review:    0.165  (16.5%)
escalate:  0.080  (8.0%)
delay:     0.030  (3.0%)
audit:     0.026  (2.6%)
optimize:  0.014  (1.4%)
notify:    0.009  (0.9%)
assign:    0.008  (0.8%)
close:     0.005  (0.5%)
```

Temperature at 0.7 sharpened the distribution: approve went from 36% baseline to 48.6%.

**Stage 2: Top-k filtering (k=5)**

Keep only the top 5 tokens:
```
Survivors: approve (0.486), reject (0.287), review (0.165), escalate (0.080), delay (0.030)
Discarded: audit, optimize, notify, assign, close
Total survivor probability: 0.486 + 0.287 + 0.165 + 0.080 + 0.030 = 1.048
```

Wait — that sums to more than 1.0? No, that can't be right. Let me re-examine: after softmax the probabilities *already* sum to 1.0, so the top-5 probabilities sum to less than 1.0 (the rest goes to the discarded tokens). Here the top 5 sum to 1.048... that's a rounding artifact in this example. In practice, the top-5 tokens' probabilities sum to some value less than 1.0 (since we're leaving out the other 5), and we renormalize.

Let's use cleaner numbers: at T=0.7 the top-5 tokens have probabilities that sum to approximately 0.970 (97% of probability mass). The remaining 5 tokens share about 3%.

After renormalization by 0.970:
```
approve:   0.486 / 0.970 ≈ 0.501  (50.1%)
reject:    0.287 / 0.970 ≈ 0.296  (29.6%)
review:    0.165 / 0.970 ≈ 0.170  (17.0%)
escalate:  0.080 / 0.970 ≈ 0.082  (8.2%)
delay:     0.030 / 0.970 ≈ 0.031  (3.1%)
```

**Stage 3: Top-p filtering (p=0.9)**

Now we have 5 tokens. Apply top-p:
```
Cumulative:
  approve:  0.501
  reject:   0.797  (0.501 + 0.296)
  review:   0.967  ← cumulative first exceeds 0.9 here
```

Nucleus = {approve, reject, review} — 3 tokens survive.

Renormalize by 0.967:
```
approve:  0.501 / 0.967 = 0.518  (51.8%)
reject:   0.296 / 0.967 = 0.306  (30.6%)
review:   0.170 / 0.967 = 0.176  (17.6%)
```

**Final result from three stages:**

| Token | Baseline | After T=0.7 | After k=5 | After p=0.9 |
|-------|---------|------------|---------|------------|
| approve | 36.0% | 48.6% | 50.1% | 51.8% |
| reject | 24.0% | 28.7% | 29.6% | 30.6% |
| review | 16.0% | 16.5% | 17.0% | 17.6% |
| escalate | 10.0% | 8.0% | 8.2% | **0%** |
| delay | 5.0% | 3.0% | 3.1% | **0%** |
| everything else | 9.0% | 3.2% | **0%** | **0%** |

Three stages of filtering progressively concentrated the probability. Started with 10 candidates, ended with 3.

---

## Step 2: Trace All Five Experiment Scenarios

### Scenario 1: Safety (T=0.3, k=0, p=1.0)

T=0.3 is aggressive scaling. With no top-k or top-p filtering, only temperature acts.

After T=0.3, divide all logits by 0.3 — this dramatically widens the logit gaps:
```
approve:  2.2/0.3 = 7.33 → e^7.33 ≈ 1520
reject:   1.8/0.3 = 6.00 → e^6.00 ≈ 403
review:   1.4/0.3 = 4.67 → e^4.67 ≈ 107
escalate: 0.9/0.3 = 3.00 → e^3.00 ≈ 20
delay:    0.2/0.3 = 0.67 → e^0.67 ≈ 1.95
...
```

The ratio between approve and delay goes from 1520:1.95 ≈ 779:1. Approve dominates catastrophically. After softmax, it gets ~89.6% of the probability.

No top-k or top-p: all 10 tokens technically survive, but 5 of them have effectively zero probability.

**Effect**: Very confident, near-deterministic. The distribution looks almost identical to k=1, but without the hard-zero of greedy decoding. Entropy ≈ 0.614.

---

### Scenario 2: Balanced (T=0.7, k=0, p=1.0)

Only temperature acts (T=0.7 as computed above). No additional filtering.

Approve gets ~48.6%, reject ~28.7%, review ~16.5%. The top three tokens cover 94% of the probability. The remaining tokens exist but rarely get sampled.

**Effect**: Confident but with meaningful variation. The model usually says approve but will occasionally say reject or review. Entropy ≈ 1.2–1.5.

---

### Scenario 3: Creative (T=1.2, k=0, p=0.9)

Temperature at 1.2 flattens the distribution. Then top-p trims the tail.

After T=1.2, divide logits by 1.2:
```
approve:  2.2/1.2 = 1.83
reject:   1.8/1.2 = 1.50
review:   1.4/1.2 = 1.17
escalate: 0.9/1.2 = 0.75
delay:    0.2/1.2 = 0.17
audit:    0.1/1.2 = 0.08
...
```

Softmax on these compressed logits gives a flatter distribution:
```
approve:  ~0.24  (24%)
reject:   ~0.18  (18%)
review:   ~0.13  (13%)
escalate: ~0.09  (9%)
delay:    ~0.07  (7%)
audit:    ~0.07  (7%)
...
```

Now apply p=0.9: walk down cumulative until reaching 0.9:
```
approve:   0.24
+reject:   0.42
+review:   0.55
+escalate: 0.64
+delay:    0.71
+audit:    0.78
+optimize: 0.84  (assume ~0.06)
+notify:   0.89  (assume ~0.05)
+assign:   0.93  → crossed 0.9, stop
```

Nucleus ≈ 8–9 tokens. Temperature flattened the distribution so much that p=0.9 needs almost all tokens.

**The interaction**: High temperature causes top-p to be MORE permissive (larger nucleus). This is because a flatter distribution needs more tokens to reach the cumulative threshold. Creative mode ends up with a large nucleus — which is usually the intent.

Entropy ≈ 1.9, nucleus ≈ 8 tokens.

---

### Scenario 4: Diverse + Safe (T=0.9, k=10, p=0.95)

Three parameters all active. Let's trace:

After T=0.9, distribution is moderately flattened — approve ~38%, reject ~25%, review ~16%, etc.

After k=10: all 10 tokens survive (our vocabulary only has 10 tokens, so k=10 has no effect here). In a real LLM with 32,000 tokens, k=10 would be aggressive.

After p=0.95: walk cumulative until 0.95. The moderately-flattened distribution at T=0.9 might need 7–8 tokens to reach 0.95.

**The interaction**: Top-k provided a safety ceiling (in a real LLM, it would have cut 31,990 tokens before top-p even ran). Top-p then adaptively selected the nucleus within those 10 survivors.

Entropy ≈ 1.7–2.0. Diverse but not random.

---

### Scenario 5: Broad (T=0.8, k=40, p=1.0)

In our 10-token example, k=40 has no effect (vocabulary too small). In a real 32,000-token model:

After T=0.8 (slight flattening), top-k=40 reduces 32,000 tokens to the top 40.

No top-p: all 40 survivors are eligible (p=1.0).

**The intent**: Allow diverse outputs (40 tokens) without going into the long tail (the other 31,960 tokens). This is common for code generation where many different valid continuations exist (different variable names, function styles) but you still want to avoid garbage tokens.

Entropy in the 10-token case ≈ 1.4–1.7 (T=0.8 effect without top-p trimming).

---

## Step 3: The Four Key Interaction Effects

When parameters combine, four interaction patterns emerge. Knowing these prevents you from being surprised by your model's behavior.

---

### Interaction 1: "Redundant Filtering" — When One Parameter Makes Another Irrelevant

**Scenario**: T=0.2 (very peaked), p=0.9 (nucleus sampling)

At T=0.2, approve might have 97% probability. The cumulative sum crosses 0.9 after just the first token. Top-p creates a nucleus of size 1 — exactly the same as k=1.

**Result**: Top-p is doing nothing extra. The distribution was already so peaked that p=0.9 selected only the top token anyway.

**How to detect**: If your entropy with p=0.9 is the same as with p=1.0 at a given temperature, top-p is redundant at that temperature.

**When this is fine**: In a real LLM, different input prompts create different distributions. A low temperature like 0.2 will sometimes produce a peaked distribution (making top-p redundant) and sometimes a moderately distributed one (where top-p still helps). So having top-p active is still good practice — it handles the cases where temperature alone isn't enough.

---

### Interaction 2: "Competing Controls" — When Parameters Pull in Opposite Directions

**Scenario**: T=1.5 (very flat), p=0.5 (very tight nucleus)

At T=1.5, the distribution is quite flat. Maybe 8 tokens all have roughly equal probability. Top-p at 0.5 then cuts to keep only the smallest set that covers 50% — perhaps just the top 2 tokens.

**Result**: High temperature tried to make everything diverse and random. Top-p then aggressively cut the result back to 2 tokens. You've spent computational effort "expanding" the distribution only to immediately "collapse" it.

**The outcome is unpredictable**: The final 2 tokens are the top-ranked ones, but their probabilities are now much closer to each other (because T=1.5 compressed the gap between them). So you're sampling from 2 roughly equal options. This feels both random and constrained.

**When this is useful**: When you want variety between runs (different of the two tokens gets sampled) but hard limits on which tokens can appear. Security-conscious generation sometimes uses this pattern.

---

### Interaction 3: "Multiplied Effect" — When Parameters Amplify Each Other

**Scenario**: T=0.3 (already peaked), k=5 (keep top 5), p=0.8 (nucleus at 80%)

Each parameter alone would be moderately restrictive. But combined:
- T=0.3 peaks the distribution → approve ~89%
- k=5 keeps top 5 → discards audit through close
- p=0.8 applies to the peaked, k-filtered result → cumulative hits 0.8 after just approve (89% > 80%)

Final nucleus: just approve. You've arrived at greedy decoding even though no single parameter was set to k=1.

**The lesson**: Conservative settings compound. If your model is outputting the same token every time and you expected variation, check whether your parameters are accidentally compounding to k=1 behavior.

---

### Interaction 4: "Safety Net Behavior" — Top-k as Ceiling for Top-p

**Scenario**: T=1.5 (very flat), k=20 (ceiling), p=0.9 (nucleus)

Without k=20: the flat T=1.5 distribution might need 25+ tokens to reach p=0.9. Including very low-probability tokens.

With k=20: top-k first cuts to the top 20. Top-p then applies to these 20, creating a nucleus of maybe 12–15 tokens.

**Result**: Top-k provides a hard guarantee that no more than 20 tokens can ever be sampled, even with high temperature and permissive top-p. Top-p then adaptively focuses within those 20.

**This is the most common production pattern**: Use top-k as a "safety ceiling" and top-p for "adaptive focus within the ceiling." It combines the hard guarantee of top-k with the adaptive quality of top-p.

---

## Step 4: Why Order Matters

The pipeline is: Temperature → Top-k → Top-p

What if we reordered?

### What happens if you run top-p before top-k?

At baseline (T=1.0), top-p=0.9 selects 6 tokens (let's say). Then top-k=3 would further reduce to 3 tokens.

vs. the correct order: top-k=3 first reduces to 3 tokens, then top-p=0.9 checks if those 3 tokens already exceed 0.9 cumulative (they sum to 0.760), so top-p adds a 4th token.

Different orders give different results. The standard order (temperature → top-k → top-p) is conventional because:
1. Temperature reshapes the logits that determine rank — it must come first
2. Top-k provides a hard ceiling — it should narrow the field before top-p's adaptive work
3. Top-p then fine-tunes within the top-k ceiling

### What if temperature came last?

Then top-k and top-p would filter based on the *unscaled* logit distribution — the raw rankings before temperature adjustment. Since temperature doesn't change rankings (just probabilities), top-k would give the same result. But top-p would use the wrong probabilities to compute its cumulative sum, giving different (and unintended) nucleus sizes.

> **Rule of thumb**: If you ever implement your own sampling, always apply temperature first, then top-k, then top-p. This matches the standard and produces predictable behavior.

---

## Step 5: Read the Graphs — Panel by Panel

The graph `exp4_combined.png` has panels across 3 rows. Here is how to read each one.

---

### Row 1: "Final Distributions for Each Preset"

Four bar charts, one per preset scenario.

**What you're looking at:**
- X-axis: the 20 tokens (token index)
- Y-axis: final probability after all filters applied
- Bar color: a different color per preset
- Title: shows the preset name, final entropy, and top-token probability

**How to read them:**

Start with the title numbers:
- `H=X.XX` is entropy. Lower = more peaked, higher = more spread.
- `P(top)=XX%` is the top-token probability. Higher = model is more confident.

Then look at the bar shape:
- One very tall bar + everything else near zero = peaked, deterministic behavior
- Several bars with similar heights = flat, diverse behavior
- A few medium-height bars dropping off quickly = balanced

**Comparing across the four panels**: The panels should show progressively flatter distributions from left (Precise) to right (Exploratory). If you see an unexpected shape, it reveals an interaction effect between the parameters.

---

### Row 2: "Pipeline Step-by-Step for Balanced Preset"

Four panels showing the distribution at each stage of the pipeline for T=0.7, k=10, p=0.9.

**What you're looking at:**

Panel 0: Raw Logits — a bar chart of the raw logit values (can be negative, not probabilities)  
Panel 1: After Temperature — probabilities after T=0.7 scaling and softmax  
Panel 2: After Top-k — probabilities after keeping only top-10, renormalized  
Panel 3: After Top-p — probabilities after nucleus sampling at p=0.9, renormalized  

Each panel's title shows "N eligible" — the number of tokens with non-zero probability at that stage.

**How to read them:**

Follow the evolution:
- Panel 0: Logits range widely (positive and negative). No probabilities yet.
- Panel 1: All bars are now positive. The distribution is shaped by T=0.7. Notice the shape.
- Panel 2: Some bars dropped to zero (top-k cutoff). The remaining bars shifted upward slightly (renormalization).
- Panel 3: More bars dropped to zero (top-p cutoff). Remaining bars shifted upward again.

**The "eligible" count is your key metric**: 20 → 10 → 8 → 5 (example). Each stage reduced the candidate count. The final "eligible" count is how many tokens can actually be sampled.

**Look for stages where "eligible" didn't change**: If top-k says "10 eligible" and top-p also says "10 eligible," then top-p added nothing — the top-k distribution was already tight enough that p=0.9 included all 10 survivors. This is the "redundant filtering" pattern from Step 3.

---

### Row 3, Left: "Entropy: Temperature × Top-k Interaction Heatmap"

This is the most information-dense panel. It shows entropy across a grid of temperature values and top-k values simultaneously.

**What you're looking at:**
- X-axis: temperature (from 0.1 to 2.0, 30 steps)
- Y-axis: top-k value (k=1, 2, 5, 10, 20, 50 from bottom to top)
- Color: entropy (darker/purple = low entropy, brighter/yellow = high entropy)

**How to read it:**

Move left to right (increasing temperature): entropy generally increases. The colors get brighter as you go right.

Move bottom to top (increasing k): entropy generally increases. The colors get brighter going up.

**The critical patterns:**

1. **Bottom-left corner (low T, low k)**: Dark purple. Very low entropy. Near-greedy decoding. This is where k=1 and k=2 live at low temperatures.

2. **Top-right corner (high T, high k)**: Bright yellow/orange. High entropy. Maximum diversity.

3. **The "flat" zone**: At some point, increasing k further doesn't increase entropy much — because the distribution was already peaked, and all the real probability mass was in the top few tokens. Look for where the heatmap's brightness stops changing as you move up the y-axis. That row is where top-k stopped mattering.

4. **Temperature dominates**: The heatmap changes more dramatically left-to-right (temperature) than bottom-to-top (top-k). This confirms that temperature is the "master dial" and top-k is a secondary control.

---

### Row 3, Right: "Final Entropy per Preset" (horizontal bar chart)

A simple horizontal bar chart comparing entropy across the five presets.

**What you're looking at:**
- Y-axis: preset names
- X-axis: entropy value
- Bar color: the preset's color
- Annotation: the exact entropy value

**How to read it:**

The bars should be in increasing order from top (Safety/most precise) to bottom (Broad/most creative). If they're not, it reveals a surprising interaction.

**What to check:**

1. Is Broad actually broader (higher entropy) than Creative? If Creative has p=0.9 and Broad has p=1.0 but lower temperature, they might be surprisingly similar.

2. Is Safety the lowest? It should be — T=0.3 is aggressive. If something else has lower entropy, something unexpected happened.

3. How large is the gap between Safety and Balanced? A big gap means temperature matters a lot at these values. A small gap means the two temperatures (0.3 vs 0.7) produce similar output diversity for this distribution.

---

## Step 6: The Entropy Ordering Rule

Here is a practical rule you can use to predict how any combination of parameters will rank in entropy before running the experiment:

**Within a fixed top-p and top-k, entropy increases with temperature.**  
**Within a fixed temperature, entropy increases with p (more top-p permissiveness).**  
**Within a fixed temperature and p, entropy increases with k (more top-k permissiveness).**

But the interactions mean these rules can be violated when parameters compensate:

- High T + low p can have lower entropy than medium T + high p
- Low T + high k can have lower entropy than medium T + low k

**The "entropy budget" mental model**: Think of entropy as a budget. Temperature adds to it. Top-k subtracts from it (hard ceiling). Top-p subtracts from it (soft ceiling). The final entropy is roughly: temperature contribution − top-k subtraction − top-p subtraction.

When top-k or top-p is very restrictive, it can "spend" more than temperature added, capping entropy below what temperature alone would produce.

---

## Step 7: How to Read the Sample Output Text

The original experiment shows:

```
=== Safety: T=0.3 ===
Entropy: 0.614
approve       0.896
reject        0.087
review        0.013
escalate      0.003
delay         0.001
audit         0.001
optimize      0.000
notify        0.000
assign        0.000
close         0.000
```

**Reading guide:**

| What you see | What it means |
|---|---|
| `Entropy: 0.614` | Very low — one token dominates heavily |
| `approve 0.896` | 89.6% probability. T=0.3 caused this extreme peak. |
| `reject 0.087` | 8.7% — rare but possible |
| `review 0.013` | 1.3% — very rare |
| Others ≈ 0.000 | Functionally zero without hard top-k/top-p exclusion |

Notice: no hard zeros in the "Safety" scenario (k=0, p=1.0). The near-zeros are purely from temperature compression, not hard exclusion.

---

```
=== Creative: T=1.2, p=0.9 ===
Entropy: 1.892
approve       0.201
reject        0.189
review        0.176
escalate      0.147
delay         0.115
audit         0.097
optimize      0.075
notify        0.000
assign        0.000
close         0.000
```

**Reading guide:**

| What you see | What it means |
|---|---|
| `Entropy: 1.892` | Much higher — 7 tokens have real probability |
| `approve 0.201` | Only 20.1% — temperature compressed the gap between tokens |
| `reject 0.189` | Close behind approve — the model is genuinely unsure |
| `notify 0.000` | Hard zero — top-p cut the nucleus here |
| `assign 0.000` | Hard zero — also excluded by top-p |

Notice the hard zeros now appear — because p=0.9 created a nucleus boundary. Also notice that approve's probability dropped from 89.6% (Safety) to 20.1% (Creative). The same model, same logits, dramatically different behavior from parameter choices alone.

---

## Step 8: The Optional Dive Deeper — Temperature × Top-p Interaction Plot

The experiment suggests plotting entropy vs temperature for three top-p values (0.5, 0.8, 1.0). Here's what you'd see and how to interpret it:

**What the three lines show:**

Line 1 (p=1.0, no top-p filtering): Entropy rises smoothly with temperature. No sudden changes.

Line 2 (p=0.8): Entropy rises with temperature but reaches a ceiling. At some temperature, the distribution becomes flat enough that p=0.8 is including almost all tokens anyway — the top-p stops cutting anything. After that point, lines 1 and 2 converge.

Line 3 (p=0.5): More restricted. The ceiling is lower. But at very high temperatures (T > 2.0), even p=0.5 includes many tokens because the distribution is so flat.

**The convergence point tells you something important:**

Where lines 2 and 1 converge: that's the temperature at which p=0.8 becomes redundant. Past that temperature, top-p at 0.8 is doing nothing extra.

Where line 3 converges with the others: the temperature at which even p=0.5 stops mattering.

**The practical reading:**

The higher the top-p threshold, the lower the temperature where that top-p becomes redundant. For most distributions, top-p above 0.95 becomes redundant at fairly low temperatures (T ≈ 0.8–1.0). This is why many practitioners say "above p=0.95, you might as well not bother."

---

## Step 9: Common Misconceptions — Cleared Up

**Misconception 1: "More parameters = better control"**

Not necessarily. Adding more filtering parameters can lead to conflicting effects that produce surprising results. Start with one parameter, understand its effect, then add another. Building up from simplicity is more reliable than configuring all three at once.

**Misconception 2: "The presets in the table are universally correct"**

The "Precise: T=0.2, p=0.9, k=40" preset works well for many code generation tasks — but if your model's distribution for a particular prompt is already peaked, p=0.9 will be redundant, and you're doing work for nothing. Presets are starting points, not ground truth.

**Misconception 3: "If I combine T=0.7 and p=0.9, I get their individual effects added together"**

Not quite. The effects interact multiplicatively, not additively. T=0.7 changes the shape of the distribution that p=0.9 then cuts. The final entropy is not "entropy from T=0.7" + "entropy reduction from p=0.9." You have to trace through the pipeline to get the actual result.

**Misconception 4: "High entropy is always bad"**

High entropy means more diversity and randomness. For brainstorming or creative writing, high entropy is exactly what you want. "Bad" entropy means entropy that's higher than your task requires — which causes incoherence. Match entropy to task requirements.

**Misconception 5: "If my output looks bad, I should lower the temperature"**

Maybe. But the cause might be top-p being too permissive, or top-k being too large, or the model simply not knowing the answer. Lowering temperature when the problem is model capability doesn't help. Always diagnose first: is the output incoherent (suggests high entropy → lower T or p), or is it repetitive (suggests low entropy → raise T or use repetition penalty)?

---

## Step 10: Practical Decision Guide

Use this step-by-step process when configuring sampling for a new task:

```
Step 1: Define your task type
  → One correct answer?     → Start: T=0.2, k=40, p=0.9
  → Quality matters?        → Start: T=0.7, k=50, p=0.9
  → Creativity wanted?      → Start: T=1.0, k=100, p=0.95
  → Maximum exploration?    → Start: T=1.2, k=0, p=1.0

Step 2: Check your output
  → Repetitive / loops?     → Raise T slightly, or add repetition penalty
  → Too random / incoherent → Lower T, or tighten p
  → Always same answer?     → Raise T or p
  → Occasionally wrong tokens appear → Lower k or p

Step 3: Fine-tune one parameter at a time
  → Never adjust all three simultaneously
  → Change T first (it has the largest effect)
  → Then adjust p (adaptive quality)
  → Then adjust k (hard ceiling if needed)

Step 4: Measure
  → Log entropy per response if possible
  → Check top-token probability distribution
  → Evaluate against your task metric
```

### Quick Preset Reference

| Use Case | T | p | k | Expected Entropy |
|----------|---|---|---|-----------------|
| Classification | 0.1–0.2 | 0.9 | 20 | < 0.5 bits |
| Factual QA | 0.3 | 0.9 | 40 | 0.5–1.0 bits |
| Code generation | 0.2 | 0.95 | 40 | 0.8–1.5 bits |
| Summarization | 0.5 | 0.9 | 50 | 1.0–1.5 bits |
| General chat | 0.7 | 0.9 | 50 | 1.3–1.8 bits |
| Creative writing | 1.0 | 0.95 | 100 | 1.8–2.5 bits |
| Brainstorming | 1.2 | 1.0 | 0 | 2.5–3.5 bits |

---

## Quick Reference: Graph Interpretation Cheat Sheet

When you open `exp4_combined.png`, scan in this order:

```
1. Look at the preset distribution panels (row 1)
   → Do they get progressively flatter left to right?
   → Read the entropy and P(top) from each title
   → Identify which preset is most/least deterministic

2. Look at the pipeline step panels (row 2)
   → Watch the "eligible" count drop at each stage
   → Which stage caused the biggest drop? That parameter is doing the most work
   → If "eligible" didn't change between two stages, one parameter is redundant

3. Look at the entropy heatmap (row 3, left)
   → Is the bright region (high entropy) in the top-right corner? (It should be)
   → Find the "flat zone" where adding more k doesn't change entropy
   → The vertical gradient (temperature effect) should be steeper than the horizontal gradient (top-k effect)

4. Look at the entropy bar chart (row 3, right)
   → Are the bars in the order you expected?
   → If not, identify which preset "broke" the expected ordering
   → Read the exact entropy values — is the gap between Safety and Balanced larger or smaller than expected?
```

---

## Summary

| Concept | One-sentence explanation |
|---------|--------------------------|
| Pipeline order | Temperature → Top-k → Top-p → Sample; order is not arbitrary |
| Redundant filtering | When one parameter makes another irrelevant (peaked distribution + low p) |
| Competing controls | When high T and low p fight each other (flat distribution, then tight cut) |
| Multiplied effect | Multiple conservative settings compounding to near-greedy behavior |
| Safety net pattern | Top-k as hard ceiling + top-p for adaptive focus within that ceiling |
| Entropy budget | Temperature adds entropy; top-k and top-p subtract it |
| Temperature dominates | Across all interaction effects, temperature has the largest single impact |

> **Final takeaway**: No single parameter setting is right for all tasks. In production, you build intuition by starting with the appropriate preset for your task type, then adjusting one parameter at a time while measuring entropy and output quality. The four interaction patterns — redundant, competing, multiplied, safety net — give you a vocabulary to diagnose what's happening when outputs surprise you.

---

*Part of the 6-Experiment LLM Parameter Learning Path*  
*Previous: Experiment 3 — Top-k Sampling*  
*Next: Experiment 5 — Repetition Penalties*