# Experiment 6: Real Use Cases — A Step-by-Step Learning Guide

> **What this file is**: A beginner-friendly, ground-up explanation of Experiment 6 (Real Use
> Cases).  
> **Who it's for**: Anyone who completed Experiments 1–5 and wants to translate parameter knowledge
> into practical task-specific configurations.  
> **What you'll get**: A complete framework for designing sampling configurations from scratch,
> understanding why different tasks need different distributions, and a repeatable calibration process
> you can apply to any LLM task.

---

## Before You Start: The Shift from Learning to Applying

The first five experiments taught you mechanisms — what each parameter does in isolation or in
combination. This final experiment asks a different question:

> Given a real task with real requirements, how do you choose the right parameters?

This is harder than it sounds. The gap between "I understand temperature" and "I can configure a
production summarization system" is significant. Experiment 6 closes that gap by working backwards
from task requirements to parameter values — the direction you'll always need to work in practice.

The key insight is this: **before you touch any parameter, you need to define what a good output
looks like for your task.** Parameters are a means to an end. The end is always a task outcome.

---

## The Four Task Distributions — What They Represent

The experiment uses four synthetic logit distributions, each designed to mimic the characteristics
of a real model producing output for a specific task. Understanding why each distribution is shaped
the way it is teaches you more than any formula.

### Distribution 1: Classification (High Confidence)

```python
logits_class = [4.5, 0.3, 0.1, -0.2, -0.5, -1.0, -1.2, -1.5, -2.0, -2.5]
tokens_class = ['INTENT_APPROVE', 'INTENT_REJECT', 'INTENT_REVIEW', ...]
```

**Why this shape?**

The first logit (4.5) is dramatically higher than all others (next is 0.3 — a gap of 4.2 points).
This represents a model that, given a well-trained classification prompt, is highly confident about
the correct label.

In a real intent classifier, if your model has been fine-tuned well and the input clearly says "
please approve my request," the INTENT_APPROVE logit should be much higher than everything else. If
it isn't — if INTENT_REJECT has a logit close to INTENT_APPROVE — that's a signal the model is
uncertain and your prompt or training data may need work.

**What happens at baseline (T=1.0)?**

```
softmax([4.5, 0.3, 0.1, -0.2, ...]) →
  INTENT_APPROVE: ~0.98  (98%)
  INTENT_REJECT:  ~0.01  (1%)
  Everything else: ~0.01 combined
```

Even without any filtering, the model is 98% confident. Entropy is very low.

**What this tells you about parameter requirements:**

Classification already produces the right behavior at baseline. The job of your parameters here is
not to add diversity — it's to preserve the natural confidence while preventing any accidental
divergence. Low temperature reinforces the already-peaked distribution; no top-k or top-p is needed
because the distribution is already clean.

---

### Distribution 2: Summarization (Moderate Confidence)

```python
logits_sum = [2.1, 1.9, 1.7, 1.3, 0.8, 0.5, 0.2, -0.2, -0.5, -1.0]
tokens_sum = ['BENEFIT', 'COST', 'TIMELINE', 'STAKEHOLDER', 'RISK', ...]
```

**Why this shape?**

The logits decline gradually. The top four tokens (BENEFIT, COST, TIMELINE, STAKEHOLDER) are all
within 0.8 logit points of each other. The model thinks any of these could legitimately be the next
important concept to include in a summary.

This reflects a real summarization model: given a business report, "benefit," "cost," "timeline,"
and "stakeholder" are all valid things to include next. The model has no single correct answer — it
must choose from several reasonable options.

**What happens at baseline (T=1.0)?**

```
softmax([2.1, 1.9, 1.7, 1.3, 0.8, 0.5, 0.2, -0.2, -0.5, -1.0]) →
  BENEFIT:     ~0.22  (22%)
  COST:        ~0.18  (18%)
  TIMELINE:    ~0.15  (15%)
  STAKEHOLDER: ~0.10  (10%)
  RISK:        ~0.07  (7%)
  OPPORTUNITY: ~0.06  (6%)
  ...
```

Entropy is moderate — several tokens have meaningful probability. This is appropriate for
summarization. The model should produce varied but relevant content.

**What this tells you about parameter requirements:**

You want to preserve the top few options (BENEFIT, COST, TIMELINE, STAKEHOLDER, RISK) while
preventing the tail (TANGENT, NOISE) from occasionally appearing. Moderate temperature keeps the
natural competition between relevant concepts; top-p around 0.9–0.95 prunes the genuinely
low-quality tokens.

---

### Distribution 3: Code Generation (Many Plausible Continuations)

```python
logits_code = [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
tokens_code = ['FOR_LOOP', 'IF_STMT', 'FUNC_DEF', 'CLASS_DEF', 'IMPORT', ...]
```

**Why this shape?**

The logits form an almost perfectly linear decline — 1.2, 1.1, 1.0, 0.9... Each token is 0.1 logit
points below the previous one. This represents a situation where the model has many equally-valid
next constructs. At the beginning of a Python function, a for loop, an if statement, a function
call, or an assignment could all be syntactically correct.

This is the opposite of classification. The code distribution has no clear winner — it's one of the
flattest distributions in the experiment.

**What happens at baseline (T=1.0)?**

```
softmax([1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]) →
  FOR_LOOP:   ~0.14  (14%)
  IF_STMT:    ~0.13  (13%)
  FUNC_DEF:   ~0.12  (12%)
  CLASS_DEF:  ~0.11  (11%)
  IMPORT:     ~0.10  (10%)
  ...
  DECORATOR:  ~0.07  (7%)
```

High entropy. All 10 constructs are genuinely eligible. This is realistic for a position in code
where many constructs could begin.

**What this tells you about parameter requirements:**

You want diversity (multiple valid code patterns should be explorable across runs) but you also need
guard rails (the model shouldn't output truly invalid constructs). Top-k=20–40 with moderate
temperature is common: it preserves the "many valid options" nature of the distribution while
ensuring you stay within plausible constructs.

---

### Distribution 4: Creative Writing (Broad Exploration Desired)

```python
logits_creative = [1.5, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2]
tokens_creative = ['BRAVE', 'QUIRKY', 'DARK', 'HUMOROUS', 'POIGNANT', ...]
```

**Why this shape?**

The logits span from 1.5 to -0.2 — a range of only 1.7 points across 10 tokens. The distribution is
moderately flat but with a slight preference for the top few. The model slightly prefers BRAVE and
QUIRKY for the next narrative tone, but DARK, HUMOROUS, and POIGNANT are all genuinely on the table.

This represents a creative writing model choosing a narrative direction. Unlike classification (one
right answer) or summarization (several relevant options), here "wrong" barely exists — any of these
tones could work depending on the story.

**What happens at baseline (T=1.0)?**

```
softmax([1.5, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2]) →
  BRAVE:      ~0.17  (17%)
  QUIRKY:     ~0.15  (15%)
  DARK:       ~0.12  (12%)
  HUMOROUS:   ~0.10  (10%)
  POIGNANT:   ~0.09  (9%)
  ...
  MYSTICAL:   ~0.06  (6%)
```

Moderate-to-high entropy. Several tones have meaningful probability. But the tail (MYSTICAL at 6%)
is low enough that you might want to preserve it — in creative writing, unusual choices are often
the best ones.

**What this tells you about parameter requirements:**

High temperature to flatten the distribution further (make BRAVE vs MYSTICAL more competitive),
combined with a permissive top-p (0.85–0.95) that preserves even the unusual choices. This is the
setting where parameters actively expand the model's range rather than contracting it.

---

## Step 1: The Four Recommended Configurations — Traced Through

### Configuration 1: Classification (T=0.2, k=0, p=1.0)

**Step 1 — Temperature scaling (÷0.2):**

Already-peaked logits get even more peaked:

```
4.5 / 0.2 = 22.5  → e^22.5 ≈ 5.9 billion
0.3 / 0.2 = 1.5   → e^1.5  ≈ 4.5
```

The ratio between INTENT_APPROVE and the next token is now 5.9 billion to 4.5 — effectively
infinite. After softmax: INTENT_APPROVE ≈ 100.00%.

**Step 2 — Top-k (k=0, no filtering):** All tokens remain.

**Step 3 — Top-p (p=1.0, no filtering):** All tokens remain.

**Final result:** INTENT_APPROVE is sampled every time. Entropy ≈ 0.0–0.1. This is the intended
behavior — classification should be deterministic.

**Why not use k=1 (greedy) instead?**

k=1 would also always pick INTENT_APPROVE. But T=0.2 with no top-k is safer across different inputs:
if the model encounters an ambiguous classification input (logits closer together), T=0.2 will still
heavily favor the top token but won't be fully greedy. k=1 is more brittle — it commits to greedy
regardless of how uncertain the distribution is.

---

### Configuration 2: Summarization (T=0.6, k=0, p=0.95)

**Step 1 — Temperature scaling (÷0.6):**

The moderate logits get moderately sharpened:

```
2.1 / 0.6 = 3.5   → BENEFIT gets boosted
1.9 / 0.6 = 3.17  → COST, still competitive
1.7 / 0.6 = 2.83  → TIMELINE, still competitive
...
-1.0 / 0.6 = -1.67 → NOISE, gets more negative
```

After softmax: Top 4 tokens become more dominant; bottom tokens (TANGENT, NOISE) shrink further.

**Step 2 — Top-k (k=0, no filtering):** All tokens survive.

**Step 3 — Top-p (p=0.95):**

Walk down sorted tokens until cumulative sum ≥ 0.95. At T=0.6, the top 6–7 tokens probably cover
95%. TANGENT and NOISE get excluded.

**Final result:** 6–7 concept tokens survive. Entropy ≈ 1.5–2.0. BENEFIT and COST are still most
likely, but RISK, OPPORTUNITY, CONSTRAINT occasionally appear. The "junk" tokens (TANGENT, NOISE)
are excluded.

---

### Configuration 3: Code Generation (T=0.8, k=30, p=1.0)

**Step 1 — Temperature scaling (÷0.8):**

The already-flat distribution gets only slightly sharpened:

```
1.2 / 0.8 = 1.5   → FOR_LOOP
1.1 / 0.8 = 1.375 → IF_STMT
...
0.3 / 0.8 = 0.375 → DECORATOR
```

Small gaps become slightly larger, but distribution is still quite flat.

**Step 2 — Top-k (k=30):**

With only 10 tokens, k=30 has no effect here. In a real 32,000-token model, k=30 would aggressively
cut the long tail of truly implausible code tokens.

**Step 3 — Top-p (p=1.0, no filtering):** All surviving tokens remain.

**Final result:** All 10 token types are eligible. Entropy ≈ 2.0–2.5. Each run may produce a
different code construct. This is the intent — you want diverse, explorable code generations across
multiple runs (useful for pass@k evaluation where you generate k solutions and check if any passes).

---

### Configuration 4: Creative Writing (T=1.1, k=0, p=0.85)

**Step 1 — Temperature scaling (÷1.1 — wait, T=1.1 means divide by 1.1):**

Actually T=1.1 > 1.0, so we divide by 1.1, which compresses the logits:

```
1.5 / 1.1 = 1.364  → BRAVE (slightly smaller logit)
1.4 / 1.1 = 1.273  → QUIRKY
...
-0.2 / 1.1 = -0.182 → MYSTICAL (less negative, slightly boosted!)
```

The distribution becomes slightly flatter. The gap between BRAVE and MYSTICAL shrinks from 1.7 to
1.546 — small but meaningful across softmax.

**Step 2 — Top-k (k=0):** All tokens remain.

**Step 3 — Top-p (p=0.85):**

At T=1.1, the distribution is flat. The cumulative sum reaches 0.85 after maybe 7–8 tokens. TRAGIC
and MYSTICAL (the tail) get cut.

**An interesting tension here:** T=1.1 tried to include more tokens (by flattening), but p=0.85 cuts
the bottom 2 back out. The net effect is: middle tokens (DARK, HUMOROUS, POIGNANT, SARCASTIC,
TENDER, WHIMSICAL) get more probability than they would at T=1.0, while the absolute tail (TRAGIC,
MYSTICAL) are still excluded.

**Final result:** 7–8 tones eligible. Entropy ≈ 2.0–2.5. A genuinely diverse set of narrative
options, but the two most unusual choices are excluded by p=0.85.

---

## Step 2: Understand the Entropy Progression

The experiment predicts this entropy ordering:

```
Classification: ~0.5–1.0
Summarization:  ~1.5–2.0
Code:           ~2.0–2.5
Creative:       ~2.0–2.8
```

Let's verify this makes sense by examining what drives entropy in each case:

### Why Classification Has Lowest Entropy

Two factors multiply together:

1. The input distribution is already extremely peaked (logit gap of 4.2 between top and second)
2. T=0.2 further amplifies this gap by a factor of 5

Result: A single token dominates catastrophically. Very low entropy.

### Why Summarization Is in the Middle

Two factors balance each other:

1. The input distribution is moderately flat (logit gap of 1.1 between top four)
2. T=0.6 partially sharpens this
3. p=0.95 cuts the genuinely bad tokens

Result: A few tokens compete meaningfully. Moderate entropy.

### Why Code Has High Entropy

Two factors push toward high entropy:

1. The input distribution is very flat (logit gaps of 0.1 between each token)
2. T=0.8 barely sharpens this
3. k=30 does nothing for this small vocabulary (would matter in a real model)

Result: All tokens remain competitive. High entropy.

### Why Creative May Equal or Exceed Code

Counterintuitive: Creative uses T=1.1 (which flattens) but also p=0.85 (which cuts the tail). Code
uses T=0.8 (which sharpens slightly) and k=30 with p=1.0 (which does almost nothing here).

The creative distribution starts more peaked than the code distribution (range 1.7 vs 0.9), but
T=1.1 compresses it significantly. The p=0.85 cut removes 2–3 tokens but boosts the survivors via
renormalization. The final entropy depends on which of these effects dominates.

**The important lesson:** You can't always predict the final entropy without tracing through the
pipeline. Experiment 6 is where this becomes clear — the same entropy target can be reached from
different starting distributions via different parameter combinations.

---

## Step 3: Read the Graphs — Panel by Panel

The graph `exp6_usecases.png` has panels across 3 rows. Here is how to read each one.

---

### Row 1, Left + Middle: "Task Map: Creativity vs Precision"

A scatter plot with tasks as points in a two-dimensional space.

**What you're looking at:**

- X-axis: "Creativity / Diversity" (0 = fully deterministic, 1 = fully exploratory)
- Y-axis: "Precision / Accuracy" (0 = quality doesn't matter, 1 = must be correct)
- Colored dots: each task, labeled
- Background quadrant shading: two zones (precise/deterministic, creative/exploratory)
- Dashed crosshair lines at (0.5, 0.5): dividing the space into four quadrants

**How to read each quadrant:**

```
Top-left (high precision, low creativity):
  Classification, Factual QA, Code generation
  → These tasks demand correct answers; exploration is a liability

Top-right (high precision, high creativity):
  This quadrant is mostly empty — it's hard to require both simultaneously
  Summarization lives near here: faithful AND fluent

Bottom-right (low precision, high creativity):
  Creative writing, brainstorming
  → These tasks reward exploration; "correct" is ill-defined

Bottom-left (low precision, low creativity):
  Rarely targeted deliberately (neither correct nor diverse)
```

**How to read task positions:**

A task's position tells you both what parameters to prioritize and what trade-offs to expect:

- Tasks near top-left: prioritize temperature reduction, accept low diversity
- Tasks near bottom-right: prioritize higher temperature and permissive top-p
- Tasks near the center: balanced settings, tune from the middle outward

**The diagonal from top-left to bottom-right** is the "parameter dial" from T=0.1 (top-left) to
T=1.2 (bottom-right). Most production tasks lie somewhere along this diagonal.

---

### Row 1, Right: "Recommended Presets" Table

A reference table showing T, p, k, and rep penalty for each task.

**How to read it:**

Each row is a task. Scan across the columns:

- T column: lower values are in the "Classification" rows, higher in "Creative" — confirms the
  temperature-as-master-dial principle
- p column: most tasks use 0.9; note which tasks deviate and ask why
- k column: code generation uses a non-zero k (guard rail); others often don't
- rep column: only tasks where looping is a risk use penalty > 1.0

**Use this table as a starting point, not a definitive answer.** The values reflect common practice,
but your specific model, fine-tuning, and prompts will require adjustment.

---

### Row 2, Three Panels: "Temperature / Top-p / Repetition Penalty by Task"

Three horizontal bar charts showing each parameter's recommended value per task.

**How to read the Temperature chart:**

The bars should increase from top (Classification) to bottom (Brainstorming). The visual length of
each bar is proportional to T.

Key things to notice:

- How large is the jump from Classification to Factual QA? (Should be small — both are precision
  tasks)
- How large is the jump from Chat to Creative Writing? (Should be noticeable — a significant shift)
- The yellow dashed line at T=0.7 is the "general-purpose default" — check how many tasks sit near
  it vs. far from it

**How to read the Top-p chart:**

Most bars should be similar in length (around 0.9), but look for deviations:

- Brainstorming: p=1.0 (no filtering)
- Code generation: slightly higher p (0.95) to allow diverse valid constructs
- Classification: might use lower p to focus even more

The x-axis typically starts at 0.4, not 0.0, because values below 0.5 are rarely used in practice.

**How to read the Repetition Penalty chart:**

Many bars should be at 1.0 (no penalty). Only tasks with known loop risks appear above 1.0. The
dashed line at 1.0 is the baseline. Bars extending past it indicate penalty is active.

Note which tasks have penalty > 1.0 and ask: why does this task have a loop risk? (Chatbots repeat
conversational patterns; creative writing can fall into phrase loops; summarization over long
documents may echo source material.)

---

### Row 3: "Final Distribution Shape per Task"

The bottom panel is a compact inline visualization showing the actual probability distribution shape
for each task after all parameters are applied.

**What you're looking at:**

Each task's distribution is shown as a small bar chart, positioned horizontally across the panel.
Tokens within each task are sorted by rank (highest probability leftmost).

- X-axis position: groups distributions by task (7 tasks spread evenly)
- Bar height: probability of each token (within each task's mini-chart)
- Bar color: the task's color (matches the scatter plot)
- "H=X.X" label above each task: the final entropy value
- Task name below: orientation labels

**How to read the shapes:**

```
Classification distribution:
  One very tall bar, then all tiny bars
  → Spike shape = low entropy, high confidence

Summarization distribution:
  A few medium bars declining gradually
  → Staircase shape = moderate entropy, competing options

Code distribution:
  Many bars of similar height
  → Flat plateau shape = high entropy, genuine uncertainty

Creative distribution:
  Moderate bars with slight taper
  → Gentle slope shape = high entropy, diverse options
```

**Reading the entropy labels (H=X.X):**

Compare these values to the predicted ranges:

- Classification should show H < 1.0
- Summarization should show H around 1.5–2.0
- Code should show H around 2.0–2.5
- Creative should show H around 2.0–2.8

If a task's actual entropy is outside its predicted range, trace back through the pipeline to find
which parameter interaction caused the surprise.

---

## Step 4: The Calibration Heuristic — A Complete Walkthrough

The experiment provides this 5-step process:

```
1. Start with temperature (deterministic vs creative goal)
2. Add top-p (0.9 baseline; reduce for precision)
3. Add top-k (hard rank cutoff if needed)
4. Add repetition penalty (only if looping appears; start at 1.1)
5. Validate with your task metric
```

Let's walk through a concrete example: you're building a chatbot for a financial services company.

**Step 1 — Define the goal:**

The chatbot should sound natural and conversational, give accurate information, and not repeat
itself. It's neither fully deterministic (you want varied phrasing) nor fully exploratory (you can't
hallucinate financial facts).

That description places it roughly at T=0.6–0.8.

Start: T=0.7

**Step 2 — Add top-p:**

0.9 is the standard baseline. For financial services, you want to avoid highly unusual phrasing that
might sound unprofessional. Tighten slightly.

Set: p=0.85

**Step 3 — Add top-k:**

Is there a hard upper limit on word choice diversity you want to enforce? For a professional
chatbot, using a wide variety of rare words is actually undesirable. A ceiling of k=50 is
reasonable.

Set: k=50

**Step 4 — Add repetition penalty:**

You don't know yet if looping is a problem. Start with no penalty.

Set: rep=1.0

**Step 5 — Validate:**

Run 100 sample outputs. Measure:

- Any repetitive loops? (No → keep rep=1.0; Yes → add rep=1.1)
- Outputs sound too robotic/formulaic? (Yes → increase T to 0.8)
- Outputs occasionally produce unusual/unprofessional language? (Yes → tighten p to 0.8)
- Entropy consistently too low (< 1.3) or too high (> 2.2)? → Adjust T first, then p

**The iterative loop:**

```
Define goal → Set initial params → Run samples → Measure → Adjust → Repeat
```

This loop usually converges in 3–5 iterations for a new task. Document every configuration and its
measured entropy so you can reverse course if a change makes things worse.

---

## Step 5: The Entropy-Driven Tuning Cheat Sheet — Explained

The experiment gives this shorthand:

```
Need lower entropy: decrease temperature, tighten top-p
Need higher entropy: increase temperature, loosen or disable filters
```

Here is the full reasoning behind these rules:

### Decreasing Entropy

Entropy is high when many tokens have similar probabilities. To reduce it:

1. **Decrease temperature first**: This is the most powerful lever. Even a T change from 0.7 to 0.3
   can more than halve entropy. Temperature affects all tokens simultaneously.

2. **Then tighten top-p**: If temperature alone isn't enough, reduce p from 0.9 to 0.7 or 0.6. This
   cuts the tail, which concentrates probability on the top tokens and reduces entropy further.

3. **Add top-k only as a last resort**: Top-k doesn't necessarily reduce entropy as much as the
   other two — if the distribution is flat within the top-k, entropy can still be high.

### Increasing Entropy

Entropy is low when one token dominates. To increase it:

1. **Increase temperature first**: Going from T=0.2 to T=0.7 can dramatically increase entropy by
   reducing the dominant token's probability advantage.

2. **Loosen top-p**: If p=0.7, increasing it to 0.9 or 1.0 allows more tokens into the nucleus,
   increasing entropy.

3. **Remove or increase top-k**: If k=10, increasing to k=50 allows more tokens to compete.

4. **Note**: If entropy is still too low after maximizing all three, the issue may be in the model
   or the prompt, not the sampling parameters. A model with extreme logit gaps (like classification
   at baseline) will have low entropy no matter what moderate parameters you set — you'd need T >
   2.0 to significantly reduce its confidence, and that typically breaks coherence.

---

## Step 6: The Real-World Checklist — Why Each Item Matters

The experiment provides a six-item checklist. Here's the reasoning behind each item:

```
[ ] Define your task success metric first
```

If you don't know what "good" means, you can't tell whether your parameter changes helped or hurt. "
Accuracy@1" (correct answer is the first one generated) is the metric for classification. "ROUGE
score" or human preference is the metric for summarization. "pass@k" (at least 1 of k generations
passes test cases) is the metric for code. Write down the metric before touching any parameter.

```
[ ] Start from a baseline (T=0.7, top_p=1.0, no penalties)
```

This baseline produces moderate behavior for most tasks. Starting here gives you a reference point.
If T=0.7 with no filters produces completely unusable output, the problem is likely in your model or
prompt, not in sampling parameters. Starting from extremes (T=0.1 immediately) makes it harder to
diagnose root causes.

```
[ ] Sweep one variable at a time
```

If you change T and p simultaneously and output quality improves, you don't know which change caused
it. You might even find that one change helped and the other hurt — but their combined effect looked
like an improvement. Always isolate variables.

```
[ ] Log prompts, settings, outputs, and metrics
```

Without logs, you'll repeat experiments you already ran and forget which configurations failed. A
simple spreadsheet with columns [prompt_sample, T, p, k, rep, entropy, metric_score] is enough.
After 20 runs, patterns become obvious.

```
[ ] Compare against a control configuration
```

Every experiment should have a control. The control is typically your current production config or
the baseline (T=0.7, p=1.0). Each new configuration should be compared to the control, not just to
an abstract "good." This prevents you from optimizing against a good previous configuration and
accidentally regressing.

```
[ ] Iterate toward your target behavior
```

Parameter tuning is iterative, not solved in one step. Your first configuration will not be optimal.
Each iteration should move entropy or output quality measurably in the right direction. If two
iterations in a row don't improve your metric, reconsider whether the parameters are even the right
lever — it might be the prompt, the model, or the evaluation method.

---

## Step 7: What the Entropy Bands Tell You in Practice

The experiment defines three entropy bands:

```
0–1.0:   extraction/classification style
1.5–2.0: balanced summarization/QA style
2.2+:    exploratory/creative style
```

Here's how to use these bands in the field:

### Band 1: Entropy 0–1.0 (Extraction/Classification)

When your output falls in this band, the model is producing near-deterministic output. One or two
tokens dominate every generation.

**Good signs**: Consistent answers to factual questions, correct classification labels, extracted
fields matching source text.

**Bad signs**: Outputs feel formulaic and robotic. Every response begins the same way. Model seems "
stuck" on a few tokens. Long generation starts looping.

**Adjustment**: If bad signs appear at this entropy level, the issue is usually T being too low.
Raise T by 0.1–0.2 and re-measure. If the task genuinely requires low entropy (true classification),
accept the robotic quality — it's a feature, not a bug.

### Band 2: Entropy 1.5–2.0 (Balanced)

This is the "Goldilocks zone" for most production language tasks. The model shows variety but
remains coherent.

**Good signs**: Responses feel naturally varied. Summaries use different phrasing for similar
content. Chatbot responses don't feel scripted.

**Bad signs**: Occasionally an unusual word or tone appears that feels off. Some responses are
better than others.

**Adjustment**: This is normal. If the "off" responses are acceptable, stay here. If they're not,
tighten p by 0.05–0.10 to slightly constrain the tail. If responses feel too uniform, loosen p or
raise T slightly.

### Band 3: Entropy 2.2+ (Exploratory/Creative)

High entropy, high diversity. Multiple quite different outputs across generations.

**Good signs**: Brainstorming produces genuinely distinct ideas. Creative writing has unexpected but
interesting directions. Code generation produces structurally different solutions on each run.

**Bad signs**: Some outputs are clearly off-topic or incoherent. Long generation occasionally goes
off the rails.

**Adjustment**: If bad outputs appear more than ~10% of the time, tighten p slightly. If you're
doing creative writing and can tolerate occasional misfires (just regenerate), this band is
appropriate. For code with pass@k evaluation, this is often intentional — you want diverse enough
solutions that at least one passes.

---

## Step 8: The Optional Challenge — How to Design Your Own Distribution

The experiment suggests creating your own logit distribution that matches your task's uncertainty.
Here's how to think through it systematically:

**Question 1: How confident is a well-trained model on this task?**

- Very confident (one clearly right answer): put one logit at 3.0+, others below 1.0
- Moderately confident (a few good options): spread top logits across 1.5–2.5
- Uncertain (many valid options): linear or near-linear decline starting around 1.5

**Question 2: How "catastrophic" is a wrong answer?**

- Very bad (medical diagnosis, legal classification): start with high confidence logits, use low T
- Moderate (customer support routing): moderate logits, moderate T
- Not bad (creative brainstorming): flat logits, high T

**Question 3: Is there a "tail" of genuinely invalid tokens?**

- Yes (code has syntax that can never be valid next): use top-k or top-p to cut the tail
  aggressively
- No (creative writing, all tones have some merit): minimal filtering

Use these answers to construct logits, then predict entropy before running the experiment. If your
prediction is significantly off, trace through the pipeline to find where your mental model was
wrong. This is how intuition deepens.

---

## Step 9: Common Misconceptions — Cleared Up

**Misconception 1: "The task table gives you the correct settings"**

The table gives you starting points informed by common practice. Your specific model, fine-tuning
data, prompt template, and hardware can all shift the optimal values. A classification model that
was fine-tuned on 50,000 examples might produce logit gaps of 5+ even at T=1.0 — meaning T=0.2 is
overkill. A weaker base model for the same task might need T=0.1 to force reliable classification.
Always validate against your task metric.

**Misconception 2: "Lower entropy is always safer"**

For precision tasks, yes. But for generation tasks, entropy that's too low causes its own problems —
looping, formulaic output, failure to generate diverse examples. "Safe" is task-dependent.
Entropy=0.5 is safe for classification and catastrophic for creative writing.

**Misconception 3: "These entropy bands are universal"**

They're heuristics based on common English-language tasks with standard model sizes. Multilingual
models, very small models, heavily fine-tuned models, and highly specialized domains may have
different entropy profiles. Always measure your baseline before assuming the bands apply.

**Misconception 4: "I can just use the same settings for all my tasks"**

Many developers do this because it's simple — T=0.7, p=0.9 for everything. This is actually quite
good for general-purpose assistants. But for specialized deployments (a code-only chatbot, a
classification-only system, a creative writing assistant), task-specific tuning meaningfully
improves quality. The time investment in running the calibration process in Step 4 pays off.

**Misconception 5: "Experiment 6 is about finding the optimal settings"**

Not quite. It's about developing a *process* for finding good settings. The optimal settings change
depending on your specific model, your data distribution, and your evaluation metric. What
Experiment 6 gives you is the framework to approach any new task confidently — not a lookup table to
memorize.

---

## Step 10: What You Have Accomplished — A Complete Picture

After all six experiments, here is what you now understand:

| Parameter              | What it does                                                              | When to use                                                                 |
|------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Temperature**        | Reshapes the entire probability distribution by scaling logits            | Always set this first; master dial for creativity vs precision              |
| **Top-p**              | Cuts the probability tail; adapts nucleus size to distribution confidence | Default: 0.9 for most tasks; reduce for precision, increase for exploration |
| **Top-k**              | Fixed-rank window; hard ceiling on token count                            | Use as safety ceiling in combination with top-p; solo use for simplicity    |
| **Repetition penalty** | History-dependent pressure against recently-seen tokens                   | Add only when looping is observed; start at 1.1; rarely exceed 1.5          |

**The pipeline order:**

```
Raw logits → Temperature → Top-k → Top-p → Repetition penalty → Sample
```

**The decision process:**

```
1. Define task → 2. Set T → 3. Set p → 4. Set k → 5. Set rep → 6. Measure → 7. Iterate
```

**The entropy bands:**

```
0–1.0   → Extraction/classification behavior
1.5–2.0 → Balanced generation behavior
2.2+    → Exploratory/creative behavior
```

---

## Quick Reference: Graph Interpretation Cheat Sheet

When you open `exp6_usecases.png`, scan in this order:

```
1. Look at the task map scatter plot (row 1, left)
   → Which quadrant is each task in?
   → Does the position match your intuition about that task?
   → Are similar tasks clustered together?

2. Look at the preset table (row 1, right)
   → Scan the T column: does it increase from Classification to Brainstorming?
   → Scan the p column: where do values deviate from the 0.9 default?
   → Scan the k column: which tasks use a non-zero k and why?

3. Look at the parameter bar charts (row 2, three panels)
   → T bars: should increase monotonically from Classification to Brainstorming
   → p bars: mostly similar, with slight deviations for edge tasks
   → rep bars: mostly at 1.0, with active penalties only for loop-prone tasks
   → The yellow dashed line at T=0.7 is the center of gravity — tasks above and below it

4. Look at the distribution shape panel (row 3)
   → Find each task's mini distribution
   → Match the shape to its entropy label: H=X.X
   → Classification: spike shape (low H)
   → Code/Creative: plateau or gentle slope (high H)
   → Verify the entropy progression matches predictions
```

---

## Final Summary

| Concept                              | One-sentence explanation                                                          |
|--------------------------------------|-----------------------------------------------------------------------------------|
| Task-first thinking                  | Define what "good output" means before choosing any parameters                    |
| Classification dist.                 | One token dominates; parameters should preserve and reinforce this                |
| Summarization dist.                  | Several tokens compete; parameters should select the best few while cutting junk  |
| Code generation dist.                | Many valid continuations; parameters should allow diversity with a guard rail     |
| Creative dist.                       | Broad competition with slight preferences; parameters should preserve exploration |
| Entropy as compass                   | Match target entropy band to task type before tuning individual parameters        |
| Calibration process                  | Define metric → set T → add p → add k → add rep → measure → iterate               |
| The "no universal setting" principle | Optimal parameters are task-dependent; presets are starting points, not answers   |

> **Final takeaway**: The parameter intuition you've built across six experiments now transfers to
> every LLM task you encounter — across different providers, model families, and application types.
> The specific numbers will change; the process and the principles won't.

---

*End of the 6-Experiment LLM Parameter Learning Path*  
*Previous: Experiment 5 — Repetition Penalties*  
*You've completed the full series.*