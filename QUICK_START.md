# ⚡ Quick Start: 7 Minutes to Parameter Mastery

This is the absolutely minimal path to understanding LLM parameters.

## Step 1: Install (1 minute)

```bash
# Navigate to your workspace
cd /Users/nchaur590@cable.comcast.com/Programming/ai-parameter-playbook

# Install experiment dependencies
pip install -r requirements-experiments.txt
```

## Step 2: Run the Simulator (2 minutes)

```bash
python3 experiments/local/sampling_simulator.py
```

You'll see output like:

```
EXPERIMENT 1: Temperature Sweep
How does temperature affect randomness?

--- Extremely deterministic (T=0.1) ---
Entropy: 0.095
approve      0.981

--- Very creative (T=2.0) ---
Entropy: 2.151
approve      0.217
```

**What you're seeing:**
- Lower temperature = higher probability on top token (more deterministic)
- Higher temperature = more spread out (more creative)

## Step 3: Read the Overview (2 minutes)

Open this file in your browser or editor:

```
docs/learning/overview.md
```

It explains:
- What the 6 experiments are
- Why each matters
- How long each takes

## Step 4: Pick Your Path (2 minutes)

### Path A: Quick Understanding (30 min)
1. [Experiment 1: Temperature](docs/learning/01-temperature.md)
2. [Experiment 2: Top-p](docs/learning/02-top-p.md)  
3. [Experiment 6: Use Cases](docs/learning/06-use-cases.md)

**Result:** Know what to tune and why

### Path B: Deep Mastery (90 min)
Do all 6 experiments in order:
1. Temperature
2. Top-p
3. Top-k
4. Combined
5. Penalties
6. Use cases

**Result:** Deep intuition + ability to design new experiments

### Path C: Theory First + Practice (60 min)
1. [Parameter Reference](docs/parameters/parameter-map.md)
2. [Decoding Controls](docs/parameters/decoding.md)
3. Run all 6 experiments

**Result:** Theory + practice foundation

---

## What You'll Understand

**After 30 minutes:**
- Temperature = how "random" to be
- Top-p = which tokens to consider
- Top-k = max how many tokens to consider
- How to tune for different tasks

**After 90 minutes:**
- *Why* each parameter works the way it does (math)
- How they interact (combinations)
- How to design parameters for any task
- How to diagnose "why is this output weird?"

---

## Next: Move to Cloud (Optional)

Once you understand parameters locally, test on real models:

```bash
# Set up Azure credentials
cp .env.example .env
# Fill in your Azure OpenAI endpoint and key

# Install cloud dependencies
pip install -r requirements.txt

# Run a preset experiment
python3 scripts/run_sweep_rest.py --preset summarization --trials 3

# Analyze results
python3 scripts/analyze_results.py --input runs/latest.jsonl
```

But you don't need this to understand parameters—everything you learned locally applies identically to real models.

---

## Key Insights You'll Gain

| Parameter | What It Does | Range | When to Use |
|-----------|-------------|-------|------------|
| Temperature | Probability distribution smoothness | 0.0-2.0 | Always relevant |
| Top-p | Max cumulative probability nucleus | 0-1.0 | Modern systems (OpenAI, local) |
| Top-k | Max number of tokens to consider | 0-50+ | Legacy systems, or with top-p |
| Repetition Penalty | Discourage repeated tokens | 1.0-2.0 | When you see repetition loops |

---

## Experiments at a Glance

```
Exp 1: Temp 0.1 → 2.0     Entropy: 0.1 → 2.1     (linear relationship)
Exp 2: Top-p 0.2 → 1.0    Entropy: 0.0 → 1.8     (cumulative filtering)
Exp 3: Top-k 1 → 10       Entropy: 0.0 → 1.8     (ranked filtering)
Exp 4: Real scenarios      Entropy: 0.6 → 2.5     (combined effects)
Exp 5: Penalties 1.0→2.0   Entropy: 1.8 → 2.0     (diversity increase)
Exp 6: Task tuning         Entropy: varies         (entropy is your dial)
```

---

## Real-World Presets

| Task | Temp | Top-p | Top-k | Typical Entropy |
|------|------|-------|-------|---|
| Classification | 0.1-0.3 | 1.0 | 0 | 0.4-0.8 |
| Extraction | 0.1-0.3 | 1.0 | 0 | 0.4-0.8 |
| Summarization | 0.6-0.8 | 0.95 | 0 | 1.5-1.8 |
| RAG Q&A | 0.2-0.4 | 1.0 | 0 | 0.8-1.2 |
| Code Generation | 0.8 | 1.0 | 30 | 2.0-2.3 |
| Brainstorming | 0.9-1.2 | 0.85 | 0 | 1.8-2.2 |
| Story Writing | 1.0-1.3 | 0.9 | 0 | 2.1-2.4 |

**Pattern:** Lower entropy = safer, higher entropy = more creative.

---

## Troubleshooting

**Q: Script says "numpy not found"**  
A: `pip install numpy`

**Q: I want to modify the experiments**  
A: Edit `experiments/local/sampling_simulator.py`, change the `settings` list

**Q: Where's the math?**  
A: [Simulator Guide](docs/playgrounds/simulator-guide.md) has all formulas

**Q: Can I use this with OpenAI / Anthropic / local LLMs?**  
A: Yes! The parameters work identically. Only the API differs.

---

## The Journey

```
Minute 0:   (you are here)
Minute 1:   pip install
Minute 3:   Simulator output (wow, entropy!)
Minute 7:   Read overview
Minute ~45: Complete 6 experiments
Minute 50:  Understand parameters at 90th percentile mastery
Hour 2:     (optional) Test on real cloud models
```

---

## Go Deeper

- **Learn Path:** [docs/learning/overview.md](docs/learning/overview.md)
- **Simulator:** [docs/playgrounds/simulator-guide.md](docs/playgrounds/simulator-guide.md)
- **Parameter Reference:** [docs/parameters/parameter-map.md](docs/parameters/parameter-map.md)
- **Cloud Integration:** [docs/setup/overview.md](docs/setup/overview.md)

---

## One More Thing

The most important insight:

> **Parameters are not magic. They're probability redistribution.**
> 
> When you see "high temperature = creative", what's really happening:
> - Low T → logits / 0.1 → huge differences in softmax → peaked distribution → deterministic
> - High T → logits / 1.5 → tiny differences in softmax → flat distribution → creative

Once you grok this, every parameter becomes predictable.

---

**Ready?**

```bash
python3 experiments/local/sampling_simulator.py
```

Then read: [docs/learning/overview.md](docs/learning/overview.md)

Happy learning! 🚀

