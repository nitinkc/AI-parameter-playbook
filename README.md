# AI Parameter Playbook — Learn LLM Parameters

## What is this?

A learning-focused playbook to understand LLM parameters through hands-on experiments.
Start free and local (no API calls), then move to cloud validation if needed.

This is the absolutely minimal path to understanding LLM parameters.

## Install

Create a virtual environment 
```shell
python3 -m venv .venv
```

and install dependencies:
```bash
source .venv/bin/activate
# Install experiment dependencies
pip install -r requirements.txt
```

## View docs locally (MkDocs only)
**Easiest way:**

```bash
./serve-docs.sh
```

## Run the Simulator

```bash
python3 notebooks/sampling_simulator.py
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

## Read the Overview

Open this file in your browser or editor:

```
docs/learning/overview.md
```

Then read: [docs/learning/overview.md](docs/learning/00-overview.md)

It explains:
- What the 6 experiments are
- Why each matters
- How long each takes

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

Outputs are stored as JSONL under `runs/`.

---

## Key Insights You'll Gain

| Parameter          | What It Does                        | Range   | When to Use                    |
|:-------------------|:------------------------------------|:--------|:-------------------------------|
| Temperature        | Probability distribution smoothness | 0.0-2.0 | Always relevant                |
| Top-p              | Max cumulative probability nucleus  | 0-1.0   | Modern systems (OpenAI, local) |
| Top-k              | Max number of tokens to consider    | 0-50+   | Legacy systems, or with top-p  |
| Repetition Penalty | Discourage repeated tokens          | 1.0-2.0 | When you see repetition loops  |

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

| Task            | Temp    | Top-p  | Top-k | Typical Entropy  |
|:----------------|:--------|:-------|:-----:|:----------------:|
| Classification  | 0.1-0.3 | 1.0    |   0   |     0.4-0.8      |
| Extraction      | 0.1-0.3 | 1.0    |   0   |     0.4-0.8      |
| Summarization   | 0.6-0.8 | 0.95   |   0   |     1.5-1.8      |
| RAG Q&A         | 0.2-0.4 | 1.0    |   0   |     0.8-1.2      |
| Code Generation | 0.8     | 1.0    |  30   |     2.0-2.3      |
| Brainstorming   | 0.9-1.2 | 0.85   |   0   |     1.8-2.2      |
| Story Writing   | 1.0-1.3 | 0.9    |   0   |     2.1-2.4      |

**Pattern:** Lower entropy = safer, higher entropy = more creative.


## One More Thing

The most important insight:

> **Parameters are not magic. They're probability redistribution.**
> 
> When you see "high temperature = creative", what's really happening:
> - Low T → logits / 0.1 → huge differences in softmax → peaked distribution → deterministic
> - High T → logits / 1.5 → tiny differences in softmax → flat distribution → creative

Once you grok this, every parameter becomes predictable.


Happy learning! 🚀



