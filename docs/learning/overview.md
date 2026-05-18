# Learning Path: Master LLM Parameters (Free & Local-First)

Welcome! This learning path teaches you **exactly what each LLM parameter does** through hands-on experiments you can run on your laptop—no API keys, no cloud costs.

## The Problem

You've heard of parameters like:

- **Temperature**: "Higher = more creative"
- **Top-p**: "Dynamic nucleus sampling"
- **Top-k**: "Keep top-k tokens"
- **Repetition penalty**: "Reduce repetition"

But what does that *actually mean* and *how much* does it matter? The only way to truly understand is to **test systematically**.

## The Solution: 6-Experiment Learning Path

We've designed a progression where each experiment isolates **one concept** at a time, building your intuition systematically:

| # | Experiment | Focus | Time | Takeaway |
|---|------------|-------|------|----------|
| 1 | Temperature Sweep | How does randomness change with temperature? | 5 min | Temperature directly reshapes probability distributions |
| 2 | Top-p (Nucleus) | How does top-p restrict candidates? | 5 min | Top-p selects a *dynamic* set of tokens |
| 3 | Top-k | How does top-k differ from top-p? | 5 min | Top-k is a *fixed-size* ranked cutoff |
| 4 | Combined Filters | What if you mix temperature + top-p + top-k? | 10 min | Order matters; interactions can surprise you |
| 5 | Repetition Penalties | How do penalties discourage repeated tokens? | 5 min | Penalties are applied *before* sampling |
| 6 | Real Use Cases | How do you tune for classification vs. summarization? | 15 min | Different tasks need different strategies |

**Total time: ~45 minutes → deep intuition you can apply to any LLM**

## How It Works

Each experiment uses a **local Python simulator** that:

1. ✅ Starts with fake logits (next-token odds from a pretend model)
2. ✅ Applies parameter transformations (temperature, filters, penalties)
3. ✅ Samples thousands of times to measure the distribution
4. ✅ Shows you: entropy, top-token probability, number of viable candidates

**Why?** Because the math is identical whether you're using a $0 simulator or a million-parameter model. You learn the *mechanics*, not model-specific quirks.

## What You'll Learn

After this path, you'll understand:

- 🎯 **What each parameter *actually* does** (math + intuition)
- 📊 **How parameters interact** (temperature + top-p, etc.)
- 🔍 **How to diagnose parameter problems** ("Why is it outputting the same thing over and over?")
- 💡 **How to tune for different tasks** (creative vs. deterministic)
- 🚀 **How to migrate to real models** (Azure OpenAI, OpenAI, local LLMs, etc.)

## Prerequisites

- Python 3.8+
- Install experiment dependencies with `pip install -r requirements-experiments.txt`

That's it.

## Quick Start

```bash
# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install experiment dependencies
pip install -r requirements-experiments.txt

# Run the interactive simulator
python3 experiments/local/sampling_simulator.py
```

You'll see output like:

```
=== Very deterministic (temp=0.2) ===
Entropy: 1.234
approve       0.851
reject        0.095
review        0.032
...
```

## After Each Experiment

We provide:

- **The question you're exploring**
- **Exactly what to change in the simulator**
- **What metrics to watch**
- **An explanation of what you observed**

## Ready?

👉 **Start with Experiment 1: [Temperature Sweep →](01-temperature.md)**

Or jump to:
- [Experiment 2: Top-p →](02-top-p.md)
- [Experiment 3: Top-k →](03-top-k.md)
- [All Experiments →](#)

---

## Optional: Move to Cloud Experiments

Once you understand the fundamentals, you can optionally test against real models:

- **Compare platforms first** → [Multi-Cloud Overview →](../cloud/overview.md)
- **Azure OpenAI** → [Setup →](../setup/overview.md)
- **OpenAI** → [Setup →](../cloud/openai/setup.md)
- **Google Vertex AI** → [Setup →](../cloud/google/setup.md)
- **Amazon Bedrock** → [Setup →](../cloud/amazon/setup.md)
- **Local LLMs** (Ollama, llama.cpp) → Free tier with your own hardware

The parameters work the same way everywhere. You've already done the hard thinking locally!

---

## FAQ

**Q: Can I skip experiments?**  
A: We don't recommend it. The path is designed so each builds on the last. But if you're short on time, Experiments 1, 2, and 6 cover 80% of the insight in 20 minutes.

**Q: Will this teach me how a real model thinks?**  
A: No—we use fake logits to isolate parameter behavior. But the *parameter math* is identical in all models, so you're learning the core intuition.

**Q: Can I use this with other models (OpenAI, Anthropic, local LLMs)?**  
A: Yes! The parameters are consistent across APIs. Once you understand them locally, you can apply the tuning strategy anywhere.

**Q: How long until I can tune models confidently?**  
A: The path takes ~45 minutes. After that, you'll have a mental model that applies to any LLM task.

---

👉 **Next: [Experiment 1: Temperature Sweep →](01-temperature.md)**

--8<-- "_abbreviations.md"
