# LLM Sampling Parameters: A Deep Intuition Guide

## What Happens During Generation?

When an LLM generates a token, it:

1. Runs a forward pass and outputs **raw logits** (one value per vocabulary token)
2. Applies **temperature scaling** to reshape the distribution
3. Applies **top-k** and/or **top-p** filters to prune candidates
4. Applies **repetition controls** to discourage loops
5. **Samples** from the final distribution

Each parameter acts at a different stage, so understanding *where* it acts matters as much as understanding *what* it does.

```text
Raw Logits -> [Temperature] -> [Top-k] -> [Top-p] -> [Repetition Penalty] -> Sample
```

## Why This Learning Path Exists

You have probably seen statements like:

- **Temperature**: "Higher = more creative"
- **Top-p**: "Dynamic nucleus sampling"
- **Top-k**: "Keep top-k tokens"
- **Repetition penalty**: "Reduce repetition"

Those are directionally true, but not enough for reliable tuning. The way to build intuition is to **test parameters systematically**.

## 6-Experiment Learning Path

| #  | Experiment           | Focus                                                 | Time   | Takeaway                                                |
|:---|:---------------------|:------------------------------------------------------|:-------|:--------------------------------------------------------|
| 1  | Temperature Sweep    | How does randomness change with temperature?          | 5 min  | Temperature directly reshapes probability distributions |
| 2  | Top-p (Nucleus)      | How does top-p restrict candidates?                   | 5 min  | Top-p selects a *dynamic* set of tokens                |
| 3  | Top-k                | How does top-k differ from top-p?                     | 5 min  | Top-k is a *fixed-size* ranked cutoff                  |
| 4  | Combined Filters     | What if you mix temperature + top-p + top-k?          | 10 min | Order matters; interactions can surprise you           |
| 5  | Repetition Penalties | How do penalties discourage repeated tokens?          | 5 min  | Penalties reduce repetition pressure before sampling   |
| 6  | Real Use Cases       | How do you tune for classification vs. summarization? | 15 min | Different tasks need different strategies              |

**Total time: ~45 minutes** for practical parameter intuition.

## How the Experiments Work

Each experiment uses a **local Python simulator** that:

1. Starts from fake logits (simulated next-token odds)
2. Applies parameter transformations
3. Samples many times to estimate the resulting distribution
4. Reports metrics like entropy and top-token probability

Why this still works: the decoding math is the same regardless of whether you use a toy simulator or a production API.

## What You Will Learn

After the path, you should be able to:

- Explain what each parameter does (intuition + math)
- Predict interactions (for example, temperature with top-p)
- Diagnose common failures (loops, over-randomness, over-determinism)
- Choose task-appropriate presets (classification vs. brainstorming)
- Transfer the same logic to cloud/local model providers

## Quick Start

```bash
# Optional virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the local simulator
python3 notebooks/sampling_simulator.py
```

Example output:

```text
=== Very deterministic (temp=0.2) ===
Entropy: 1.234
approve       0.851
reject        0.095
review        0.032
...
```

## Ready to Begin?

👉 **Start with [Experiment 1: Temperature Sweep](01-temperature.md)**

## Optional: Move to Cloud Experiments

Once you understand fundamentals locally, test against real APIs:

- **Compare platforms first**: [Multi-Cloud Overview](../cloud/overview.md)
- **Azure OpenAI**: [Setup](../setup/overview.md)
- **OpenAI**: [Setup](../cloud/openai/setup.md)
- **Google Vertex AI**: [Setup](../cloud/google/setup.md)
- **Amazon Bedrock**: [Setup](../cloud/amazon/setup.md)

## Quick Reference Card

```text
╔══════════════════════════════════════════════════════════════╗
║              LLM PARAMETER CHEAT SHEET                      ║
╠══════════════╦═══════════════╦═══════════════════════════════╣
║ Parameter    ║ Range         ║ Rule of Thumb                 ║
╠══════════════╬═══════════════╬═══════════════════════════════╣
║ Temperature  ║ 0.0 - 2.0     ║ 0.2=precise, 0.7=balanced,    ║
║              ║               ║ 1.0=creative                  ║
╠══════════════╬═══════════════╬═══════════════════════════════╣
║ Top-p        ║ 0.0 - 1.0     ║ 0.9 default; lower=focused    ║
╠══════════════╬═══════════════╬═══════════════════════════════╣
║ Top-k        ║ 1 - inf       ║ 40-50 precise; 100+ diverse   ║
╠══════════════╬═══════════════╬═══════════════════════════════╣
║ Rep. Penalty ║ 1.0 - 2.0     ║ 1.0=off; 1.2=light; 1.5=hard  ║
╚══════════════╩═══════════════╩═══════════════════════════════╝
```

## Further Reading

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [Sampling Methods for Language Models](https://huggingface.co/blog/how-to-generate)
- [Temperature Scaling for Calibration](https://arxiv.org/abs/1706.04599)

--8<-- "_abbreviations.md"
