# Interactive Playgrounds: Run Experiments Locally

This section is the hands-on track: editable Python workflows you can run without cloud credentials.

## Overview

You now have two local tools:

1. **`experiments/sampling_simulator.py`** - interactive simulator for parameter intuition
2. **`experiment_harness.py`** - batch sweep runner that exports CSV/JSON artifacts

Both run fully offline.

## Quick Start

### 1) Run the simulator

```bash
pip install -r requirements.txt
python3 notebooks/sampling_simulator.py
```

### 2) Run the experiment harness

```bash
pip install -r requirements.txt
python3 experiment_harness.py --preset combined --samples 10000 --format both
```

The harness writes files under `runs/local/` by default.

## Harness Presets

Use the `--preset` flag:

- `combined`
- `temperature_sweep`
- `top_p_sweep`
- `top_k_sweep`
- `repetition_sweep`

Examples:

```bash
python3 experiment_harness.py --preset temperature_sweep --temperatures 0.1,0.2,0.5,1.0,1.5,2.0
python3 experiment_harness.py --preset top_p_sweep --top-ps 0.2,0.5,0.8,0.95,1.0
python3 experiment_harness.py --preset top_k_sweep --top-ks 1,3,5,10,0
python3 experiment_harness.py --preset repetition_sweep --repetition-penalties 1.0,1.1,1.3,1.5,2.0
```

### Custom scenario file

Optionally pass your own token/logit set:

```json
{
  "tokens": ["token_a", "token_b", "token_c"],
  "logits": [1.5, 1.1, 0.2]
}
```

Run with:

```bash
python3 experiment_harness.py --scenario-file my_scenario.json --preset combined
```

## Artifact Outputs

Each harness run reports:

- Run name and parameter values
- Entropy
- Top token and top-token probability
- Viable token count

Artifacts include:

- `harness_<preset>_<timestamp>.csv`
- `harness_<preset>_<timestamp>.json`

## Simulator Customization

Edit `experiments/sampling_simulator.py` to change tokens/logits or add settings:

```python
settings = [
    dict(name='My custom experiment', temperature=0.7, top_k=5, top_p=0.85),
]
```

Penalty-focused example:

```python
settings = [
    dict(name='No penalty', repetition_penalty=1.0, sequence_len=0),
    dict(name='Mild (1.1x)', repetition_penalty=1.1, sequence_len=64),
    dict(name='Moderate (1.3x)', repetition_penalty=1.3, sequence_len=64),
    dict(name='Heavy (1.5x)', repetition_penalty=1.5, sequence_len=64),
]
```

## Suggested Learning Flow

1. Complete the 6-step learning path.
2. Re-run each concept with the harness preset.
3. Compare CSV/JSON artifacts across runs.
4. Move to cloud APIs only after local behavior is predictable.

## Troubleshooting

**Q: `ModuleNotFoundError: numpy`**  
A: Install dependencies:

```bash
pip install -r requirements.txt
```

**Q: Where are my harness outputs?**  
A: In `runs/local/` by default, or your custom `--output-dir`.

**Q: Can I use real model logits?**  
A: Yes, provide them via `--scenario-file` JSON.

👉 **Next: [Back to Learning Path](../learning/00-overview.md) or [Jump to Experiment 1](../learning/01-temperature.md)**

--8<-- "_abbreviations.md"
