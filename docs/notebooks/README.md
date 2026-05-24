# Local Experiments

This folder contains runnable experimentation assets kept outside `docs/`.

## Files

- `sampling_simulator.py` - Local parameter simulation runner (temperature, top-p, top-k, penalties)
- `visualization_temperature.ipynb` - Entropy vs temperature notebook
- `visualization_top_p.ipynb` - Top-p effect notebook
- `visualization_top_k.ipynb` - Top-k effect notebook
- `../experiment_harness.py` - Batch sweep harness that exports CSV/JSON artifacts

## Run Simulator

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 notebooks/sampling_simulator.py
```

## Run Harness

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 experiment_harness.py --preset combined --samples 10000 --format both
```

Useful preset examples:

```bash
python3 experiment_harness.py --preset temperature_sweep
python3 experiment_harness.py --preset top_p_sweep
python3 experiment_harness.py --preset top_k_sweep
python3 experiment_harness.py --preset repetition_sweep
```

## Run Visualizations (Notebooks)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m ipykernel install --user --name ai-parameter-playbook --display-name "Python (ai-parameter-playbook)"
jupyter lab
```

Then open and run:

- `experiments/visualization_temperature.ipynb`
- `experiments/visualization_top_p.ipynb`
- `experiments/visualization_top_k.ipynb`

