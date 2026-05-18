# Local Experiments

This folder contains runnable experimentation assets kept outside `docs/`.

## Files

- `sampling_simulator.py` - Local parameter simulation runner (temperature, top-p, top-k, penalties)
- `visualization_temperature.ipynb` - Entropy vs temperature notebook
- `visualization_top_p.ipynb` - Top-p effect notebook
- `visualization_top_k.ipynb` - Top-k effect notebook

## Run simulator (script)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-experiments.txt
python3 experiments/local/sampling_simulator.py
```

## Run visualizations (notebooks)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-experiments.txt
python3 -m ipykernel install --user --name ai-parameter-playbook --display-name "Python (ai-parameter-playbook)"
jupyter lab
```

Then open and run:

- `experiments/local/visualization_temperature.ipynb`
- `experiments/local/visualization_top_p.ipynb`
- `experiments/local/visualization_top_k.ipynb`

## Dependency split

- Docs-only dependencies: `requirements-docs.txt`
- Experiment/runtime dependencies: `requirements-experiments.txt`
