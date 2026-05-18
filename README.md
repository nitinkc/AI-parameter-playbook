# AI Parameter Playbook — Learn LLM Parameters (Free & Local-First)

## What is this?

A learning-focused playbook to understand LLM parameters through hands-on experiments.
Start free and local (no API calls), then move to cloud validation if needed.

## Dependency split

- Docs site only: `requirements-docs.txt`
- Experiments/simulators/scripts: `requirements-experiments.txt` (also mirrored in `requirements.txt`)

## Quick Start: local simulator (no cloud)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-experiments.txt
python3 experiments/local/sampling_simulator.py
```

## View docs locally (MkDocs only)

**Easiest way:**

```bash
./serve-docs.sh
```

Open http://localhost:8000 and navigate to Learning Path and Playgrounds.

**Manual way (if you prefer):**

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
pip install -r requirements-docs.txt
mkdocs serve
```


## Optional: cloud experiments (Azure OpenAI)

```bash
python3 -m venv .venv-exp
source .venv-exp/bin/activate
pip install -r requirements-experiments.txt
cp .env.example .env
python3 scripts/run_sweep_rest.py --preset summarization --trials 3
python3 scripts/analyze_results.py --input runs/latest.jsonl
```

Outputs are stored as JSONL under `runs/`.


## Current Structure

**Docs (pure markdown + config, no code):**
- `docs/` – only `.md` files and assets
- `mkdocs.yml` – docs configuration
- `requirements-docs.txt` – mkdocs + material deps only
- `serve-docs.sh` – helper script to launch docs server

**Experiments (all runnable code):**
- `experiments/local/sampling_simulator.py` – local parameter simulator
- `scripts/run_sweep_rest.py`, `analyze_results.py` – cloud experiment runners
- `requirements-experiments.txt` – numpy, matplotlib, requests, etc.
- `requirements.txt` – mirrors `requirements-experiments.txt`

**Shared/top-level:**
- `mkdocs.yml` – still documents code *behavior*, not the code itself

## Quick Commands

### View docs
```bash
./serve-docs.sh
# or manually:
python3 -m venv .venv-docs
source .venv-docs/bin/activate
pip install -r requirements-docs.txt
mkdocs serve
```

### Run simulator
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-experiments.txt
python3 experiments/local/sampling_simulator.py
```

### Run cloud experiments
```bash
python3 -m venv .venv-exp
source .venv-exp/bin/activate
pip install -r requirements-experiments.txt
cp .env.example .env
python3 scripts/run_sweep_rest.py --preset summarization --trials 3
```
