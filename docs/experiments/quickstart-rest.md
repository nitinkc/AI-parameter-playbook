# Quickstart: REST sweep runner

This runner calls Azure OpenAI Chat Completions via the documented REST endpoint:

```
POST https://{endpoint}/openai/deployments/{deployment-id}/chat/completions?api-version=2024-10-21
```

and passes parameter knobs like `temperature`, `top_p`, `presence_penalty`, etc.

## Steps

1. Create `.env` from `.env.example` and fill in your endpoint/key/deployment.
2. Install dependencies.
3. Run a preset sweep.

```bash
pip install -r requirements.txt
cp .env.example .env
python scripts/run_sweep_rest.py --preset summarization --trials 3
```

Outputs are written to `runs/YYYYMMDD_HHMMSS_<preset>.jsonl` and a symlink-like pointer `runs/latest.jsonl`.

## Safety note

Use non-sensitive sample data. If you test on enterprise documents, make sure your logging complies with your org's policies.

--8<-- "_abbreviations.md"
