# Preset sweeps (4 enterprise use cases)

Presets are opinionated defaults designed to be a **starting point**. Each preset includes:

- default parameter settings
- a recommended prompt pattern
- evaluation metrics aligned to the use case

Presets in this repo:

- `summarization`
- `rag_qa`
- `classification`
- `code_generation`

Run with:

```bash
python scripts/run_sweep_rest.py --preset rag_qa --trials 5
```

Then analyze:

```bash
python scripts/analyze_results.py --input runs/latest.jsonl
```

--8<-- "_abbreviations.md"
