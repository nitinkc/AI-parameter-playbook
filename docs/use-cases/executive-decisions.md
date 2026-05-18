# Executive decision guide (defaults that survive production)

This page turns parameter tuning into decisions you can defend.

## 1) Choose the objective first

- **Reliability / compliance** → lower temperature, structured outputs, short max_tokens
- **Ideation / exploration** → higher temperature and/or modest presence_penalty
- **Cost control** → strict max_tokens, cached prompts where supported

## 2) Prefer structure over “prompt yelling”

Structured outputs (`response_format: json_schema`) reduce parsing errors and produce predictable payloads for downstream workflows.

## 3) Manage variance explicitly

- Use `seed` to measure output variance in staging.
- Establish acceptable variance thresholds (e.g., label stability > 99%).

## 4) Decide who owns the defaults

- Central platform team owns default presets.
- Product teams own deviations (with an experiment log).

## 5) Watch for drift

Backend/model updates can change behavior. Maintain a regression suite for:

- top prompts per use case
- refusal criteria
- JSON validity

## 6) Recommended defaults (starting point)

These are safe initial defaults you can adapt:

- Summarization: temp 0.2, schema output, max_tokens bounded
- RAG Q&A: temp 0.0–0.2, schema with citations
- Classification: temp 0, enum schema
- Code: temp 0.2, bounded max_tokens, CI tests

Pair these with a measurable evaluation plan.

--8<-- "_abbreviations.md"
