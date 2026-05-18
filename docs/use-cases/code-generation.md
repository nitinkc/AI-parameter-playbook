# 4) Code Generation

## Goal

Generate code that compiles/lints, follows your conventions, and limits hallucinated imports/APIs.

## Recommended baseline settings

- `temperature`: 0–0.3
- `top_p`: 1
- `frequency_penalty`: 0–0.2 (optional)
- `max_tokens`: bounded to prevent runaway output
- `stop`: helpful when you require *only code* (e.g., stop on triple backticks)

## Prompt pattern

- Provide a clear spec, inputs/outputs, and constraints.
- Ask for a single file.
- If you need strict formatting, use structured outputs to return `{language, code, notes}`.

## How to test

- Run code via CI (unit tests / type checks)
- Measure pass rate across parameter sweeps
- Track defect types (syntax errors vs logic errors)

For executive decisions, emphasize **automated evaluation**.

--8<-- "_abbreviations.md"
