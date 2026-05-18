# 1) Summarization (enterprise)

## Goal

Produce concise, faithful summaries with low variance.

## Typical risks

- hallucinated facts
- missing critical details
- overlong output

## Recommended baseline settings

- `temperature`: 0.0–0.3
- `top_p`: 1.0 (tune only if you choose nucleus sampling)
- `presence_penalty`: 0
- `frequency_penalty`: 0–0.3 (optional)
- `max_tokens`: set pretty low (e.g., 128–512) depending on policy

## Strong pattern: structured summary schema

Use `response_format` with `json_schema` so your summary always includes the fields leadership cares about.

Example schema:

```json
{
  "name": "Summary",
  "strict": true,
  "schema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "title": {"type": "string"},
      "key_points": {"type": "array", "items": {"type": "string"}},
      "risks": {"type": "array", "items": {"type": "string"}},
      "next_steps": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["title", "key_points", "risks", "next_steps"]
  }
}
```

## How to test

Sweep temperature and max_tokens first; measure:

- summary length compliance
- JSON validity (if structured)
- human faithfulness score

Related docs: Parameters → Decoding, Structured Outputs; Experiments → Presets.

--8<-- "_abbreviations.md"
