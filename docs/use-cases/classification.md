# 3) Classification & Routing

## Goal

Return stable labels (intent, category, priority) with minimal variance.

## Recommended baseline settings

- `temperature`: 0
- `top_p`: 1
- `max_tokens`: small (e.g., 32)
- `response_format`: `json_schema` with an enum

Example schema (intent routing):

```json
{
  "name": "Intent",
  "strict": true,
  "schema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "intent": {"type": "string", "enum": ["procurement", "hr", "it", "finance", "other"]},
      "rationale": {"type": "string"}
    },
    "required": ["intent", "rationale"]
  }
}
```

## How to test

- Use a labeled dataset
- Measure accuracy / confusion matrix
- Add a regression test suite so upgrades don’t change routing behavior unexpectedly

This is one of the strongest areas for **executive confidence**, because structure + low randomness yields stable outputs.

--8<-- "_abbreviations.md"
