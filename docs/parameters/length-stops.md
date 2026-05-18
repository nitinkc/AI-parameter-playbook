# Length, stops & cost controls

## `max_tokens` / `max_completion_tokens`

- `max_tokens` limits the number of generated tokens and is supported by Azure endpoints.
- OpenAI’s reference also documents `max_completion_tokens` as the preferred modern parameter, and notes `max_tokens` may be deprecated / incompatible with some reasoning models.

Use these to:

- bound cost
- bound latency
- enforce concise answers

## `stop`

- Up to a small number of stop sequences can end generation.
- Useful for code blocks, list truncation control, or multi-part outputs.

## Useful executive takeaway

**Length limits are one of the strongest cost controls** you have, independent of decoding randomness.

Next: **Determinism & Debugging**.

--8<-- "_abbreviations.md"
