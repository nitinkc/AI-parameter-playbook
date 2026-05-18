# Determinism & debugging

## `seed`

Both Azure and OpenAI documentation describe `seed` as a *best effort* toward deterministic sampling when all other parameters remain the same; determinism is not guaranteed.

## `logprobs` and `top_logprobs`

OpenAI’s chat completions reference includes:

- `logprobs` (whether to return token log probabilities)
- `top_logprobs` (how many alternatives to return per position)

Azure OpenAI’s REST reference also documents `logprobs` and `top_logprobs` in the chat completions request body.

## Why executives should care

- `seed` helps you quantify **variance risk** in regulated settings.
- `logprobs` helps you build **confidence scoring**, detect ambiguities, and design retries.

Next: **Structured Outputs & Tools**.

--8<-- "_abbreviations.md"
