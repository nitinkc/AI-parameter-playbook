# Structured outputs & tools

## `response_format`

Azure AI Foundry’s chat completion reference and OpenAI’s chat completion reference document `response_format` with:

- `{ "type": "text" }` (default)
- `{ "type": "json_object" }` (JSON mode) — valid JSON, but schema adherence is not enforced
- `{ "type": "json_schema", "json_schema": {...} }` (Structured Outputs) — enforces a JSON Schema

Important note from Azure docs: in JSON mode (`json_object`), you should also instruct the model to output JSON; otherwise the model may stream whitespace until it reaches the token limit.

## Structured outputs (recommended)

Microsoft Learn describes Structured Outputs as a way to force the model to follow a supplied JSON Schema, and contrasts it with older JSON mode that only guarantees valid JSON.

## `tools` and `tool_choice`

Azure OpenAI REST reference documents:

- `tools`: list of function tools the model may call
- `tool_choice`: control whether tools are used (none/auto/required) or force a specific tool.

This playbook uses tools sparingly, but you can incorporate them into experiments to measure impacts on accuracy and cost.

Next: **Experimentation**.

--8<-- "_abbreviations.md"
