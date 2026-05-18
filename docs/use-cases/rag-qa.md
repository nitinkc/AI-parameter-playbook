# 2) RAG / Grounded Q&A

## Goal

Answer questions **using only provided context** (retrieved passages), with low hallucination.

## Recommended baseline settings

- `temperature`: 0.0–0.2
- `top_p`: 1.0
- penalties: 0 (start neutral)
- `max_tokens`: enough for a short answer + citations

## Prompt pattern

- System: *Only answer from the provided context. If insufficient, say you don’t know.*
- User: question
- Add a final message containing the retrieved context, delimited.

## Strong pattern: structured output with citations

Return JSON that contains:

- `answer`
- `citations`: an array of `{source_id, quote}` tied to your context chunks
- `confidence`: low/medium/high (optional)

Structured outputs (`json_schema`) are recommended for extraction and workflows.

## How to test

Use a fixed set of Q&A pairs and compare:

- groundedness (answer supported by context)
- refusal rate when context is insufficient
- variance across seeds

This use case typically benefits from low temperature and strict output structure.

--8<-- "_abbreviations.md"
