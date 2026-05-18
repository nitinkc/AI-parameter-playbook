# Metrics & analysis

The analyzer outputs lightweight metrics that help you compare configurations:

- length (chars, words)
- distinct-2 (bigram diversity)
- repetition score (repeated bigrams)
- JSON validity (when using structured outputs)

For executive decisions, you’ll often combine these with:

- human rating (1–5)
- task success rate (e.g., correct label)
- groundedness checks (for RAG)

The scripts are intentionally simple so you can modify them.

--8<-- "_abbreviations.md"
