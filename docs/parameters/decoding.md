# Decoding controls (creativity vs reliability)

## `temperature`

- Controls sampling randomness.
- In OpenAI’s API reference it ranges **0–2**, where higher values increase randomness and lower values increase determinism.
- In Azure AI Foundry’s chat completions preview reference it ranges **0–1**, with the same interpretation and the recommendation *not to adjust temperature and top_p together*. (Different endpoints/models can impose different ranges.)

### Practical guidance

- **0–0.2**: deterministic-ish, best for extraction, classification, grounded Q&A
- **0.3–0.7**: balanced
- **0.8–1.2+**: brainstorming / variability (higher hallucination risk)

## `top_p` (nucleus sampling)

- Alternative to temperature.
- Restricts candidates to a set of tokens whose cumulative probability mass is `top_p`.
- Both Azure and OpenAI docs warn that **combining top_p and temperature is hard to predict**, so usually tune one or the other.

## Recommended practice for learning

Start with a sweep where you vary **only temperature** (keeping top_p=1), then do a sweep varying **only top_p** (keeping temperature fixed), so you can attribute changes to a single knob.

Next: **Repetition & Novelty**.

--8<-- "_abbreviations.md"
