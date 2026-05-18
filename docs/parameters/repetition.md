# Repetition & novelty controls

## `frequency_penalty` (-2 to 2)

Penalizes tokens proportionally to how often they have already appeared, reducing verbatim repetitions. Azure AI Foundry and OpenAI references document the range **[-2, 2]**.

## `presence_penalty` (-2 to 2)

Penalizes tokens that have already appeared at least once, nudging the model to introduce new topics. Azure AI Foundry and OpenAI references document the range **[-2, 2]**.

## Common patterns

- Reduce repetition in longer content: `frequency_penalty` ~ 0.2–0.8
- Encourage topic exploration in ideation: `presence_penalty` ~ 0.3–1.0

Caution: High penalties can degrade factuality or completeness.

Next: **Length, Stops & Cost**.

--8<-- "_abbreviations.md"
