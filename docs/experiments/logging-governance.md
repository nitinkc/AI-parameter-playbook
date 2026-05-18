# Logging & governance

## What to log

At minimum per run:

- timestamp
- preset name
- prompt/messages
- parameters
- output
- token usage (if returned)
- seed

## What NOT to log

Avoid logging:

- secrets (keys, tokens)
- personal data
- confidential documents

## Executive takeaway

A good tuning program is also a **risk management program**: logging enables auditability and supports rollbacks if behavior changes.

--8<-- "_abbreviations.md"
