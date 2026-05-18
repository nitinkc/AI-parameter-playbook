# Runtime configuration (single source of truth)

To keep experiments repeatable and scripts maintainable, this repo uses `.env` to store:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`  ✅ **Put your deployment name here**
- `AZURE_OPENAI_API_VERSION` (default: `2024-10-21`)

Copy `.env.example` to `.env`.

## Why this matters

Centralizing configuration prevents errors during sweeps (e.g., accidentally testing different deployments or API versions).

--8<-- "_abbreviations.md"
