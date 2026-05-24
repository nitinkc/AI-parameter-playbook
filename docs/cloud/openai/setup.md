# OpenAI Setup

## Overview

Direct access to OpenAI's latest models (GPT-4, GPT-4 Turbo, o1) via the OpenAI API.

## Prerequisites

1. **OpenAI Account** with API access (https://platform.openai.com/signup)
2. **API Key** from https://platform.openai.com/api-keys
3. **Python 3.8+**
4. **Billing enabled** on your OpenAI account

## Step 1: Create an API Key

1. Go to https://platform.openai.com/api-keys
2. Click **Create new secret key**
3. Copy the key (you won't see it again)
4. Restrict key permissions if needed (recommended for production)

## Step 2: Configure Environment

Edit your `.env` file:

```bash
# OpenAI
OPENAI_API_KEY=sk-...your-key...
OPENAI_MODEL=gpt-4-turbo  # or gpt-4, gpt-3.5-turbo, o1 (preview)
```

## Step 3: Available Models (May 2026)

**Latest:**
- `gpt-4-turbo` – Best reasoning, vision support
- `gpt-4o` – Optimized for speed/cost
- `o1-preview` – Reasoning-focused (limited availability)

**Standard:**
- `gpt-4` – Original GPT-4
- `gpt-3.5-turbo` – Fast, cheap

**Context windows:**
- GPT-4 Turbo: 128K tokens
- GPT-4o: 128K tokens
- o1-preview: 128K tokens (reasoning steps don't count toward limit)

[Full model list](https://platform.openai.com/docs/models)

## Step 4: Run an Experiment

```bash
# Activate venv and install deps
source .venv-exp/bin/activate
pip install -r requirements.txt

# Run preset sweep
python3 scripts/cloud/openai_sweep.py --preset summarization --trials 3

# Analyze results
python3 scripts/analyze_results.py --input runs/latest.jsonl
```

## Cost Estimates

OpenAI uses per-token pricing. Typical costs per 1000 API calls:

- **GPT-3.5 Turbo**: ~$0.01-0.03
- **GPT-4 Turbo**: ~$0.30-0.90
- **GPT-4o**: ~$0.15-0.45
- **o1-preview**: ~$3.00-15.00

[Current pricing](https://openai.com/pricing)

## Monitoring Usage

**Check your usage:**

```bash
# In your account at https://platform.openai.com/usage
# Or via CLI:
curl https://api.openai.com/v1/usage/requests \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Set spending limits:**

Go to https://platform.openai.com/account/billing/limits and set:
- Hard limit (when to stop)
- Soft limit (warn when approaching)

## Troubleshooting

**"Invalid API Key" error:**
Verify your key at https://platform.openai.com/api-keys. Delete and create a new one if needed.

**"Rate limit exceeded" error:**
OpenAI has per-minute and per-day limits. Check your account tier and wait before retrying.

**"Model not found" error:**
Verify the model name at https://platform.openai.com/docs/models. Some models (like o1) are in limited preview.

**"Quota exceeded" error:**
Increase your spending limit in the console or wait for monthly reset.

---

**Next:** [Run your first experiment](../../experiments/quickstart-rest.md)

--8<-- "_abbreviations.md"
