# Cloud Parameter Equivalence

This page maps parameters across all supported cloud platforms so you can use the same tuning strategy everywhere.

## Core Parameters

### Temperature

Controls randomness/creativity of responses.

| Platform | Parameter | Range | Default | Notes |
|----------|-----------|-------|---------|-------|
| Azure OpenAI | `temperature` | 0-2 | 1.0 | Can exceed 1.0 |
| OpenAI | `temperature` | 0-2 | 1.0 | Can exceed 1.0 |
| Vertex AI | `temperature` | 0-2 | 1.0 | Can exceed 1.0 |
| Bedrock (Claude) | `temperature` | 0-1 | 1.0 | Capped at 1.0 |
| Bedrock (Llama) | `temperature` | 0-1 | 0.5 | Capped at 1.0 |

**Migration tip:** If tuning temperature for Azure/OpenAI/Vertex and want to use Bedrock, divide by 2 if > 1.0.

### Top-p (Nucleus Sampling)

Filters candidate tokens via cumulative probability.

| Platform | Parameter | Range | Default | Notes |
|----------|-----------|-------|---------|-------|
| Azure OpenAI | `top_p` | 0-1 | 1.0 | Pure nucleus |
| OpenAI | `top_p` | 0-1 | 1.0 | Pure nucleus |
| Vertex AI | `top_p` | 0-1 | 1.0 | Pure nucleus |
| Bedrock (Anthropic) | `top_p` | 0-1 | 0.999 | Very close to 1 by default |
| Bedrock (Meta/Mistral) | `top_p` | 0-1 | 0.9 | Slightly tighter |

**Migration tip:** All platforms support 0-1 range identically.

### Top-k

Keeps top-k most likely tokens.

| Platform | Parameter | Range | Default | Notes |
|----------|-----------|-------|---------|-------|
| Azure OpenAI | N/A | — | — | Use top_p instead |
| OpenAI | N/A | — | — | Use top_p instead |
| Vertex AI | `top_k` | 1-40 | N/A | Optional, works with top_p |
| Bedrock (Anthropic) | `top_k` | 0-500 | N/A | Optional, not commonly used |
| Bedrock (Meta) | `top_k` | 0-500 | N/A | Optional |

**Migration tip:** Azure/OpenAI don't support top_k; use top_p for similar effect.

### Max Tokens / Max Output

Limits response length.

| Platform | Parameter | Notes |
|----------|-----------|-------|
| Azure OpenAI | `max_completion_tokens` | Renamed from `max_tokens` in v2024-10-21+ |
| OpenAI | `max_tokens` | Standard parameter |
| Vertex AI | `max_output_tokens` | Vertex-specific naming |
| Bedrock | `max_tokens` | Standard parameter |

**Migration tip:** Always cap max tokens appropriately for your use case.

### Stop Sequences

Tells model when to stop generating.

| Platform | Parameter | Format | Notes |
|----------|-----------|--------|-------|
| Azure OpenAI | `stop` | List of strings | Up to 4 stop sequences |
| OpenAI | `stop` | List of strings | Up to 4 stop sequences |
| Vertex AI | `stop_sequences` | List of strings | Variable limit |
| Bedrock | `stop_sequences` | List of strings | Variable limit |

**Migration tip:** Same concept, slightly different parameter names.

## Advanced Parameters

### Frequency Penalty

Reduces likelihood of repeated tokens.

| Platform | Supported | Range | Notes |
|----------|-----------|-------|-------|
| Azure OpenAI | ✅ Yes | -2 to 2 | Called `frequency_penalty` |
| OpenAI | ✅ Yes | -2 to 2 | Called `frequency_penalty` |
| Vertex AI | ❌ No | — | Use top_p instead |
| Bedrock | ❌ No | — | Not directly supported |

### Presence Penalty

Penalizes new tokens that haven't appeared yet.

| Platform | Supported | Range | Notes |
|----------|-----------|-------|-------|
| Azure OpenAI | ✅ Yes | -2 to 2 | Called `presence_penalty` |
| OpenAI | ✅ Yes | -2 to 2 | Called `presence_penalty` |
| Vertex AI | ❌ No | — | Not available |
| Bedrock | ❌ No | — | Not available |

### Seed / Determinism

Makes responses reproducible with same seed.

| Platform | Parameter | Reproducible | Notes |
|----------|-----------|--------------|-------|
| Azure OpenAI | `seed` | ✅ Yes | Guarantees determinism (with caveats) |
| OpenAI | `seed` | ✅ Yes | Best-effort determinism |
| Vertex AI | `seed` | ⚠️ Partial | Reduces variability but not guaranteed |
| Bedrock | N/A | ❌ No | No native seed parameter |

## Quick Migration Guide

### From Azure OpenAI → Bedrock (Claude)

```python
# Azure
{
    "temperature": 0.8,
    "top_p": 0.95,
    "presence_penalty": 0.2,
    "frequency_penalty": 0.0,
    "max_completion_tokens": 256
}

# Bedrock equivalent (what to keep/drop)
{
    "temperature": 0.8,      # Keep as-is
    "top_p": 0.95,            # Keep as-is
    # presence_penalty: drop (not supported)
    # frequency_penalty: drop (not supported)
    "max_tokens": 256         # Rename from max_completion_tokens
}
```

### From OpenAI → Vertex AI

```python
# OpenAI
{
    "temperature": 1.0,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "max_tokens": 500,
    "stop": ["\n\nUser:"]
}

# Vertex equivalent
{
    "temperature": 1.0,         # Keep as-is
    "top_p": 0.9,                # Keep as-is
    # frequency_penalty: drop (not supported)
    "max_output_tokens": 500,   # Rename
    "stop_sequences": ["\n\nUser:"]  # Rename
}
```

### From Bedrock (Llama) → Azure OpenAI

```python
# Bedrock
{
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 256
}

# Azure equivalent
{
    "temperature": 0.5 * 2,      # Llama temp is ~half of Azure
    "top_p": 0.9,                 # Keep as-is
    # top_k: drop (not supported, use top_p)
    "max_completion_tokens": 256  # Rename
}
```

## Testing Across Platforms

When moving parameters from one platform to another:

1. **Keep same temperature/top_p values** if ranges overlap
2. **Drop unsupported parameters** (e.g., frequency_penalty for Bedrock)
3. **Test on new platform** with 2-3 small samples first
4. **Compare outputs** for consistency
5. **Adjust incrementally** if behavior differs

Example:

```bash
# Test same prompt on 3 platforms
python3 scripts/cloud/azure_sweep.py --preset summarization --trials 1
python3 scripts/cloud/openai_sweep.py --preset summarization --trials 1
python3 scripts/cloud/bedrock_sweep.py --preset summarization --trials 1

# Compare results
python3 scripts/analyze_results.py --compare-platforms
```

---

**See also:** [Platform Setup Guides](../cloud/overview.md)

--8<-- "_abbreviations.md"
