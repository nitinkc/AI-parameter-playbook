# Cloud Platform Support

This playbook supports multiple LLM cloud providers. Learn parameters locally, then test on your preferred cloud.

## Supported Platforms

1. **Azure OpenAI** – Microsoft's managed OpenAI service
2. **OpenAI** – Direct OpenAI API access
3. **Google Vertex AI** – Google's generative AI platform
4. **Amazon Bedrock** – AWS managed LLM service

All platforms support the core parameters documented here. Implementation details vary slightly per platform.

## Parameter Equivalence

| Concept | Azure OpenAI | OpenAI | Vertex AI | Bedrock |
|---------|--------------|--------|-----------|---------|
| Temperature | `temperature` (0-2) | `temperature` (0-2) | `temperature` (0-2) | `temperature` (0-1) |
| Top-p | `top_p` (0-1) | `top_p` (0-1) | `top_p` (0-1) | `topP` (0-1) |
| Top-k | (via `top_p`) | (via `top_p`) | `top_k` | `topK` |
| Max tokens | `max_completion_tokens` | `max_tokens` | `max_output_tokens` | `max_tokens` |
| Stop sequences | `stop` | `stop` | `stop_sequences` | `stop_sequences` |
| Frequency penalty | `frequency_penalty` | `frequency_penalty` | (not directly) | (not directly) |
| Presence penalty | `presence_penalty` | `presence_penalty` | (not directly) | (not directly) |
| Seed/determinism | `seed` | `seed` | `seed` | (limited) |

## Quick Setup by Platform

### Azure OpenAI

```bash
# Set environment variables
export AZURE_OPENAI_ENDPOINT=https://<name>.openai.azure.com/
export AZURE_OPENAI_API_KEY=<your-key>
export AZURE_OPENAI_DEPLOYMENT=gpt-4

# Run experiment
python3 scripts/cloud/azure_sweep.py --preset summarization --trials 3
```

**Docs:** [Azure OpenAI Setup](../setup/overview.md)

### OpenAI

```bash
# Set environment variables
export OPENAI_API_KEY=<your-key>
export OPENAI_MODEL=gpt-4

# Run experiment
python3 scripts/cloud/openai_sweep.py --preset summarization --trials 3
```

**Docs:** [OpenAI Setup](openai/setup.md)

### Google Vertex AI

```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# Set environment variables
export GOOGLE_PROJECT_ID=<your-project>
export GOOGLE_REGION=us-central1
export VERTEX_MODEL=gemini-pro

# Run experiment
python3 scripts/cloud/vertex_sweep.py --preset summarization --trials 3
```

**Docs:** [Google Vertex AI Setup](google/setup.md)

### Amazon Bedrock

```bash
# Configure AWS credentials (use aws configure or set env vars)
export AWS_REGION=us-east-1
export BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0

# Run experiment
python3 scripts/cloud/bedrock_sweep.py --preset summarization --trials 3
```

**Docs:** [Amazon Bedrock Setup](amazon/setup.md)

---

## Which Platform Should I Use?

| Need | Best Option |
|------|-----------|
| Cheapest startup | OpenAI API (pay-as-you-go) |
| Enterprise+compliance | Azure OpenAI (SOC2, FedRAMP) |
| Google ecosystem | Vertex AI (integrates with GCP) |
| AWS ecosystem | Bedrock (integrates with AWS) |
| Variety of models | Bedrock (Claude, Llama, Mistral) |
| Cutting-edge models | OpenAI (GPT-4 Turbo, o1) |

---

**See also:** [Parameter Equivalence Guide](../parameters/cloud-equivalence.md)

--8<-- "_abbreviations.md"
