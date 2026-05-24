# Google Vertex AI Setup

## Overview

Google Cloud Vertex AI provides access to state-of-the-art models (Gemini, PaLM) with enterprise-grade features.

## Prerequisites

1. **Google Cloud Project** with billing enabled
2. **gcloud CLI** installed: https://cloud.google.com/sdk/docs/install
3. **Python 3.8+**

## Step 1: Create a Google Cloud Project

```bash
# Create project
gcloud projects create my-llm-project --name="LLM Parameter Testing"

# Set default project
gcloud config set project my-llm-project

# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com
```

## Step 2: Authenticate

```bash
# Login and set up Application Default Credentials
gcloud auth application-default login
```

This creates credentials that the Python SDK will use automatically.

## Step 3: Configure Environment

Edit your `.env` file:

```bash
# Google Cloud
GOOGLE_PROJECT_ID=my-llm-project
GOOGLE_REGION=us-central1
VERTEX_MODEL=gemini-1.5-pro  # or gemini-pro, text-bison, etc.
```

## Step 4: Supported Models

As of May 2026, Vertex AI supports:

- **Gemini 1.5 Pro** – Latest multimodal model
- **Gemini 1.5 Flash** – Fast, efficient variant
- **Gemini 2.0** (preview) – Next-generation
- **Claude 3** (via Bedrock integration)

**List available models:**

```bash
gcloud ai models list --region=us-central1
```

## Step 5: Run an Experiment

```bash
# Activate venv and install deps
source .venv-exp/bin/activate
pip install -r requirements.txt

# Run preset sweep
python3 scripts/cloud/vertex_sweep.py --preset summarization --trials 3

# Analyze results
python3 scripts/analyze_results.py --input runs/latest.jsonl
```

## Cost Estimates

Vertex AI pricing varies by region and model. Typical costs for 1000 API calls:

- **Gemini 1.5 Flash**: ~$0.075
- **Gemini 1.5 Pro**: ~$0.30
- **Text-Bison**: ~$0.01

[Current pricing](https://cloud.google.com/vertex-ai/pricing)

## Troubleshooting

**"Permission denied" error:**
```bash
gcloud projects add-iam-policy-binding my-llm-project \
  --member=user:your-email@gmail.com \
  --role=roles/aiplatform.user
```

**"Model not found" error:**
Ensure the model is available in your region:
```bash
gcloud ai models list --region=us-central1 | grep gemini
```

**"Quota exceeded" error:**
Check and increase quotas:
```bash
gcloud compute project-info describe --project=my-llm-project \
  --format='value(quotas)'
```

---

**Next:** [Run your first experiment](../../experiments/quickstart-rest.md)

--8<-- "_abbreviations.md"
