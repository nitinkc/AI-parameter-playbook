# Amazon Bedrock Setup

## Overview

Amazon Bedrock provides access to Claude, Llama, Mistral, and other leading models through AWS.

## Prerequisites

1. **AWS Account** with billing enabled
2. **AWS CLI** installed: https://aws.amazon.com/cli/
3. **Python 3.8+**
4. **Model access enabled** in your AWS region

## Step 1: Enable Model Access

Bedrock requires explicit model access. In the AWS Console:

1. Navigate to **Bedrock** → **Model Access**
2. Click **Modify Model Access**
3. Enable at least one model (e.g., Claude 3, Llama 2)
4. Accept usage agreements
5. Click **Save changes**

This can take 5-10 minutes to propagate.

## Step 2: Configure AWS Credentials

**Option A: Use AWS CLI**

```bash
aws configure
# Enter: AWS Access Key ID, Secret Access Key, default region, output format
```

**Option B: Set environment variables**

```bash
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
export AWS_REGION=us-east-1
```

## Step 3: Configure Environment

Edit your `.env` file:

```bash
# Amazon Bedrock
AWS_REGION=us-east-1
BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
# Model ID format: <provider>.<model-name>-<version>
```

## Step 4: Supported Models (May 2026)

**Anthropic:**
- `anthropic.claude-3-opus-20240229-v1:0` – Most capable
- `anthropic.claude-3-sonnet-20240229-v1:0` – Balanced
- `anthropic.claude-3-haiku-20240307-v1:0` – Fast, cheap

**Meta Llama:**
- `meta.llama2-70b-chat-v1`
- `meta.llama2-13b-chat-v1`

**Mistral:**
- `mistral.mistral-7b-instruct-v0:2`
- `mistral.mistral-large-2402-v1:0`

**List available models:**

```bash
aws bedrock list-foundation-models --region us-east-1
```

## Step 5: Run an Experiment

```bash
# Activate venv and install deps
source .venv-exp/bin/activate
pip install -r requirements.txt

# Run preset sweep
python3 scripts/cloud/bedrock_sweep.py --preset summarization --trials 3

# Analyze results
python3 scripts/analyze_results.py --input runs/latest.jsonl
```

## Cost Estimates

Bedrock pricing is per 1M input/output tokens. Typical costs per 1000 API calls:

- **Claude 3 Haiku**: ~$0.03 input, $0.08 output
- **Claude 3 Sonnet**: ~$0.10 input, $0.30 output
- **Claude 3 Opus**: ~$0.30 input, $0.90 output
- **Llama 2 70B**: ~$0.01 input, $0.02 output

[Current pricing](https://aws.amazon.com/bedrock/pricing/)

## Troubleshooting

**"AccessDeniedException" error:**
Ensure model access is enabled in Bedrock console and your IAM role has `bedrock:*` permissions.

**"Model not found" error:**
Verify the model is enabled in your region:
```bash
aws bedrock list-foundation-models --region us-east-1 --query 'modelSummaries[*].modelId'
```

**"Quota exceeded" error:**
Bedrock has rate limits. Request an increase via AWS Support Console.

**"Invoke model" fails silently:**
Check AWS CloudTrail for error details:
```bash
aws cloudtrail lookup-events --region us-east-1 --lookup-attributes AttributeKey=EventName,AttributeValue=InvokeModel
```

---

**Next:** [Run your first experiment](../../experiments/quickstart-rest.md)

--8<-- "_abbreviations.md"
