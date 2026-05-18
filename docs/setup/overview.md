# Azure OpenAI setup (conceptual)

Azure OpenAI’s inference is accessed via **data plane** endpoints. Calls are versioned using an `api-version` query parameter in `YYYY-MM-DD` format. The Microsoft Learn reference describes the GA inference spec as `2024-10-21`. 

> Reference: Microsoft Learn — *Azure OpenAI in Microsoft Foundry Models REST API reference*.

## Endpoint shape (Chat Completions)

For Azure OpenAI resources, the chat completions route follows the pattern:

```
POST https://{endpoint}/openai/deployments/{deployment-id}/chat/completions?api-version=2024-10-21
```

and is authenticated via `api-key` or Microsoft Entra ID.

## What this playbook assumes

- You already have an Azure OpenAI resource + a deployed model.
- You will place your deployment name in **one place**: `AZURE_OPENAI_DEPLOYMENT` in `.env`.

Next: **Authentication & Endpoints**.

--8<-- "_abbreviations.md"
