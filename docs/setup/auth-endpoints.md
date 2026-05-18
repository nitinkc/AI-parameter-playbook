# Authentication & endpoints

## Supported auth

Azure OpenAI supports:

- **API key** auth via the `api-key` HTTP header.
- **Microsoft Entra ID** auth via an `Authorization: Bearer <token>` header.

These methods are described in the Azure OpenAI REST API reference.

## Versioning

Azure OpenAI uses an `api-version` query parameter. The GA inference spec documented for Azure OpenAI is `2024-10-21`.

## Minimal REST call

```http
POST https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME/chat/completions?api-version=2024-10-21
api-key: <YOUR_KEY>
Content-Type: application/json

{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello in one sentence."}
  ],
  "temperature": 0.2,
  "max_tokens": 64
}
```

The `temperature` and `max_tokens` parameters are part of the documented request body.

Next: **Runtime Configuration**.

--8<-- "_abbreviations.md"
