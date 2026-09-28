# Developer API

## Authentication

The REST API lives at `https://api.nimbusnotes.example/v2`. Create a personal access token under
**Profile → Developer → Tokens** and send it as `Authorization: Bearer <token>`. Tokens can be scoped to read-only.

## Rate limits

- Free: 60 requests per minute.
- Team: 600 requests per minute.
- Enterprise: 3,000 requests per minute.

Exceeding the limit returns HTTP `429` with a `Retry-After` header.

## Webhooks

Webhooks fire on `note.created`, `note.updated` and `note.deleted`. Each delivery is signed with an
`X-Nimbus-Signature` HMAC-SHA256 header. Failed deliveries are retried up to 5 times with exponential backoff.
