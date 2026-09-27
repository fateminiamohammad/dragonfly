# Security

Please **do not** open a public issue for a vulnerability. Report it privately through GitHub:
**Security → Report a vulnerability** on this repository. We aim to reply within 7 days.

## Deployment notes

- An empty `DRAGONFLY_API_KEYS` means an **open server**. Only use that locally.
- In production, expose only nginx. `docker-compose.prod.yml` publishes no model-service port.
- Plugins run in-process with the service's permissions. Install only plugins you trust.
