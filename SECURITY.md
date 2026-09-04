# Security policy

## Deployment baseline

The application is intended to run locally by default. For any internet-facing
deployment:

1. Set a random `FLASK_SECRET_KEY` and a separate random `ADMIN_TOKEN`.
2. Set `PUBLIC_MODE=1`, `FLASK_DEBUG=0`, and terminate TLS at a reverse proxy.
3. Send `X-Admin-Token` only from a protected administrator client. Never put the
   token in a URL or commit it to the repository.
4. Keep `/api/settings`, `/api/dependencies/install`, and
   `/api/local-models/scan` behind an additional network or identity-control
   layer. These endpoints are disabled without `ADMIN_TOKEN` in public mode.
5. Use an external queue/rate limiter and persistent session store before
   accepting untrusted, high-volume training jobs.

## Reporting a vulnerability

Please do not publish exploit details in a public issue. Contact the project
maintainer privately with reproduction steps, affected version, and impact.
