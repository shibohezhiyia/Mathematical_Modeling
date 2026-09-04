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

## Secrets and private files

API keys are runtime values only. Put them in a local `.env` file or enter them
in the UI for a single request; never hard-code them in Python, JavaScript,
Markdown, screenshots, notebooks, or test fixtures. The repository ignores
`data/`, `workspace/`, databases, logs, model files, and
`workspace/**/.api_key`, but ignored files must still be checked before the
first commit.

Before publishing, inspect both staged files and Git history. If a real key was
ever committed, revoke it at the provider immediately and rotate it; deleting
the current file alone does not remove it from Git history.

## Reporting a vulnerability

Please do not publish exploit details in a public issue. Contact the project
maintainer privately with reproduction steps, affected version, and impact.
