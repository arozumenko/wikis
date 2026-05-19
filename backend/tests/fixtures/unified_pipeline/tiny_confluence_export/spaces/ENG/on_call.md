---
title: On-Call Runbook
space_key: ENG
parent_path: /Engineering Home
---

# On-Call Runbook

## Alerts

| Alert | Severity | Action |
|---|---|---|
| `AuthServiceDown` | P0 | Restart auth pod; page on-call lead |
| `HighErrorRate` | P1 | Check recent deploys; roll back if needed |

## Contacts

- Primary: engineering@example.com
- Secondary: ops@example.com
