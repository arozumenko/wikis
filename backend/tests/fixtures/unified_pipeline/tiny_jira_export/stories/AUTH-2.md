---
issue_type: story
epic_key: AUTH-1
title: Implement login endpoint
---

# AUTH-2: Implement login endpoint

## Summary

As a user I want to log in so that I receive a session token.

## Tasks

- Create `POST /auth/login` endpoint.
- Validate email/password against the user database.
- Issue a signed JWT on success.
- Return 401 on invalid credentials.
