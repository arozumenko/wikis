---
issue_type: story
epic_key: AUTH-1
title: Add token validation middleware
---

# AUTH-3: Add token validation middleware

## Summary

As a backend service I want JWT validation middleware so that every
protected endpoint automatically verifies the caller's identity.

## Tasks

- Implement `JWTMiddleware` that extracts and validates tokens.
- Return 401 when token is missing or expired.
- Inject validated `user_id` into request context.
