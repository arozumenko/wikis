---
title: Authentication Design
space_key: ENG
parent_path: /Engineering Home
---

# Authentication Design

## Overview

We use RS256 JWTs issued by the web app and validated by the backend.

## Token Lifecycle

1. User submits credentials via `POST /auth/login`.
2. Web app validates credentials and issues a signed JWT.
3. Backend validates the JWT using the public key from JWKS endpoint.
4. Session expires after 24 hours.
