# API Reference

## Authentication Endpoints

### POST /auth/login

Authenticates a user and returns a session token.

**Request body**:
```json
{"username": "string", "password": "string"}
```

**Response**:
```json
{"token": "string", "expires_at": "ISO8601"}
```
