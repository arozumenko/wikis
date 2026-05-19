# Deployment Guide

## Prerequisites

- Python 3.11
- Docker 24+
- PostgreSQL 15 or SQLite (for local dev)

## Steps

1. Clone the repository.
2. Copy `.env.example` to `.env`.
3. Run `docker compose up -d`.
4. Access the API at `http://localhost:8000`.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | Database connection string | `sqlite:///./app.db` |
| `SECRET_KEY` | JWT signing key | — |
