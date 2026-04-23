-- Init script for Wikis PostgreSQL (runs once on first start).
-- Enables pgvector in the default database (wikis_auth).
-- The wikis_store database is created separately by the shell init script.

CREATE EXTENSION IF NOT EXISTS vector;
