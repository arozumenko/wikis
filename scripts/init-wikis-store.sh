#!/bin/bash
# Init script for Wikis PostgreSQL (runs once on first start).
# Creates the wiki storage database and enables pgvector in both databases.
set -e

# Create the wiki storage database (ignore if already exists)
psql -v ON_ERROR_STOP=0 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE wikis_store OWNER $POSTGRES_USER;
EOSQL

# Enable pgvector in wikis_store
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "wikis_store" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;
EOSQL
