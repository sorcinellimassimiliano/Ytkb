-- Postgres init script: enable required extensions before the app connects.
-- Run automatically by the official postgres image from /docker-entrypoint-initdb.d.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
