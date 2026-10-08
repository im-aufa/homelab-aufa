-- db/setup.sql
-- Run this ONCE on your shared postgres instance before starting chat-api.
-- Connect as the postgres superuser:
--   psql -h <host> -U postgres
-- Then paste or \i this file.

-- 1. Create the dedicated Cerina user and database.
--    Replace 'CHANGE_ME' with a strong password before running.
CREATE USER cerina WITH PASSWORD 'CHANGE_ME';
CREATE DATABASE cerina OWNER cerina;

-- 2. Connect to the new database and enable the pgvector extension.
\c cerina

CREATE EXTENSION vector;

-- 3. Verify the extension is active.
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
