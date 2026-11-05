DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'nusalens_ro') THEN
        CREATE ROLE nusalens_ro WITH LOGIN PASSWORD 'readonly_pass';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE nusalens TO nusalens_ro;
GRANT USAGE ON SCHEMA public TO nusalens_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO nusalens_ro;

REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM nusalens_ro;
REVOKE ALL ON DATABASE nusalens FROM PUBLIC;
GRANT CONNECT ON DATABASE nusalens TO nusalens_ro;

ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO nusalens_ro;

ALTER ROLE nusalens_ro SET statement_timeout = '5s';
ALTER ROLE nusalens_ro SET default_transaction_read_only = on;
