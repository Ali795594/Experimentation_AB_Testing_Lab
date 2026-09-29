-- =====================================================================
-- 02_load_data.sql
-- Loads the cleaned CSV (data/processed/ab_test_users_sql.csv, written by
-- src/data_cleaning.py) into Postgres.
--
-- Run from the sql/ folder:
--   psql -U your_user -d your_database -f 01_schema.sql
--   psql -U your_user -d your_database -f 02_load_data.sql
-- =====================================================================

\copy ab_test_users (experiment_id, user_id, exp_date, variant, device, country_or_region, customer_type, sessions, purchases, revenue, conversion, average_order_value) FROM '../data/processed/ab_test_users_sql.csv' WITH (FORMAT csv, HEADER true, NULL '');

SELECT COUNT(*) AS row_count FROM ab_test_users;
