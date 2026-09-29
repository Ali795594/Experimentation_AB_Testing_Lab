-- =====================================================================
-- 01_schema.sql
-- Experimentation & A/B Testing Intelligence Platform
-- =====================================================================

DROP TABLE IF EXISTS ab_test_users;

CREATE TABLE ab_test_users (
    row_id               SERIAL PRIMARY KEY,
    experiment_id        VARCHAR(50)     NOT NULL,
    user_id              VARCHAR(30)     NOT NULL,
    exp_date             DATE            NOT NULL,
    variant              VARCHAR(20)     NOT NULL,
    device               VARCHAR(20),
    country_or_region    VARCHAR(50),
    customer_type        VARCHAR(20),
    sessions             INTEGER,
    purchases            INTEGER,
    revenue              NUMERIC(10, 2),
    conversion            INTEGER,
    average_order_value  NUMERIC(10, 2)
);

CREATE INDEX idx_ab_experiment_id ON ab_test_users (experiment_id);
CREATE INDEX idx_ab_variant       ON ab_test_users (variant);
CREATE INDEX idx_ab_user_id       ON ab_test_users (user_id);
