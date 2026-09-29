-- =====================================================================
-- 03_analysis_queries.sql
-- Analytical queries for the Experimentation & A/B Testing Intelligence
-- Platform.
-- =====================================================================

-- 1. Sample size and conversion rate by experiment and variant
SELECT
    experiment_id,
    variant,
    COUNT(*)                                   AS n_users,
    SUM(conversion)                            AS n_conversions,
    ROUND(AVG(conversion::NUMERIC), 4)         AS conversion_rate
FROM ab_test_users
GROUP BY experiment_id, variant
ORDER BY experiment_id, variant;

-- 2. Revenue per user by experiment and variant
SELECT
    experiment_id,
    variant,
    ROUND(AVG(COALESCE(revenue, 0)), 2)  AS avg_revenue_per_user,
    ROUND(SUM(COALESCE(revenue, 0)), 2)  AS total_revenue
FROM ab_test_users
GROUP BY experiment_id, variant
ORDER BY experiment_id, variant;

-- 3. Conversion rate by device and variant
SELECT
    experiment_id,
    device,
    variant,
    COUNT(*)                            AS n_users,
    ROUND(AVG(conversion::NUMERIC), 4)  AS conversion_rate
FROM ab_test_users
WHERE device IS NOT NULL
GROUP BY experiment_id, device, variant
ORDER BY experiment_id, device, variant;

-- 4. Conversion rate by region and variant
SELECT
    experiment_id,
    country_or_region,
    variant,
    COUNT(*)                            AS n_users,
    ROUND(AVG(conversion::NUMERIC), 4)  AS conversion_rate
FROM ab_test_users
WHERE country_or_region IS NOT NULL
GROUP BY experiment_id, country_or_region, variant
ORDER BY experiment_id, country_or_region, variant;

-- 5. Conversion rate by customer type and variant
SELECT
    experiment_id,
    customer_type,
    variant,
    COUNT(*)                            AS n_users,
    ROUND(AVG(conversion::NUMERIC), 4)  AS conversion_rate
FROM ab_test_users
GROUP BY experiment_id, customer_type, variant
ORDER BY experiment_id, customer_type, variant;

-- 6. Average order value (converters only) by experiment/variant
SELECT
    experiment_id,
    variant,
    ROUND(AVG(average_order_value), 2) AS avg_order_value,
    COUNT(*) FILTER (WHERE purchases > 0) AS n_purchasers
FROM ab_test_users
WHERE purchases > 0
GROUP BY experiment_id, variant
ORDER BY experiment_id, variant;

-- 7. Sample balance check (control vs treatment ratio)
SELECT
    experiment_id,
    COUNT(*) FILTER (WHERE variant = 'control')   AS n_control,
    COUNT(*) FILTER (WHERE variant = 'treatment') AS n_treatment,
    ROUND(
        COUNT(*) FILTER (WHERE variant = 'treatment')::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE variant = 'control'), 0), 3
    ) AS treatment_to_control_ratio
FROM ab_test_users
GROUP BY experiment_id;

-- 8. Users who appear in more than one variant within the same experiment
--    (contamination check - should return 0 rows on the CLEANED table)
SELECT
    experiment_id,
    user_id,
    COUNT(DISTINCT variant) AS n_variants
FROM ab_test_users
GROUP BY experiment_id, user_id
HAVING COUNT(DISTINCT variant) > 1;

-- 9. Daily conversion trend within an experiment window
SELECT
    experiment_id,
    exp_date,
    variant,
    COUNT(*)                            AS n_users,
    ROUND(AVG(conversion::NUMERIC), 4)  AS conversion_rate
FROM ab_test_users
GROUP BY experiment_id, exp_date, variant
ORDER BY experiment_id, exp_date, variant;

-- 10. Data-quality sanity check: impossible values that should be absent
--     after cleaning
SELECT
    COUNT(*) FILTER (WHERE revenue < 0)              AS negative_revenue_rows,
    COUNT(*) FILTER (WHERE purchases > sessions)      AS purchases_exceed_sessions_rows
FROM ab_test_users;
