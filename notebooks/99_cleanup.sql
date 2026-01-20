-- ============================================================================
-- 99_cleanup.sql
-- Snowflake ML Compliance Demo - Cleanup Script
--
-- Run this to remove all demo objects after the demo is complete.
-- WARNING: This will DELETE all data and objects created during the demo!
-- ============================================================================

-- ============================================================================
-- OPTIONAL: Cancel any running fine-tuning jobs
-- ============================================================================

-- Check for any running fine-tuning jobs first
-- SELECT SNOWFLAKE.CORTEX.FINETUNE('DESCRIBE', 'compliance_classifier_v1');

-- If a job is running, you can cancel it with:
-- SELECT SNOWFLAKE.CORTEX.FINETUNE('CANCEL', '<job_id>');


-- ============================================================================
-- DROP DATABASE
-- This cascades to all schemas, tables, views, stages, and functions
-- ============================================================================

DROP DATABASE IF EXISTS COMPLIANCE_DEMO CASCADE;


-- ============================================================================
-- DROP WAREHOUSE  
-- ============================================================================

DROP WAREHOUSE IF EXISTS COMPLIANCE_DEMO_WH;


-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Confirm objects are gone
SHOW DATABASES LIKE 'COMPLIANCE_DEMO';
SHOW WAREHOUSES LIKE 'COMPLIANCE_DEMO_WH';

SELECT '✅ Cleanup complete!' AS STATUS;
SELECT 'All demo objects have been removed.' AS MESSAGE;
