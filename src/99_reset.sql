-- ============================================================================
-- 99_reset.sql
-- Cleanup script for ML Email Compliance Demo
-- 
-- Run this script to tear down all demo objects and start fresh.
-- Safe to run multiple times (IF EXISTS on all drops).
-- ============================================================================

-- ----------------------------------------------------------------------------
-- CAUTION: This script will DELETE all demo data and objects!
-- ----------------------------------------------------------------------------

USE ROLE ACCOUNTADMIN;

-- ----------------------------------------------------------------------------
-- 1. DROP CORTEX SEARCH SERVICES (if any were created)
-- ----------------------------------------------------------------------------

DROP CORTEX SEARCH SERVICE IF EXISTS ML_COMPLIANCE_DEMO.SEARCH.EMAIL_SEARCH_SERVICE;

-- ----------------------------------------------------------------------------
-- 2. DROP DATABASE (cascades to all schemas, tables, stages)
-- ----------------------------------------------------------------------------

DROP DATABASE IF EXISTS ML_COMPLIANCE_DEMO;

-- ----------------------------------------------------------------------------
-- 3. DROP WAREHOUSE
-- ----------------------------------------------------------------------------

DROP WAREHOUSE IF EXISTS ML_COMPLIANCE_WH;

-- ----------------------------------------------------------------------------
-- 4. DROP ROLE
-- ----------------------------------------------------------------------------

DROP ROLE IF EXISTS ML_COMPLIANCE_RL;

-- ----------------------------------------------------------------------------
-- 5. VERIFICATION
-- ----------------------------------------------------------------------------

-- Confirm objects are gone
SHOW DATABASES LIKE 'ML_COMPLIANCE_DEMO';
SHOW WAREHOUSES LIKE 'ML_COMPLIANCE_WH';
SHOW ROLES LIKE 'ML_COMPLIANCE_RL';

SELECT '✅ Reset complete!' AS STATUS;
SELECT 'Run 00_setup.sql to recreate demo environment.' AS NEXT_STEP;
