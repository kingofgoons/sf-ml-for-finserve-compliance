-- ============================================================================
-- 00_setup.sql
-- Snowflake ML for Email Compliance Demo
-- 
-- Run this script FIRST before starting the hands-on lab.
-- All statements are idempotent (safe to re-run).
--
-- NOTE: This demo builds on concepts from the GenAI Compliance session.
--       We focus here on Snowpark ML, Feature Store, Model Registry,
--       Fine-tuning, and Vector Search at scale.
-- ============================================================================

-- ============================================================================
-- ACCOUNTADMIN SECTION: Create resources and role (run once)
-- ============================================================================
USE ROLE ACCOUNTADMIN;

-- ----------------------------------------------------------------------------
-- 1. DATABASE & WAREHOUSE
-- ----------------------------------------------------------------------------

CREATE DATABASE IF NOT EXISTS ML_COMPLIANCE_DEMO;

CREATE WAREHOUSE IF NOT EXISTS ML_COMPLIANCE_WH
    WAREHOUSE_SIZE = 'MEDIUM'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for ML email compliance demo';

-- ----------------------------------------------------------------------------
-- 2. CUSTOM ROLE FOR DEMO
-- ----------------------------------------------------------------------------

CREATE ROLE IF NOT EXISTS ML_COMPLIANCE_RL;

-- Database & schema privileges
GRANT USAGE ON DATABASE ML_COMPLIANCE_DEMO TO ROLE ML_COMPLIANCE_RL;
GRANT ALL ON DATABASE ML_COMPLIANCE_DEMO TO ROLE ML_COMPLIANCE_RL;
GRANT CREATE SCHEMA ON DATABASE ML_COMPLIANCE_DEMO TO ROLE ML_COMPLIANCE_RL;

-- Warehouse privileges
GRANT USAGE ON WAREHOUSE ML_COMPLIANCE_WH TO ROLE ML_COMPLIANCE_RL;

-- Cortex AI access (required for AI SQL functions: AI_CLASSIFY, AI_EMBED, etc.)
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE ML_COMPLIANCE_RL;

-- Grant role to current user
SET my_user = (SELECT CURRENT_USER());
GRANT ROLE ML_COMPLIANCE_RL TO USER IDENTIFIER($my_user);

-- ============================================================================
-- SWITCH TO CUSTOM ROLE: All remaining operations use least-privilege
-- ============================================================================
USE ROLE ML_COMPLIANCE_RL;
USE DATABASE ML_COMPLIANCE_DEMO;
USE WAREHOUSE ML_COMPLIANCE_WH;

-- ----------------------------------------------------------------------------
-- 3. SCHEMAS
-- ----------------------------------------------------------------------------

-- Schema for raw email data
CREATE SCHEMA IF NOT EXISTS RAW_DATA;

-- Schema for ML features and models
CREATE SCHEMA IF NOT EXISTS ML;

-- Schema for Cortex Search services and embeddings
CREATE SCHEMA IF NOT EXISTS SEARCH;

-- ----------------------------------------------------------------------------
-- 4. RAW EMAIL TABLE
-- ----------------------------------------------------------------------------

USE SCHEMA RAW_DATA;

-- Primary table for email archive data
-- Matches the synthetic data schema from generate_data.py
CREATE OR REPLACE TABLE EMAILS (
    EMAIL_ID        VARCHAR(36) PRIMARY KEY,
    SENDER          VARCHAR(100) NOT NULL,
    RECIPIENT       VARCHAR(100) NOT NULL,
    CC              VARCHAR(500),           -- Nullable, comma-separated
    SUBJECT         VARCHAR(500),
    BODY            VARCHAR(16777216),      -- Large text field for email body
    SENT_AT         TIMESTAMP_NTZ NOT NULL,
    SENDER_DEPT     VARCHAR(50),
    RECIPIENT_DEPT  VARCHAR(50),
    COMPLIANCE_LABEL VARCHAR(50),           -- Ground truth for training/eval
    
    -- Metadata columns
    LOADED_AT       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    SOURCE_FILE     VARCHAR(255)
);

-- Clustering for common query patterns
ALTER TABLE EMAILS CLUSTER BY (SENT_AT, SENDER_DEPT);

COMMENT ON TABLE EMAILS IS 
    'Raw email archive data for compliance surveillance. Labels: CLEAN, INSIDER_TRADING, CONFIDENTIALITY_BREACH, PERSONAL_TRADING, INFO_BARRIER_VIOLATION';

-- ----------------------------------------------------------------------------
-- 5. STAGING FOR DATA LOAD
-- ----------------------------------------------------------------------------

CREATE OR REPLACE STAGE EMAIL_DATA_STAGE
    FILE_FORMAT = (
        TYPE = 'CSV'
        FIELD_OPTIONALLY_ENCLOSED_BY = '"'
        SKIP_HEADER = 1
        NULL_IF = ('')
    )
    COMMENT = 'Stage for loading synthetic email CSV data';

/*
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UPLOADING DATA TO THE STAGE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OPTION 1: Snowsight UI (recommended)                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  1. Go to Data → Databases → ML_COMPLIANCE_DEMO → RAW_DATA → Stages         │
│  2. Click on EMAIL_DATA_STAGE                                               │
│  3. Click "+ Files" button                                                  │
│  4. Upload emails_synthetic.csv from the data/ folder                       │
│                                                                             │
│  OPTION 2: SnowSQL CLI                                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  PUT file:///path/to/data/emails_synthetic.csv                              │
│      @ML_COMPLIANCE_DEMO.RAW_DATA.EMAIL_DATA_STAGE                          │
│      AUTO_COMPRESS=FALSE;                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
*/

-- Verify files are uploaded (run after upload)
-- LIST @EMAIL_DATA_STAGE;

-- ----------------------------------------------------------------------------
-- 6. ML SCHEMA OBJECTS
-- ----------------------------------------------------------------------------

USE SCHEMA ML;

-- Table to store model predictions for analysis
CREATE OR REPLACE TABLE EMAIL_PREDICTIONS (
    EMAIL_ID            VARCHAR(36) PRIMARY KEY,
    PREDICTED_LABEL     VARCHAR(50),
    CONFIDENCE_SCORE    FLOAT,
    MODEL_VERSION       VARCHAR(50),
    PREDICTED_AT        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    -- LLM-based analysis fields (for comparison with ML models)
    LLM_CLASSIFICATION  VARCHAR(50),
    LLM_REASONING       VARCHAR(4000),
    RISK_SCORE          FLOAT
);

-- Table for feature store outputs (communication patterns)
CREATE OR REPLACE TABLE COMMUNICATION_FEATURES (
    EMPLOYEE_EMAIL          VARCHAR(100) PRIMARY KEY,
    DEPT                    VARCHAR(50),
    
    -- Volume features
    EMAILS_SENT_7D          INT,
    EMAILS_SENT_30D         INT,
    EMAILS_RECEIVED_7D      INT,
    EMAILS_RECEIVED_30D     INT,
    
    -- Pattern features
    AFTER_HOURS_RATIO       FLOAT,      -- % of emails sent outside 8am-6pm
    CROSS_DEPT_RATIO        FLOAT,      -- % of emails to other departments
    AVG_BODY_LENGTH         FLOAT,
    
    -- Risk signals
    BARRIER_DEPT_CONTACT_CT INT,        -- Count of Research<->Trading contacts
    
    -- Metadata
    COMPUTED_AT             TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ----------------------------------------------------------------------------
-- 7. SEARCH SCHEMA OBJECTS  
-- ----------------------------------------------------------------------------

USE SCHEMA SEARCH;

-- Table for email embeddings (populated by AI_EMBED)
CREATE OR REPLACE TABLE EMAIL_EMBEDDINGS (
    EMAIL_ID        VARCHAR(36) PRIMARY KEY,
    SUBJECT         VARCHAR(500),
    BODY_PREVIEW    VARCHAR(1000),          -- Truncated for display
    EMBEDDING       VECTOR(FLOAT, 768),     -- e5-base-v2 dimension
    EMBEDDED_AT     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ----------------------------------------------------------------------------
-- 8. VERIFICATION
-- ----------------------------------------------------------------------------

-- Quick check that everything was created
SHOW SCHEMAS IN DATABASE ML_COMPLIANCE_DEMO;
SHOW TABLES IN SCHEMA ML_COMPLIANCE_DEMO.RAW_DATA;
SHOW TABLES IN SCHEMA ML_COMPLIANCE_DEMO.ML;
SHOW TABLES IN SCHEMA ML_COMPLIANCE_DEMO.SEARCH;

SELECT '✅ Setup complete!' AS STATUS;
SELECT 'Next: Load data with 01_snowpark_email_processing.py' AS NEXT_STEP;
