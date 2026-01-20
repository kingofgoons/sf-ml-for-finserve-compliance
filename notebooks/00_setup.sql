-- ============================================================================
-- 00_setup.sql
-- Snowflake ML Compliance Demo - Environment Setup
-- 
-- This script creates all required Snowflake objects for the demo.
-- Run this FIRST if you're setting up manually (vs using setup_snowflake.py)
--
-- Objects created:
--   - Warehouse: COMPLIANCE_DEMO_WH
--   - Database: COMPLIANCE_DEMO
--   - Schemas: EMAIL_SURVEILLANCE, ML, SEARCH
--   - Tables: EMAILS, FINETUNE_TRAINING, EMAIL_EMBEDDINGS
-- ============================================================================

-- ============================================================================
-- STEP 1: CREATE WAREHOUSE
-- ============================================================================
-- A warehouse is Snowflake's compute engine. Think of it as a cluster of 
-- servers that execute your queries. You only pay when it's running.
-- 
-- Key settings:
--   - WAREHOUSE_SIZE: Controls compute power (and cost). MEDIUM is good for demos.
--   - AUTO_SUSPEND: Warehouse stops after 60 seconds of inactivity (saves cost).
--   - AUTO_RESUME: Automatically starts when you run a query.

CREATE WAREHOUSE IF NOT EXISTS COMPLIANCE_DEMO_WH
    WAREHOUSE_SIZE = 'MEDIUM'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for ML compliance demo - auto-suspends after 60s';

-- Activate the warehouse
USE WAREHOUSE COMPLIANCE_DEMO_WH;


-- ============================================================================
-- STEP 2: CREATE DATABASE AND SCHEMAS
-- ============================================================================
-- Snowflake organizes data in a hierarchy: Database > Schema > Table
-- We create one database with multiple schemas to separate concerns:
--   - EMAIL_SURVEILLANCE: Raw email data
--   - ML: Machine learning artifacts (features, models, predictions)
--   - SEARCH: Vector embeddings and search indexes

CREATE DATABASE IF NOT EXISTS COMPLIANCE_DEMO;
USE DATABASE COMPLIANCE_DEMO;

-- Schema for raw email data
CREATE SCHEMA IF NOT EXISTS EMAIL_SURVEILLANCE;

-- Schema for ML artifacts (Feature Store, Model Registry, etc.)
CREATE SCHEMA IF NOT EXISTS ML;

-- Schema for vector search and embeddings
CREATE SCHEMA IF NOT EXISTS SEARCH;


-- ============================================================================
-- STEP 3: CREATE TABLES
-- ============================================================================

USE SCHEMA EMAIL_SURVEILLANCE;

-- Primary email archive table
-- This holds 10,000 synthetic hedge fund email communications
CREATE OR REPLACE TABLE EMAILS (
    EMAIL_ID        VARCHAR(36) PRIMARY KEY,     -- UUID
    SENDER          VARCHAR(100) NOT NULL,       -- sender@acmefund.com
    RECIPIENT       VARCHAR(100) NOT NULL,       -- recipient@acmefund.com
    CC              VARCHAR(500),                -- Optional CC recipients
    SUBJECT         VARCHAR(500),                -- Email subject line
    BODY            VARCHAR(16777216),           -- Email body (up to 16MB)
    SENT_AT         TIMESTAMP_NTZ NOT NULL,      -- When the email was sent
    SENDER_DEPT     VARCHAR(50),                 -- Sender's department
    RECIPIENT_DEPT  VARCHAR(50),                 -- Recipient's department
    COMPLIANCE_LABEL VARCHAR(50),                -- Ground truth label (for training)
    LOADED_AT       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

COMMENT ON TABLE EMAILS IS 
    'Raw email archive data for compliance surveillance. Labels: CLEAN, INSIDER_TRADING, CONFIDENTIALITY_BREACH, PERSONAL_TRADING, INFO_BARRIER_VIOLATION';


-- ============================================================================
-- STEP 4: CREATE ML SCHEMA OBJECTS
-- ============================================================================

USE SCHEMA ML;

-- Fine-tuning training data
-- Stores prompt/completion pairs for LLM fine-tuning
CREATE OR REPLACE TABLE FINETUNE_TRAINING (
    SAMPLE_ID       NUMBER AUTOINCREMENT,
    PROMPT          VARCHAR(16777216),           -- The input prompt
    COMPLETION      VARCHAR(16777216),           -- The expected completion
    LOADED_AT       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

COMMENT ON TABLE FINETUNE_TRAINING IS 
    'Labeled prompt/completion pairs for fine-tuning LLMs on compliance classification';

-- Model predictions table (populated during the demo)
CREATE OR REPLACE TABLE MODEL_PREDICTIONS (
    EMAIL_ID            VARCHAR(36) PRIMARY KEY,
    PREDICTED_LABEL     VARCHAR(50),
    CONFIDENCE_SCORE    FLOAT,
    MODEL_NAME          VARCHAR(100),
    MODEL_VERSION       VARCHAR(50),
    PREDICTED_AT        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

COMMENT ON TABLE MODEL_PREDICTIONS IS 
    'Stores predictions from ML models for comparison and analysis';


-- ============================================================================
-- STEP 5: CREATE SEARCH SCHEMA OBJECTS
-- ============================================================================

USE SCHEMA SEARCH;

-- Email embeddings table for vector search
-- The VECTOR type stores embeddings efficiently for similarity search
CREATE OR REPLACE TABLE EMAIL_EMBEDDINGS (
    EMAIL_ID        VARCHAR(36) PRIMARY KEY,
    SUBJECT         VARCHAR(500),
    BODY_PREVIEW    VARCHAR(1000),               -- Truncated body for display
    COMPLIANCE_LABEL VARCHAR(50),
    SENDER          VARCHAR(100),
    RECIPIENT       VARCHAR(100),
    EMBEDDING       VECTOR(FLOAT, 768),          -- 768-dim embedding vector
    EMBEDDED_AT     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

COMMENT ON TABLE EMAIL_EMBEDDINGS IS 
    'Email embeddings for semantic search using snowflake-arctic-embed-m (768 dimensions)';


-- ============================================================================
-- STEP 6: CREATE DATA STAGE
-- ============================================================================

USE SCHEMA EMAIL_SURVEILLANCE;

-- A stage is like a landing zone for files (similar to S3)
-- We use it to load CSV/JSON data into tables
CREATE OR REPLACE STAGE DATA_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for loading demo data files';


-- ============================================================================
-- STEP 7: FILE FORMATS
-- ============================================================================

-- CSV format for email data
CREATE OR REPLACE FILE FORMAT CSV_FORMAT
    TYPE = 'CSV'
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    SKIP_HEADER = 1
    NULL_IF = ('')
    FIELD_DELIMITER = ','
    ESCAPE_UNENCLOSED_FIELD = NONE;

-- JSONL format for fine-tuning data
CREATE OR REPLACE FILE FORMAT JSONL_FORMAT
    TYPE = 'JSON'
    STRIP_OUTER_ARRAY = FALSE;


-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Show what we created
SHOW SCHEMAS IN DATABASE COMPLIANCE_DEMO;
SHOW TABLES IN SCHEMA COMPLIANCE_DEMO.EMAIL_SURVEILLANCE;
SHOW TABLES IN SCHEMA COMPLIANCE_DEMO.ML;
SHOW TABLES IN SCHEMA COMPLIANCE_DEMO.SEARCH;

SELECT '✅ Setup complete!' AS STATUS;
SELECT 'Next: Load data using scripts/setup_snowflake.py or upload files manually' AS NEXT_STEP;
