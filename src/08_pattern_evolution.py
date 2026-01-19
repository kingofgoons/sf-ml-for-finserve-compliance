# Snowflake Notebook / Python Worksheet
# Part 3.2: Communication Pattern Evolution & Model Updates
#
# This script demonstrates:
# - Model drift detection
# - Batch re-embedding workflows
# - A/B testing between model versions
# - Performance monitoring framework
#
# Prerequisites:
# - Run 01-07 scripts first
#
# Time: ~4-5 minutes

# %% [markdown]
# # Part 3.2: Communication Pattern Evolution & Model Updates
# 
# **Goal:** Maintain models as communication patterns change.
# 
# ML models degrade over time:
# - New violation patterns emerge
# - Communication styles evolve
# - Regulatory requirements change
# 
# This module covers operational ML: keeping models accurate in production.

# %% Setup
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, current_timestamp

session = get_active_session()

print(f"Connected as: {session.get_current_user()}")
session.use_database("ML_COMPLIANCE_DEMO")
session.use_schema("ML")

# %% [markdown]
# ## 1. Model Performance Monitoring
# 
# Track model accuracy over time to detect drift.

# %% Create monitoring table
session.sql("""
CREATE TABLE IF NOT EXISTS ML.MODEL_PERFORMANCE_LOG (
    LOG_ID NUMBER AUTOINCREMENT,
    LOG_DATE TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    MODEL_NAME STRING,
    MODEL_VERSION STRING,
    
    -- Performance metrics
    ACCURACY FLOAT,
    PRECISION_SCORE FLOAT,
    RECALL_SCORE FLOAT,
    F1_SCORE FLOAT,
    
    -- Volume metrics
    TOTAL_PREDICTIONS INT,
    POSITIVE_PREDICTIONS INT,
    
    -- Ground truth (if available)
    CONFIRMED_VIOLATIONS INT,
    FALSE_POSITIVES INT,
    
    NOTES STRING
)
""").collect()
print("✅ Created model performance log table")

# %% Log current model performance
session.sql("""
INSERT INTO ML.MODEL_PERFORMANCE_LOG (
    MODEL_NAME, MODEL_VERSION, ACCURACY, PRECISION_SCORE, RECALL_SCORE,
    TOTAL_PREDICTIONS, POSITIVE_PREDICTIONS, NOTES
)
WITH model_stats AS (
    SELECT 
        COUNT(*) AS total,
        SUM(PREDICTED_VIOLATION) AS positive,
        SUM(CASE WHEN PREDICTED_VIOLATION = (CASE WHEN COMPLIANCE_LABEL != 'CLEAN' THEN 1 ELSE 0 END) THEN 1 ELSE 0 END) AS correct,
        SUM(CASE WHEN PREDICTED_VIOLATION = 1 AND COMPLIANCE_LABEL != 'CLEAN' THEN 1 ELSE 0 END) AS true_pos,
        SUM(CASE WHEN PREDICTED_VIOLATION = 1 AND COMPLIANCE_LABEL = 'CLEAN' THEN 1 ELSE 0 END) AS false_pos,
        SUM(CASE WHEN PREDICTED_VIOLATION = 0 AND COMPLIANCE_LABEL != 'CLEAN' THEN 1 ELSE 0 END) AS false_neg
    FROM ML.MODEL_PREDICTIONS_V1
)
SELECT 
    'EMAIL_COMPLIANCE_CLASSIFIER',
    'V1',
    correct * 1.0 / total,
    CASE WHEN (true_pos + false_pos) > 0 THEN true_pos * 1.0 / (true_pos + false_pos) ELSE 0 END,
    CASE WHEN (true_pos + false_neg) > 0 THEN true_pos * 1.0 / (true_pos + false_neg) ELSE 0 END,
    total,
    positive,
    'Initial baseline measurement'
FROM model_stats
""").collect()
print("✅ Logged current model performance")

# %% View performance history
print("--- Model Performance History ---")
session.table("ML.MODEL_PERFORMANCE_LOG").show()

# %% [markdown]
# ## 2. Drift Detection
# 
# Compare recent predictions to historical baselines.

# %% Create drift detection query
print("--- Drift Detection Analysis ---")

session.sql("""
WITH baseline AS (
    SELECT 
        AVG(ACCURACY) AS baseline_accuracy,
        AVG(PRECISION_SCORE) AS baseline_precision,
        AVG(RECALL_SCORE) AS baseline_recall
    FROM ML.MODEL_PERFORMANCE_LOG
    WHERE LOG_DATE < DATEADD('day', -7, CURRENT_TIMESTAMP())
),
recent AS (
    SELECT 
        AVG(ACCURACY) AS recent_accuracy,
        AVG(PRECISION_SCORE) AS recent_precision,
        AVG(RECALL_SCORE) AS recent_recall
    FROM ML.MODEL_PERFORMANCE_LOG
    WHERE LOG_DATE >= DATEADD('day', -7, CURRENT_TIMESTAMP())
)
SELECT 
    ROUND(b.baseline_accuracy, 4) AS baseline_acc,
    ROUND(r.recent_accuracy, 4) AS recent_acc,
    ROUND((r.recent_accuracy - b.baseline_accuracy) * 100, 2) AS accuracy_change_pct,
    CASE 
        WHEN ABS(r.recent_accuracy - b.baseline_accuracy) > 0.05 THEN '⚠️ DRIFT DETECTED'
        ELSE '✅ Stable'
    END AS drift_status
FROM baseline b, recent r
""").show()

# %% [markdown]
# ## 3. A/B Testing Framework
# 
# Compare model versions on the same data.

# %% Create A/B test results table
session.sql("""
CREATE OR REPLACE TABLE ML.AB_TEST_RESULTS AS
WITH model_v1 AS (
    SELECT 
        p.EMAIL_ID,
        p.PREDICTED_VIOLATION AS V1_PREDICTION,
        CASE WHEN e.COMPLIANCE_LABEL != 'CLEAN' THEN 1 ELSE 0 END AS ACTUAL
    FROM ML.MODEL_PREDICTIONS_V1 p
    JOIN RAW_DATA.EMAILS e ON p.EMAIL_ID = e.EMAIL_ID
),
-- Simulate V2 predictions (in practice, this would be your new model)
-- Here we use LLM as a proxy for "new model"
model_v2 AS (
    SELECT 
        EMAIL_ID,
        CASE 
            WHEN AI_CLASSIFY(BODY, ['COMPLIANCE_VIOLATION', 'CLEAN_EMAIL']):"label"::STRING = 'COMPLIANCE_VIOLATION'
            THEN 1 ELSE 0 
        END AS V2_PREDICTION
    FROM RAW_DATA.EMAILS
    LIMIT 50  -- Sample for demo
)
SELECT 
    v1.EMAIL_ID,
    v1.V1_PREDICTION,
    COALESCE(v2.V2_PREDICTION, v1.V1_PREDICTION) AS V2_PREDICTION,
    v1.ACTUAL,
    
    -- Agreement analysis
    CASE WHEN v1.V1_PREDICTION = COALESCE(v2.V2_PREDICTION, v1.V1_PREDICTION) THEN 1 ELSE 0 END AS MODELS_AGREE,
    
    -- Who's right when they disagree?
    CASE WHEN v1.V1_PREDICTION = v1.ACTUAL THEN 1 ELSE 0 END AS V1_CORRECT,
    CASE WHEN COALESCE(v2.V2_PREDICTION, v1.V1_PREDICTION) = v1.ACTUAL THEN 1 ELSE 0 END AS V2_CORRECT
    
FROM model_v1 v1
LEFT JOIN model_v2 v2 ON v1.EMAIL_ID = v2.EMAIL_ID
""").collect()
print("✅ Created A/B test results")

# %% Analyze A/B test results
print("--- A/B Test: Model V1 vs V2 (LLM) ---")
session.sql("""
SELECT 
    'V1 (XGBoost)' AS model,
    SUM(V1_CORRECT) AS correct,
    COUNT(*) AS total,
    ROUND(SUM(V1_CORRECT) * 100.0 / COUNT(*), 1) AS accuracy_pct
FROM ML.AB_TEST_RESULTS

UNION ALL

SELECT 
    'V2 (LLM-based)',
    SUM(V2_CORRECT),
    COUNT(*),
    ROUND(SUM(V2_CORRECT) * 100.0 / COUNT(*), 1)
FROM ML.AB_TEST_RESULTS
""").show()

# %% Show disagreements
print("--- Cases Where Models Disagree ---")
session.sql("""
SELECT 
    EMAIL_ID,
    V1_PREDICTION,
    V2_PREDICTION,
    ACTUAL,
    CASE 
        WHEN V1_CORRECT = 1 AND V2_CORRECT = 0 THEN 'V1 correct'
        WHEN V1_CORRECT = 0 AND V2_CORRECT = 1 THEN 'V2 correct'
        ELSE 'Both wrong'
    END AS WINNER
FROM ML.AB_TEST_RESULTS
WHERE MODELS_AGREE = 0
LIMIT 10
""").show()

# %% [markdown]
# ## 4. Batch Re-Embedding Workflow
# 
# When embedding models are updated, re-embed historical data.

# %% Create re-embedding procedure
session.sql("""
CREATE OR REPLACE PROCEDURE ML.REEMBED_EMAILS(
    NEW_MODEL_NAME STRING,
    BATCH_SIZE INT DEFAULT 1000
)
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    -- Create new embeddings table with version
    CREATE OR REPLACE TABLE SEARCH.EMAIL_EMBEDDINGS_NEW AS
    SELECT 
        EMAIL_ID,
        SENDER,
        RECIPIENT,
        SUBJECT,
        LEFT(BODY, 500) AS BODY_PREVIEW,
        COMPLIANCE_LABEL,
        SENT_AT,
        SNOWFLAKE.CORTEX.EMBED_TEXT_768(
            :NEW_MODEL_NAME,
            CONCAT(SUBJECT, ' ', BODY)
        ) AS EMBEDDING,
        :NEW_MODEL_NAME AS EMBEDDING_MODEL,
        CURRENT_TIMESTAMP() AS EMBEDDED_AT
    FROM RAW_DATA.EMAILS;
    
    -- Log the re-embedding
    INSERT INTO ML.MODEL_PERFORMANCE_LOG (MODEL_NAME, MODEL_VERSION, NOTES)
    VALUES ('EMBEDDINGS', :NEW_MODEL_NAME, 'Re-embedded all emails');
    
    RETURN 'Re-embedding complete: ' || (SELECT COUNT(*) FROM SEARCH.EMAIL_EMBEDDINGS_NEW) || ' emails processed';
END;
$$
""").collect()
print("✅ Created re-embedding procedure: ML.REEMBED_EMAILS")

# %% Demo the procedure (using same model)
print("--- Re-Embedding Demo ---")
session.call("ML.REEMBED_EMAILS", "e5-base-v2")

# %% [markdown]
# ## 5. Model Version Management
# 
# Track which model version generated each prediction.

# %% Create versioned predictions view
session.sql("""
CREATE OR REPLACE VIEW ML.VERSIONED_PREDICTIONS AS
SELECT 
    p.EMAIL_ID,
    p.PREDICTED_VIOLATION,
    p.COMPLIANCE_LABEL,
    'EMAIL_COMPLIANCE_CLASSIFIER' AS MODEL_NAME,
    'V1' AS MODEL_VERSION,
    CURRENT_TIMESTAMP() AS PREDICTION_DATE,
    
    -- Include email metadata for audit
    e.SENDER,
    e.RECIPIENT,
    e.SUBJECT,
    e.SENT_AT
    
FROM ML.MODEL_PREDICTIONS_V1 p
JOIN RAW_DATA.EMAILS e ON p.EMAIL_ID = e.EMAIL_ID
""").collect()
print("✅ Created versioned predictions view")

# %% [markdown]
# ## 6. Retraining Triggers
# 
# Define when to retrain models.

# %% Create retraining recommendation logic
print("--- Retraining Recommendation Engine ---")

session.sql("""
WITH performance_trend AS (
    SELECT 
        MODEL_NAME,
        MODEL_VERSION,
        ACCURACY,
        LAG(ACCURACY) OVER (PARTITION BY MODEL_NAME ORDER BY LOG_DATE) AS PREV_ACCURACY,
        LOG_DATE
    FROM ML.MODEL_PERFORMANCE_LOG
),
recommendations AS (
    SELECT 
        MODEL_NAME,
        MODEL_VERSION,
        MAX(LOG_DATE) AS LAST_LOGGED,
        AVG(ACCURACY) AS AVG_ACCURACY,
        MIN(ACCURACY) AS MIN_ACCURACY,
        
        -- Retraining triggers
        CASE 
            WHEN AVG(ACCURACY) < 0.8 THEN 'RETRAIN: Accuracy below threshold'
            WHEN MIN(ACCURACY) < 0.7 THEN 'RETRAIN: Significant accuracy drop detected'
            WHEN DATEDIFF('day', MIN(LOG_DATE), CURRENT_DATE()) > 90 THEN 'RETRAIN: Model older than 90 days'
            ELSE 'OK: Model performing within bounds'
        END AS RECOMMENDATION
    FROM performance_trend
    GROUP BY MODEL_NAME, MODEL_VERSION
)
SELECT * FROM recommendations
""").show()

# %% Print retraining guidelines
print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    RETRAINING TRIGGER GUIDELINES                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📉 PERFORMANCE-BASED TRIGGERS                                      │
│  • Accuracy drops below 80%                                         │
│  • Precision drops below 70% (too many false positives)             │
│  • Recall drops below 75% (missing violations)                      │
│                                                                     │
│  📅 TIME-BASED TRIGGERS                                             │
│  • Model older than 90 days                                         │
│  • Embedding model updated by provider                              │
│  • Quarterly scheduled review                                       │
│                                                                     │
│  📊 DATA-BASED TRIGGERS                                             │
│  • New violation type identified                                    │
│  • Training data size increased by 50%+                             │
│  • Label distribution shifted significantly                         │
│                                                                     │
│  🏢 BUSINESS TRIGGERS                                               │
│  • New regulatory requirements                                      │
│  • Organizational restructure (new departments)                     │
│  • Acquisition of new entity                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# %% [markdown]
# ## 7. Monitoring Dashboard Query
# 
# Query for a monitoring dashboard.

# %% Create comprehensive monitoring query
print("--- Model Health Dashboard ---")

session.sql("""
SELECT 
    '📊 Model Performance' AS section,
    MODEL_NAME || '/' || MODEL_VERSION AS model,
    ROUND(ACCURACY * 100, 1) || '%' AS accuracy,
    ROUND(PRECISION_SCORE * 100, 1) || '%' AS precision,
    ROUND(RECALL_SCORE * 100, 1) || '%' AS recall,
    TO_CHAR(LOG_DATE, 'YYYY-MM-DD') AS last_updated
FROM ML.MODEL_PERFORMANCE_LOG
WHERE LOG_DATE = (SELECT MAX(LOG_DATE) FROM ML.MODEL_PERFORMANCE_LOG)

UNION ALL

SELECT 
    '📈 Prediction Volume',
    'Last 7 days',
    COUNT(*)::STRING || ' emails',
    SUM(PREDICTED_VIOLATION)::STRING || ' flagged',
    ROUND(SUM(PREDICTED_VIOLATION) * 100.0 / COUNT(*), 1) || '% flag rate',
    TO_CHAR(CURRENT_DATE(), 'YYYY-MM-DD')
FROM ML.MODEL_PREDICTIONS_V1

UNION ALL

SELECT 
    '🔍 Search Index',
    'EMAIL_SEARCH_SERVICE',
    (SELECT COUNT(*)::STRING FROM SEARCH.EMAIL_EMBEDDINGS) || ' emails indexed',
    'e5-base-v2',
    '768 dimensions',
    TO_CHAR(CURRENT_DATE(), 'YYYY-MM-DD')
""").show()

# %% [markdown]
# ## 8. Operational Runbook Summary

# %% Print runbook
print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    ML OPERATIONS RUNBOOK                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📅 DAILY                                                           │
│  • Check MODEL_PERFORMANCE_LOG for anomalies                        │
│  • Review high-confidence violation flags                           │
│                                                                     │
│  📅 WEEKLY                                                          │
│  • Run A/B comparison if testing new model                          │
│  • Review false positive feedback from analysts                     │
│  • Log performance metrics                                          │
│                                                                     │
│  📅 MONTHLY                                                         │
│  • Full model evaluation on held-out test set                       │
│  • Review drift detection trends                                    │
│  • Update training data with new labeled examples                   │
│                                                                     │
│  📅 QUARTERLY                                                       │
│  • Evaluate retraining triggers                                     │
│  • Review embedding model updates from Snowflake                    │
│  • Regulatory compliance review                                     │
│  • Model documentation update                                       │
│                                                                     │
│  🚨 AD-HOC                                                          │
│  • Retrain on new violation type discovery                          │
│  • Re-embed on embedding model updates                              │
│  • A/B test before major model changes                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# %% [markdown]
# ## ✅ Part 3.2 Complete — Demo Finished!
# 
# **What we learned:**
# - Performance monitoring and logging
# - Drift detection patterns
# - A/B testing framework for model comparison
# - Batch re-embedding procedures
# - Retraining trigger guidelines
# - Operational runbook for ML in production
# 
# ---
# 
# ## 🎉 Congratulations!
# 
# You've completed the ML-Focused Hands-On Lab covering:
# 
# | Part | Topic | Key Snowflake Features |
# |------|-------|------------------------|
# | 1.1 | Snowpark Processing | DataFrames, UDFs |
# | 1.2 | Feature Store | Entities, Feature Views |
# | 1.3 | Model Registry | Model versioning, inference |
# | 2.1 | Benchmarking | ML vs keyword comparison |
# | 2.2 | LLM Integration | AI_CLASSIFY, AI_COMPLETE |
# | 2.3 | Fine-Tuning | CORTEX.FINETUNE |
# | 3.1 | Vector Search | AI_EMBED, Cortex Search |
# | 3.2 | Operations | Monitoring, A/B testing |
# 
# **To clean up:** Run `99_reset.sql`

