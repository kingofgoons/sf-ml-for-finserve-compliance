# Snowflake Notebook / Python Worksheet
# Part 2.2: LLM-to-Email Compliance Integration
#
# This script demonstrates:
# - Using AI_CLASSIFY for email categorization
# - Using AI_COMPLETE for structured compliance analysis
# - Building an end-to-end LLM scoring pipeline
# - Comparing LLM vs ML classifier results
#
# Prerequisites:
# - Run 01-04 scripts first
# - Cortex LLM functions enabled in your account
#
# Time: ~4-5 minutes

# %% [markdown]
# # Part 2.2: LLM-to-Email Compliance Integration
# 
# **Goal:** Chain LLM analysis into structured compliance workflows.
# 
# Building on what you learned in the GenAI session, we'll now integrate
# Cortex LLM functions into our ML pipeline:
# 
# - **AI_CLASSIFY**: Zero-shot email categorization
# - **AI_COMPLETE**: Extract structured compliance signals with reasoning
# - **Compare**: LLM predictions vs our trained ML model

# %% Setup
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, lit

session = get_active_session()

print(f"Connected as: {session.get_current_user()}")
session.use_database("ML_COMPLIANCE_DEMO")
session.use_schema("ML")

# %% [markdown]
# ## 1. AI_CLASSIFY for Email Categorization
# 
# Zero-shot classification — no training data needed!
# The model uses the category names to understand what to look for.

# %% Define compliance categories
COMPLIANCE_CATEGORIES = [
    "INSIDER_TRADING",
    "CONFIDENTIALITY_BREACH", 
    "PERSONAL_TRADING_VIOLATION",
    "INFORMATION_BARRIER_VIOLATION",
    "CLEAN"
]

# %% Run AI_CLASSIFY on emails
print("🤖 Running AI_CLASSIFY on emails...")

# AI_CLASSIFY returns the most likely category
classified_df = session.sql(f"""
    SELECT 
        EMAIL_ID,
        SUBJECT,
        COMPLIANCE_LABEL AS ACTUAL_LABEL,
        AI_CLASSIFY(
            BODY,
            {COMPLIANCE_CATEGORIES}
        ):"label"::STRING AS LLM_CLASSIFICATION
    FROM RAW_DATA.EMAILS
    LIMIT 20  -- Limit for demo (LLM calls have latency)
""")

print("--- AI_CLASSIFY Results ---")
classified_df.show()

# %% Compare AI_CLASSIFY to ground truth
print("--- AI_CLASSIFY Accuracy by Category ---")
session.sql(f"""
    WITH classified AS (
        SELECT 
            COMPLIANCE_LABEL AS ACTUAL,
            AI_CLASSIFY(
                BODY,
                {COMPLIANCE_CATEGORIES}
            ):"label"::STRING AS PREDICTED
        FROM RAW_DATA.EMAILS
        LIMIT 50
    )
    SELECT 
        ACTUAL,
        SUM(CASE WHEN ACTUAL = PREDICTED THEN 1 ELSE 0 END) AS CORRECT,
        COUNT(*) AS TOTAL,
        ROUND(SUM(CASE WHEN ACTUAL = PREDICTED THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS ACCURACY_PCT
    FROM classified
    GROUP BY ACTUAL
    ORDER BY TOTAL DESC
""").show()

# %% [markdown]
# ## 2. AI_COMPLETE for Structured Analysis
# 
# Get detailed reasoning and extract specific compliance signals.

# %% Create compliance analysis prompt
COMPLIANCE_PROMPT = """Analyze this email for compliance violations. 

Email:
{email_body}

Respond in JSON format:
{{
    "risk_level": "HIGH" | "MEDIUM" | "LOW",
    "violation_type": "INSIDER_TRADING" | "CONFIDENTIALITY_BREACH" | "PERSONAL_TRADING" | "INFO_BARRIER" | "NONE",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "key_phrases": ["suspicious phrase 1", "suspicious phrase 2"]
}}"""

# %% Run AI_COMPLETE with structured output
print("🤖 Running AI_COMPLETE for detailed analysis...")

# Analyze a sample of emails
analysis_df = session.sql("""
    SELECT 
        EMAIL_ID,
        SUBJECT,
        COMPLIANCE_LABEL AS ACTUAL_LABEL,
        SNOWFLAKE.CORTEX.COMPLETE(
            'mistral-large2',
            CONCAT(
                'Analyze this email for compliance violations. ',
                'Email: ', BODY, ' ',
                'Respond in JSON with: risk_level (HIGH/MEDIUM/LOW), ',
                'violation_type (INSIDER_TRADING/CONFIDENTIALITY_BREACH/PERSONAL_TRADING/INFO_BARRIER/NONE), ',
                'confidence (0-1), reasoning (brief), key_phrases (list).'
            )
        ) AS LLM_ANALYSIS
    FROM RAW_DATA.EMAILS
    WHERE COMPLIANCE_LABEL != 'CLEAN'  -- Focus on violations for demo
    LIMIT 5
""")

print("--- AI_COMPLETE Detailed Analysis ---")
analysis_df.show()

# %% [markdown]
# ## 3. Build LLM Scoring Pipeline
# 
# Create a reusable function that combines classification + risk scoring.

# %% Create LLM compliance scoring function
session.sql("""
CREATE OR REPLACE FUNCTION ML.LLM_COMPLIANCE_SCORE(email_body STRING)
RETURNS VARIANT
LANGUAGE SQL
AS $$
    PARSE_JSON(
        SNOWFLAKE.CORTEX.COMPLETE(
            'mistral-large2',
            CONCAT(
                'You are a financial compliance analyst. Analyze this email and return JSON only: ',
                '{"risk_score": 0-100, "violation_likely": true/false, "category": "INSIDER_TRADING|CONFIDENTIALITY|PERSONAL_TRADING|INFO_BARRIER|CLEAN", "explanation": "one sentence"}. ',
                'Email: ', email_body
            )
        )
    )
$$
""").collect()
print("✅ Created LLM scoring function: ML.LLM_COMPLIANCE_SCORE")

# %% Test the scoring function
print("--- LLM Compliance Scores ---")
session.sql("""
    SELECT 
        EMAIL_ID,
        LEFT(SUBJECT, 40) AS SUBJECT,
        COMPLIANCE_LABEL,
        ML.LLM_COMPLIANCE_SCORE(BODY) AS LLM_SCORE
    FROM RAW_DATA.EMAILS
    WHERE COMPLIANCE_LABEL != 'CLEAN'
    LIMIT 5
""").show()

# %% [markdown]
# ## 4. Compare LLM vs ML Classifier
# 
# How does the LLM compare to our trained XGBoost model?

# %% Run comparison on a sample
print("🔄 Comparing LLM vs ML predictions...")

comparison_df = session.sql("""
    WITH llm_predictions AS (
        SELECT 
            e.EMAIL_ID,
            e.SUBJECT,
            e.COMPLIANCE_LABEL AS ACTUAL,
            CASE WHEN e.COMPLIANCE_LABEL != 'CLEAN' THEN 1 ELSE 0 END AS ACTUAL_VIOLATION,
            
            -- ML Model prediction
            COALESCE(p.PREDICTED_VIOLATION, 0) AS ML_PREDICTION,
            
            -- LLM prediction (simplified binary)
            CASE 
                WHEN AI_CLASSIFY(e.BODY, ['COMPLIANCE_VIOLATION', 'CLEAN_EMAIL']):"label"::STRING = 'COMPLIANCE_VIOLATION'
                THEN 1 
                ELSE 0 
            END AS LLM_PREDICTION
            
        FROM RAW_DATA.EMAILS e
        LEFT JOIN ML.MODEL_PREDICTIONS_V1 p ON e.EMAIL_ID = p.EMAIL_ID
        LIMIT 30  -- Sample for demo speed
    )
    SELECT *
    FROM llm_predictions
""")

print("--- LLM vs ML Predictions ---")
comparison_df.show()

# %% Calculate agreement between methods
print("--- Method Agreement Analysis ---")
session.sql("""
    WITH predictions AS (
        SELECT 
            CASE WHEN COMPLIANCE_LABEL != 'CLEAN' THEN 1 ELSE 0 END AS ACTUAL,
            COALESCE(p.PREDICTED_VIOLATION, 0) AS ML_PRED,
            CASE 
                WHEN AI_CLASSIFY(e.BODY, ['COMPLIANCE_VIOLATION', 'CLEAN_EMAIL']):"label"::STRING = 'COMPLIANCE_VIOLATION'
                THEN 1 ELSE 0 
            END AS LLM_PRED
        FROM RAW_DATA.EMAILS e
        LEFT JOIN ML.MODEL_PREDICTIONS_V1 p ON e.EMAIL_ID = p.EMAIL_ID
        LIMIT 50
    )
    SELECT
        -- Agreement stats
        SUM(CASE WHEN ML_PRED = LLM_PRED THEN 1 ELSE 0 END) AS METHODS_AGREE,
        SUM(CASE WHEN ML_PRED != LLM_PRED THEN 1 ELSE 0 END) AS METHODS_DISAGREE,
        ROUND(SUM(CASE WHEN ML_PRED = LLM_PRED THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS AGREEMENT_PCT,
        
        -- When they disagree, who's right?
        SUM(CASE WHEN ML_PRED != LLM_PRED AND ML_PRED = ACTUAL THEN 1 ELSE 0 END) AS ML_RIGHT_LLM_WRONG,
        SUM(CASE WHEN ML_PRED != LLM_PRED AND LLM_PRED = ACTUAL THEN 1 ELSE 0 END) AS LLM_RIGHT_ML_WRONG
    FROM predictions
""").show()

# %% [markdown]
# ## 5. Ensemble Approach: ML + LLM
# 
# Combine both methods for higher confidence predictions.

# %% Create ensemble scoring view
session.sql("""
CREATE OR REPLACE VIEW ML.ENSEMBLE_PREDICTIONS AS
SELECT 
    e.EMAIL_ID,
    e.SUBJECT,
    e.COMPLIANCE_LABEL AS ACTUAL_LABEL,
    CASE WHEN e.COMPLIANCE_LABEL != 'CLEAN' THEN 1 ELSE 0 END AS ACTUAL_VIOLATION,
    
    -- Individual predictions
    COALESCE(p.PREDICTED_VIOLATION, 0) AS ML_PREDICTION,
    CASE 
        WHEN AI_CLASSIFY(e.BODY, ['COMPLIANCE_VIOLATION', 'CLEAN_EMAIL']):"label"::STRING = 'COMPLIANCE_VIOLATION'
        THEN 1 ELSE 0 
    END AS LLM_PREDICTION,
    
    -- Ensemble: flag if EITHER method flags it (high recall)
    CASE 
        WHEN COALESCE(p.PREDICTED_VIOLATION, 0) = 1 
          OR AI_CLASSIFY(e.BODY, ['COMPLIANCE_VIOLATION', 'CLEAN_EMAIL']):"label"::STRING = 'COMPLIANCE_VIOLATION'
        THEN 1 ELSE 0 
    END AS ENSEMBLE_HIGH_RECALL,
    
    -- Ensemble: flag only if BOTH methods agree (high precision)
    CASE 
        WHEN COALESCE(p.PREDICTED_VIOLATION, 0) = 1 
         AND AI_CLASSIFY(e.BODY, ['COMPLIANCE_VIOLATION', 'CLEAN_EMAIL']):"label"::STRING = 'COMPLIANCE_VIOLATION'
        THEN 1 ELSE 0 
    END AS ENSEMBLE_HIGH_PRECISION

FROM RAW_DATA.EMAILS e
LEFT JOIN ML.MODEL_PREDICTIONS_V1 p ON e.EMAIL_ID = p.EMAIL_ID
""").collect()
print("✅ Created ensemble predictions view: ML.ENSEMBLE_PREDICTIONS")

# %% Compare ensemble strategies
print("--- Ensemble Strategy Comparison ---")
session.sql("""
    SELECT 
        'ML Only' AS strategy,
        SUM(CASE WHEN ML_PREDICTION = 1 AND ACTUAL_VIOLATION = 1 THEN 1 ELSE 0 END) AS true_positives,
        SUM(CASE WHEN ML_PREDICTION = 1 AND ACTUAL_VIOLATION = 0 THEN 1 ELSE 0 END) AS false_positives,
        SUM(ML_PREDICTION) AS total_flagged
    FROM ML.ENSEMBLE_PREDICTIONS
    
    UNION ALL
    
    SELECT 
        'LLM Only',
        SUM(CASE WHEN LLM_PREDICTION = 1 AND ACTUAL_VIOLATION = 1 THEN 1 ELSE 0 END),
        SUM(CASE WHEN LLM_PREDICTION = 1 AND ACTUAL_VIOLATION = 0 THEN 1 ELSE 0 END),
        SUM(LLM_PREDICTION)
    FROM ML.ENSEMBLE_PREDICTIONS
    
    UNION ALL
    
    SELECT 
        'Ensemble (Either)',
        SUM(CASE WHEN ENSEMBLE_HIGH_RECALL = 1 AND ACTUAL_VIOLATION = 1 THEN 1 ELSE 0 END),
        SUM(CASE WHEN ENSEMBLE_HIGH_RECALL = 1 AND ACTUAL_VIOLATION = 0 THEN 1 ELSE 0 END),
        SUM(ENSEMBLE_HIGH_RECALL)
    FROM ML.ENSEMBLE_PREDICTIONS
    
    UNION ALL
    
    SELECT 
        'Ensemble (Both)',
        SUM(CASE WHEN ENSEMBLE_HIGH_PRECISION = 1 AND ACTUAL_VIOLATION = 1 THEN 1 ELSE 0 END),
        SUM(CASE WHEN ENSEMBLE_HIGH_PRECISION = 1 AND ACTUAL_VIOLATION = 0 THEN 1 ELSE 0 END),
        SUM(ENSEMBLE_HIGH_PRECISION)
    FROM ML.ENSEMBLE_PREDICTIONS
""").show()

# %% [markdown]
# ## 6. LLM Output to Snowpark UDF Integration
# 
# Parse structured LLM output and feed it as parameters to a Snowpark Python UDF.
# This enables chaining: LLM analysis → Python processing → downstream actions.

# %% Create a Python UDF that processes LLM-extracted data
session.sql("""
CREATE OR REPLACE FUNCTION ML.PROCESS_COMPLIANCE_RESULT(
    risk_score INT,
    violation_type STRING,
    key_phrases ARRAY
)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
HANDLER = 'process_result'
AS $$
def process_result(risk_score: int, violation_type: str, key_phrases: list) -> dict:
    '''
    Process LLM-extracted compliance data and generate action recommendations.
    This UDF receives parsed LLM output as structured parameters.
    '''
    # Business logic based on LLM analysis
    actions = []
    
    if risk_score >= 80:
        actions.append("ESCALATE_IMMEDIATELY")
        actions.append("NOTIFY_COMPLIANCE_OFFICER")
    elif risk_score >= 50:
        actions.append("FLAG_FOR_REVIEW")
        actions.append("ADD_TO_WATCHLIST")
    else:
        actions.append("LOG_AND_MONITOR")
    
    # Violation-specific actions
    if violation_type == "INSIDER_TRADING":
        actions.append("CHECK_TRADING_RECORDS")
        actions.append("CROSS_REFERENCE_ANNOUNCEMENTS")
    elif violation_type == "INFO_BARRIER":
        actions.append("AUDIT_DEPARTMENT_COMMUNICATIONS")
    
    return {
        "risk_score": risk_score,
        "violation_type": violation_type,
        "key_phrases": key_phrases,
        "recommended_actions": actions,
        "priority": "HIGH" if risk_score >= 80 else "MEDIUM" if risk_score >= 50 else "LOW"
    }
$$
""").collect()
print("✅ Created Snowpark UDF: ML.PROCESS_COMPLIANCE_RESULT")

# %% Demonstrate LLM → UDF pipeline
print("--- LLM Output → Snowpark UDF Pipeline ---")

session.sql("""
WITH llm_analysis AS (
    -- Step 1: LLM extracts structured data from email
    SELECT 
        EMAIL_ID,
        SUBJECT,
        PARSE_JSON(
            SNOWFLAKE.CORTEX.COMPLETE(
                'mistral-large2',
                CONCAT(
                    'Analyze this email for compliance risk. Return ONLY valid JSON: ',
                    '{"risk_score": 0-100, "violation_type": "INSIDER_TRADING|CONFIDENTIALITY|INFO_BARRIER|NONE", "key_phrases": ["phrase1", "phrase2"]}. ',
                    'Email: ', LEFT(BODY, 400)
                )
            )
        ) AS LLM_OUTPUT
    FROM RAW_DATA.EMAILS
    WHERE COMPLIANCE_LABEL != 'CLEAN'
    LIMIT 3
)
-- Step 2: Feed parsed LLM output to Snowpark UDF
SELECT 
    EMAIL_ID,
    LEFT(SUBJECT, 30) AS SUBJECT,
    LLM_OUTPUT:risk_score::INT AS EXTRACTED_RISK,
    LLM_OUTPUT:violation_type::STRING AS EXTRACTED_TYPE,
    
    -- Call Snowpark UDF with LLM-extracted parameters
    ML.PROCESS_COMPLIANCE_RESULT(
        COALESCE(LLM_OUTPUT:risk_score::INT, 0),
        COALESCE(LLM_OUTPUT:violation_type::STRING, 'NONE'),
        COALESCE(LLM_OUTPUT:key_phrases::ARRAY, ARRAY_CONSTRUCT())
    ) AS UDF_RESULT
    
FROM llm_analysis
""").show()

# %% Show the complete pipeline flow
print("""
┌─────────────────────────────────────────────────────────────────────┐
│              LLM → SNOWPARK UDF INTEGRATION PATTERN                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Raw Email  │───▶│  LLM Analysis │───▶│ PARSE_JSON() │          │
│  │              │    │ (AI_COMPLETE) │    │              │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                   │                 │
│                                                   ▼                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Actions    │◀───│ Snowpark UDF │◀───│ Structured   │          │
│  │  (escalate,  │    │ (Python      │    │ Parameters   │          │
│  │   flag, etc) │    │  logic)      │    │ (risk, type) │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                     │
│  BENEFITS:                                                          │
│  • LLM provides intelligence (understanding, extraction)            │
│  • UDF provides deterministic business logic                        │
│  • Structured handoff via JSON parsing                              │
│  • Auditable, testable pipeline                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# %% [markdown]
# ## 8. Cost Considerations
# 
# LLM calls have compute costs — when to use each approach?

# %% Print cost/benefit analysis
print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE EACH APPROACH                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ML CLASSIFIER (XGBoost)                                            │
│  ✅ High throughput (millions of emails)                            │
│  ✅ Low latency (~ms per prediction)                                │
│  ✅ Fixed compute cost                                              │
│  ⚠️  Requires training data                                         │
│  ⚠️  May miss novel violation patterns                              │
│                                                                     │
│  LLM (AI_CLASSIFY, AI_COMPLETE)                                     │
│  ✅ Zero-shot (no training data needed)                             │
│  ✅ Provides reasoning/explanation                                  │
│  ✅ Catches novel patterns                                          │
│  ⚠️  Higher latency (~1-2s per call)                                │
│  ⚠️  Token-based cost                                               │
│                                                                     │
│  RECOMMENDED ARCHITECTURE:                                          │
│  ─────────────────────────────────────────────────────────────────  │
│  1. ML classifier as first-pass filter (fast, cheap)                │
│  2. LLM analysis on ML-flagged emails (deeper analysis)             │
│  3. LLM for edge cases where methods disagree                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# %% [markdown]
# ## ✅ Part 2.2 Complete!
# 
# **What we learned:**
# - AI_CLASSIFY for zero-shot email categorization
# - AI_COMPLETE for detailed compliance analysis with reasoning
# - Creating reusable LLM scoring functions
# - **LLM output → PARSE_JSON → Snowpark UDF pipeline**
# - Ensemble approaches combining ML + LLM
# - Cost/benefit tradeoffs
# 
# **Key insight:** ML and LLM are complementary:
# - ML: Fast, scalable first-pass filter
# - LLM: Deep analysis, explanations, novel patterns
# - **Snowpark UDFs: Deterministic business logic on LLM-extracted data**
# 
# **Next:** Run `06_fine_tuning.py` to customize an LLM for hedge fund terminology.

