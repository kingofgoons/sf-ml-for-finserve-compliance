# Snowflake Notebook / Python Worksheet
# Part 2.1: Email Surveillance Model Benchmarking
#
# This script demonstrates:
# - Comparing keyword-based vs ML-based detection
# - Side-by-side performance metrics
# - False positive analysis
# - Investigation efficiency gains
#
# Prerequisites:
# - Run 01-03 scripts first
#
# Time: ~4-5 minutes

# %% [markdown]
# # Part 2.1: Email Surveillance Model Benchmarking
# 
# **Goal:** Compare traditional keyword-based rules vs our ML classifier.
# 
# Traditional compliance systems rely on keyword matching:
# - "insider", "confidential", "don't tell anyone"
# - High false positive rates → analyst fatigue
# - Easy to evade with paraphrasing
# 
# We'll benchmark both approaches to quantify the improvement.

# %% Setup
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, when, lit, upper, contains
from snowflake.ml.registry import Registry
import pandas as pd

session = get_active_session()

print(f"Connected as: {session.get_current_user()}")
session.use_database("ML_COMPLIANCE_DEMO")
session.use_schema("ML")

# %% [markdown]
# ## 1. Define Keyword-Based Rules
# 
# Traditional compliance systems use keyword lists to flag emails.

# %% Create keyword-based detection
# These are common keywords used in legacy compliance systems

INSIDER_KEYWORDS = [
    "INSIDER", "MATERIAL", "NON-PUBLIC", "MNPI", "ANNOUNCEMENT",
    "BEFORE IT'S PUBLIC", "TIP", "HEADS UP"
]

SECRECY_KEYWORDS = [
    "DELETE THIS", "DON'T TELL", "KEEP BETWEEN", "OFF THE RECORD",
    "CONFIDENTIAL", "SECRET", "DESTROY", "NO ONE KNOWS"
]

TRADING_KEYWORDS = [
    "BUY NOW", "SELL NOW", "LOAD UP", "DUMP", "POSITION",
    "BEFORE FRIDAY", "ACT FAST", "MOVE QUICKLY"
]

# %% Create keyword detection function via SQL
session.sql("""
CREATE OR REPLACE FUNCTION ML.KEYWORD_BASED_FLAG(body STRING)
RETURNS INT
LANGUAGE SQL
AS $$
    CASE 
        -- Check for insider trading keywords
        WHEN UPPER(body) LIKE '%INSIDER%' OR
             UPPER(body) LIKE '%NON-PUBLIC%' OR
             UPPER(body) LIKE '%BEFORE IT%PUBLIC%' OR
             UPPER(body) LIKE '%ANNOUNCEMENT%MONDAY%' OR
             UPPER(body) LIKE '%TIP%' OR
             UPPER(body) LIKE '%HEADS UP%'
        THEN 1
        -- Check for secrecy keywords
        WHEN UPPER(body) LIKE '%DELETE%' OR
             UPPER(body) LIKE '%DON''T TELL%' OR
             UPPER(body) LIKE '%KEEP THIS BETWEEN%' OR
             UPPER(body) LIKE '%CONFIDENTIAL%' OR
             UPPER(body) LIKE '%SECRET%'
        THEN 1
        -- Check for trading urgency keywords  
        WHEN UPPER(body) LIKE '%BUY NOW%' OR
             UPPER(body) LIKE '%LOAD UP%' OR
             UPPER(body) LIKE '%ACT FAST%' OR
             UPPER(body) LIKE '%BEFORE FRIDAY%'
        THEN 1
        ELSE 0
    END
$$
""").collect()
print("✅ Created keyword-based detection function")

# %% [markdown]
# ## 2. Run Both Detection Methods
# 
# Apply keyword rules and ML model to all emails.

# %% Load emails with both predictions
comparison_df = session.sql("""
    SELECT 
        e.EMAIL_ID,
        e.SUBJECT,
        e.COMPLIANCE_LABEL,
        
        -- Ground truth: is this actually a violation?
        CASE WHEN e.COMPLIANCE_LABEL != 'CLEAN' THEN 1 ELSE 0 END AS ACTUAL_VIOLATION,
        
        -- Keyword-based detection
        ML.KEYWORD_BASED_FLAG(e.BODY) AS KEYWORD_FLAG,
        
        -- ML model prediction (from Part 1.3)
        COALESCE(p.PREDICTED_VIOLATION, 0) AS ML_FLAG
        
    FROM RAW_DATA.EMAILS e
    LEFT JOIN ML.MODEL_PREDICTIONS_V1 p ON e.EMAIL_ID = p.EMAIL_ID
""")

print("--- Detection Comparison Preview ---")
comparison_df.show(10)

# %% Save comparison for analysis
comparison_df.write.mode("overwrite").save_as_table("ML.MODEL_COMPARISON")
print("✅ Saved comparison to ML.MODEL_COMPARISON")

# %% [markdown]
# ## 3. Calculate Performance Metrics
# 
# Compare precision, recall, and false positive rates.

# %% Compute metrics for both methods
metrics_df = session.sql("""
    WITH stats AS (
        SELECT
            -- Keyword method
            SUM(CASE WHEN KEYWORD_FLAG = 1 AND ACTUAL_VIOLATION = 1 THEN 1 ELSE 0 END) AS kw_tp,
            SUM(CASE WHEN KEYWORD_FLAG = 1 AND ACTUAL_VIOLATION = 0 THEN 1 ELSE 0 END) AS kw_fp,
            SUM(CASE WHEN KEYWORD_FLAG = 0 AND ACTUAL_VIOLATION = 1 THEN 1 ELSE 0 END) AS kw_fn,
            SUM(CASE WHEN KEYWORD_FLAG = 0 AND ACTUAL_VIOLATION = 0 THEN 1 ELSE 0 END) AS kw_tn,
            
            -- ML method
            SUM(CASE WHEN ML_FLAG = 1 AND ACTUAL_VIOLATION = 1 THEN 1 ELSE 0 END) AS ml_tp,
            SUM(CASE WHEN ML_FLAG = 1 AND ACTUAL_VIOLATION = 0 THEN 1 ELSE 0 END) AS ml_fp,
            SUM(CASE WHEN ML_FLAG = 0 AND ACTUAL_VIOLATION = 1 THEN 1 ELSE 0 END) AS ml_fn,
            SUM(CASE WHEN ML_FLAG = 0 AND ACTUAL_VIOLATION = 0 THEN 1 ELSE 0 END) AS ml_tn,
            
            COUNT(*) AS total
        FROM ML.MODEL_COMPARISON
    )
    SELECT
        'Keyword Rules' AS method,
        kw_tp AS true_positives,
        kw_fp AS false_positives,
        kw_fn AS false_negatives,
        kw_tn AS true_negatives,
        ROUND(kw_tp * 100.0 / NULLIF(kw_tp + kw_fp, 0), 1) AS precision_pct,
        ROUND(kw_tp * 100.0 / NULLIF(kw_tp + kw_fn, 0), 1) AS recall_pct,
        ROUND(kw_fp * 100.0 / NULLIF(kw_fp + kw_tn, 0), 1) AS false_positive_rate,
        kw_tp + kw_fp AS total_flagged
    FROM stats
    
    UNION ALL
    
    SELECT
        'ML Classifier' AS method,
        ml_tp,
        ml_fp,
        ml_fn,
        ml_tn,
        ROUND(ml_tp * 100.0 / NULLIF(ml_tp + ml_fp, 0), 1),
        ROUND(ml_tp * 100.0 / NULLIF(ml_tp + ml_fn, 0), 1),
        ROUND(ml_fp * 100.0 / NULLIF(ml_fp + ml_tn, 0), 1),
        ml_tp + ml_fp
    FROM stats
""")

print("=" * 70)
print("                    MODEL COMPARISON RESULTS")
print("=" * 70)
metrics_df.show()

# %% [markdown]
# ## 4. False Positive Analysis
# 
# False positives are costly — analysts waste time investigating clean emails.

# %% Show false positives by each method
print("--- Keyword False Positives (flagged but CLEAN) ---")
session.sql("""
    SELECT 
        EMAIL_ID,
        SUBJECT,
        COMPLIANCE_LABEL
    FROM ML.MODEL_COMPARISON
    WHERE KEYWORD_FLAG = 1 AND ACTUAL_VIOLATION = 0
    LIMIT 10
""").show()

print("\n--- ML False Positives (flagged but CLEAN) ---")
session.sql("""
    SELECT 
        EMAIL_ID,
        SUBJECT,
        COMPLIANCE_LABEL
    FROM ML.MODEL_COMPARISON
    WHERE ML_FLAG = 1 AND ACTUAL_VIOLATION = 0
    LIMIT 10
""").show()

# %% [markdown]
# ## 5. Missed Violations (False Negatives)
# 
# Even more critical: which violations did each method miss?

# %% Show false negatives
print("--- Keyword False Negatives (missed violations) ---")
session.sql("""
    SELECT 
        EMAIL_ID,
        SUBJECT,
        COMPLIANCE_LABEL
    FROM ML.MODEL_COMPARISON
    WHERE KEYWORD_FLAG = 0 AND ACTUAL_VIOLATION = 1
    LIMIT 10
""").show()

print("\n--- ML False Negatives (missed violations) ---")
session.sql("""
    SELECT 
        EMAIL_ID,
        SUBJECT,
        COMPLIANCE_LABEL
    FROM ML.MODEL_COMPARISON
    WHERE ML_FLAG = 0 AND ACTUAL_VIOLATION = 1
    LIMIT 10
""").show()

# %% [markdown]
# ## 6. Investigation Efficiency Analysis
# 
# Calculate the workload reduction for compliance analysts.

# %% Calculate efficiency metrics
efficiency_df = session.sql("""
    WITH totals AS (
        SELECT 
            SUM(KEYWORD_FLAG) AS kw_flagged,
            SUM(ML_FLAG) AS ml_flagged,
            SUM(CASE WHEN KEYWORD_FLAG = 1 AND ACTUAL_VIOLATION = 1 THEN 1 ELSE 0 END) AS kw_true_hits,
            SUM(CASE WHEN ML_FLAG = 1 AND ACTUAL_VIOLATION = 1 THEN 1 ELSE 0 END) AS ml_true_hits,
            SUM(ACTUAL_VIOLATION) AS total_violations,
            COUNT(*) AS total_emails
        FROM ML.MODEL_COMPARISON
    )
    SELECT
        total_emails AS "Total Emails",
        total_violations AS "Actual Violations",
        kw_flagged AS "Keyword Flags",
        ml_flagged AS "ML Flags",
        kw_flagged - ml_flagged AS "Reduction in Reviews",
        ROUND((kw_flagged - ml_flagged) * 100.0 / NULLIF(kw_flagged, 0), 1) AS "Workload Reduction %",
        ROUND(kw_true_hits * 100.0 / NULLIF(kw_flagged, 0), 1) AS "Keyword Hit Rate %",
        ROUND(ml_true_hits * 100.0 / NULLIF(ml_flagged, 0), 1) AS "ML Hit Rate %"
    FROM totals
""")

print("=" * 70)
print("                INVESTIGATION EFFICIENCY ANALYSIS")
print("=" * 70)
efficiency_df.show()

# %% [markdown]
# ## 7. Detection by Violation Type
# 
# Which types of violations does each method catch best?

# %% Break down by compliance label
print("--- Detection Rate by Violation Type ---")
session.sql("""
    SELECT 
        COMPLIANCE_LABEL,
        COUNT(*) AS total_count,
        SUM(KEYWORD_FLAG) AS keyword_flagged,
        SUM(ML_FLAG) AS ml_flagged,
        ROUND(SUM(KEYWORD_FLAG) * 100.0 / COUNT(*), 1) AS keyword_detection_pct,
        ROUND(SUM(ML_FLAG) * 100.0 / COUNT(*), 1) AS ml_detection_pct
    FROM ML.MODEL_COMPARISON
    GROUP BY COMPLIANCE_LABEL
    ORDER BY 
        CASE COMPLIANCE_LABEL 
            WHEN 'CLEAN' THEN 999 
            ELSE 1 
        END,
        total_count DESC
""").show()

# %% [markdown]
# ## 8. Summary Visualization (Text-based)
# 
# Visual summary of the comparison.

# %% Print summary
results = session.sql("""
    SELECT 
        SUM(KEYWORD_FLAG) AS kw_flags,
        SUM(ML_FLAG) AS ml_flags,
        SUM(CASE WHEN KEYWORD_FLAG = 1 AND ACTUAL_VIOLATION = 1 THEN 1 ELSE 0 END) AS kw_tp,
        SUM(CASE WHEN KEYWORD_FLAG = 1 AND ACTUAL_VIOLATION = 0 THEN 1 ELSE 0 END) AS kw_fp,
        SUM(CASE WHEN ML_FLAG = 1 AND ACTUAL_VIOLATION = 1 THEN 1 ELSE 0 END) AS ml_tp,
        SUM(CASE WHEN ML_FLAG = 1 AND ACTUAL_VIOLATION = 0 THEN 1 ELSE 0 END) AS ml_fp,
        SUM(ACTUAL_VIOLATION) AS total_violations
    FROM ML.MODEL_COMPARISON
""").collect()[0]

print("\n" + "=" * 70)
print("                        BENCHMARK SUMMARY")
print("=" * 70)
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                        KEYWORD RULES                                │
├─────────────────────────────────────────────────────────────────────┤
│  Emails Flagged:     {results['KW_FLAGS']:4d}                                        │
│  True Positives:     {results['KW_TP']:4d}  (actual violations caught)              │
│  False Positives:    {results['KW_FP']:4d}  (clean emails wrongly flagged)          │
│  Precision:          {results['KW_TP']*100/(results['KW_TP']+results['KW_FP']) if (results['KW_TP']+results['KW_FP']) > 0 else 0:5.1f}%                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        ML CLASSIFIER                                │
├─────────────────────────────────────────────────────────────────────┤
│  Emails Flagged:     {results['ML_FLAGS']:4d}                                        │
│  True Positives:     {results['ML_TP']:4d}  (actual violations caught)              │
│  False Positives:    {results['ML_FP']:4d}  (clean emails wrongly flagged)          │
│  Precision:          {results['ML_TP']*100/(results['ML_TP']+results['ML_FP']) if (results['ML_TP']+results['ML_FP']) > 0 else 0:5.1f}%                                       │
└─────────────────────────────────────────────────────────────────────┘

📊 Total Violations in Dataset: {results['TOTAL_VIOLATIONS']}
""")

# %% [markdown]
# ## 9. Snowflake In-House vs Frontier Model Benchmark
# 
# Compare Snowflake's cost-effective in-house models against larger frontier models.

# %% Define models to benchmark
MODELS_TO_TEST = {
    "snowflake-arctic-embed": "Snowflake in-house (fast, cheap)",
    "mistral-large2": "Frontier model (more capable, higher cost)",
}

# %% Benchmark classification accuracy
print("🏁 Benchmarking: Snowflake In-House vs Frontier Models")
print("=" * 70)

# Test on a sample of emails
benchmark_df = session.sql("""
WITH test_emails AS (
    SELECT 
        EMAIL_ID,
        BODY,
        COMPLIANCE_LABEL,
        CASE WHEN COMPLIANCE_LABEL != 'CLEAN' THEN 1 ELSE 0 END AS ACTUAL_VIOLATION
    FROM RAW_DATA.EMAILS
    LIMIT 20  -- Sample for demo speed
),
model_predictions AS (
    SELECT 
        EMAIL_ID,
        ACTUAL_VIOLATION,
        COMPLIANCE_LABEL,
        
        -- Snowflake in-house model (mistral-7b - smaller, faster)
        CASE 
            WHEN SNOWFLAKE.CORTEX.COMPLETE(
                'mistral-7b',
                CONCAT('Is this email a compliance violation? Answer only YES or NO. Email: ', LEFT(BODY, 500))
            ) ILIKE '%YES%' THEN 1 ELSE 0 
        END AS INHOUSE_PREDICTION,
        
        -- Frontier model (mistral-large2 - larger, more capable)
        CASE 
            WHEN SNOWFLAKE.CORTEX.COMPLETE(
                'mistral-large2',
                CONCAT('Is this email a compliance violation? Answer only YES or NO. Email: ', LEFT(BODY, 500))
            ) ILIKE '%YES%' THEN 1 ELSE 0 
        END AS FRONTIER_PREDICTION
        
    FROM test_emails
)
SELECT * FROM model_predictions
""")

print("--- Model Predictions Comparison ---")
benchmark_df.show(10)

# %% Calculate benchmark metrics
print("\n--- Benchmark Results: In-House vs Frontier ---")
session.sql("""
WITH predictions AS (
    SELECT 
        CASE WHEN COMPLIANCE_LABEL != 'CLEAN' THEN 1 ELSE 0 END AS ACTUAL,
        CASE 
            WHEN SNOWFLAKE.CORTEX.COMPLETE(
                'mistral-7b',
                CONCAT('Is this a compliance violation? YES or NO only. Email: ', LEFT(BODY, 300))
            ) ILIKE '%YES%' THEN 1 ELSE 0 
        END AS INHOUSE,
        CASE 
            WHEN SNOWFLAKE.CORTEX.COMPLETE(
                'mistral-large2',
                CONCAT('Is this a compliance violation? YES or NO only. Email: ', LEFT(BODY, 300))
            ) ILIKE '%YES%' THEN 1 ELSE 0 
        END AS FRONTIER
    FROM RAW_DATA.EMAILS
    LIMIT 15
)
SELECT 
    'mistral-7b (In-House)' AS model,
    SUM(CASE WHEN INHOUSE = ACTUAL THEN 1 ELSE 0 END) AS correct,
    COUNT(*) AS total,
    ROUND(SUM(CASE WHEN INHOUSE = ACTUAL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS accuracy_pct,
    '~0.001 credits/call' AS est_cost
FROM predictions

UNION ALL

SELECT 
    'mistral-large2 (Frontier)',
    SUM(CASE WHEN FRONTIER = ACTUAL THEN 1 ELSE 0 END),
    COUNT(*),
    ROUND(SUM(CASE WHEN FRONTIER = ACTUAL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1),
    '~0.01 credits/call'
FROM predictions
""").show()

# %% Print cost-performance tradeoff
print("""
┌─────────────────────────────────────────────────────────────────────┐
│              SNOWFLAKE MODEL SELECTION GUIDE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  🏠 IN-HOUSE MODELS (mistral-7b, snowflake-arctic)                  │
│  ✅ Lower cost per inference                                        │
│  ✅ Faster response times                                           │
│  ✅ Good for high-volume, simpler tasks                             │
│  ⚠️  May miss nuanced violations                                    │
│                                                                     │
│  🚀 FRONTIER MODELS (mistral-large2, llama3.1-70b)                  │
│  ✅ Higher accuracy on complex tasks                                │
│  ✅ Better reasoning and explanation                                │
│  ✅ Handles ambiguous cases better                                  │
│  ⚠️  Higher cost, slower                                            │
│                                                                     │
│  💡 RECOMMENDED ARCHITECTURE                                        │
│  ─────────────────────────────────────────────────────────────────  │
│  1. In-house model for initial screening (high volume)              │
│  2. Frontier model for flagged items (deep analysis)                │
│  3. Fine-tuned in-house model for domain-specific tasks             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# %% [markdown]
# ## ✅ Part 2.1 Complete!
# 
# **Key Insights:**
# - ML models can significantly reduce false positives
# - Keyword rules often flag legitimate emails with common words
# - ML learns contextual patterns, not just keyword presence
# - **In-house models: cost-effective for high-volume screening**
# - **Frontier models: better accuracy for complex cases**
# 
# **Business Value:**
# - Fewer alerts to investigate
# - Higher "hit rate" per investigation
# - Reduced analyst fatigue
# - Better coverage of sophisticated violations
# - **Cost optimization via tiered model architecture**
# 
# **Next:** Run `05_llm_compliance_pipeline.py` to add LLM-based analysis.

