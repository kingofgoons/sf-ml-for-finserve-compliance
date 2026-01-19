# Snowflake Notebook / Python Worksheet
# Part 2.3: Fine-Tuning for Financial Communication Language
#
# This script demonstrates:
# - Preparing fine-tuning datasets
# - Running CORTEX.FINETUNE jobs
# - Evaluating fine-tuned vs base models
#
# Prerequisites:
# - Run 01-05 scripts first
# - Fine-tuning enabled in your account
#
# Time: ~4-5 minutes (setup), fine-tuning job runs async

# %% [markdown]
# # Part 2.3: Fine-Tuning for Financial Communication Language
# 
# **Goal:** Customize an LLM for hedge fund compliance terminology.
# 
# Base LLMs are trained on general text. Fine-tuning lets us:
# - Teach domain-specific terminology (MNPI, Chinese walls, blackout periods)
# - Improve accuracy on our specific compliance categories
# - Create a specialized model for our use case
# 
# Cortex Fine-tuning runs entirely within Snowflake — no data leaves your account.

# %% Setup
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col

session = get_active_session()

print(f"Connected as: {session.get_current_user()}")
session.use_database("ML_COMPLIANCE_DEMO")
session.use_schema("ML")

# %% [markdown]
# ## 1. Prepare Fine-Tuning Dataset
# 
# Fine-tuning requires prompt/completion pairs in a specific format.
# We'll create training examples from our labeled email data.

# %% Create fine-tuning training table
session.sql("""
CREATE OR REPLACE TABLE ML.FINETUNE_TRAINING_DATA (
    prompt STRING,
    completion STRING
)
""").collect()
print("✅ Created fine-tuning training table")

# %% Generate training examples from labeled emails
# Format: prompt asks for classification, completion is the correct label + reasoning

session.sql("""
INSERT INTO ML.FINETUNE_TRAINING_DATA (prompt, completion)
SELECT 
    CONCAT(
        'Classify this hedge fund email for compliance violations. ',
        'Categories: INSIDER_TRADING, CONFIDENTIALITY_BREACH, PERSONAL_TRADING, INFO_BARRIER_VIOLATION, CLEAN. ',
        'Email: ', BODY, ' ',
        'Classification:'
    ) AS prompt,
    CASE COMPLIANCE_LABEL
        WHEN 'INSIDER_TRADING' THEN 
            'INSIDER_TRADING - This email contains material non-public information (MNPI) or tips about trading before public announcements.'
        WHEN 'CONFIDENTIALITY_BREACH' THEN 
            'CONFIDENTIALITY_BREACH - This email shares confidential client or fund information with unauthorized recipients.'
        WHEN 'PERSONAL_TRADING' THEN 
            'PERSONAL_TRADING - This email discusses personal trading activity that may violate pre-clearance or disclosure requirements.'
        WHEN 'INFO_BARRIER_VIOLATION' THEN 
            'INFO_BARRIER_VIOLATION - This email crosses information barriers (Chinese walls) between Research and Trading departments.'
        ELSE 
            'CLEAN - This email contains normal business communication with no compliance concerns.'
    END AS completion
FROM RAW_DATA.EMAILS
""").collect()

print("--- Fine-Tuning Dataset Preview ---")
session.table("ML.FINETUNE_TRAINING_DATA").show(5)

# %% Check dataset size
count = session.table("ML.FINETUNE_TRAINING_DATA").count()
print(f"\n📊 Training dataset size: {count} examples")
print("   (Cortex fine-tuning requires minimum ~100 examples for best results)")

# %% [markdown]
# ## 2. Validate Training Data Format
# 
# Ensure our data meets fine-tuning requirements.

# %% Validate prompt/completion lengths
session.sql("""
    SELECT 
        'Prompts' AS field,
        MIN(LENGTH(prompt)) AS min_length,
        MAX(LENGTH(prompt)) AS max_length,
        AVG(LENGTH(prompt))::INT AS avg_length
    FROM ML.FINETUNE_TRAINING_DATA
    
    UNION ALL
    
    SELECT 
        'Completions',
        MIN(LENGTH(completion)),
        MAX(LENGTH(completion)),
        AVG(LENGTH(completion))::INT
    FROM ML.FINETUNE_TRAINING_DATA
""").show()

# %% Check label distribution in training data
print("--- Training Data Label Distribution ---")
session.sql("""
    SELECT 
        SPLIT_PART(completion, ' - ', 1) AS label,
        COUNT(*) AS count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM ML.FINETUNE_TRAINING_DATA
    GROUP BY 1
    ORDER BY count DESC
""").show()

# %% [markdown]
# ## 3. Create Fine-Tuning Job
# 
# Launch the fine-tuning job. This runs asynchronously in Snowflake.

# %% Start fine-tuning job
# Note: This creates an async job that may take 15-60 minutes to complete

FINETUNE_JOB_NAME = "compliance_classifier_ft"

print("🚀 Starting fine-tuning job...")
print("   This will run asynchronously. Check status with SHOW CORTEX FINE_TUNING JOBS.")

# The FINETUNE function creates a fine-tuned model
session.sql(f"""
SELECT SNOWFLAKE.CORTEX.FINETUNE(
    'CREATE',                                    -- Action
    '{FINETUNE_JOB_NAME}',                      -- Job name  
    'mistral-7b',                               -- Base model
    'SELECT prompt, completion FROM ML.FINETUNE_TRAINING_DATA',  -- Training data
    {{}}                                         -- Options (use defaults)
)
""").show()

# %% [markdown]
# ## 4. Monitor Fine-Tuning Progress
# 
# Check the status of your fine-tuning job.

# %% Check job status
print("--- Fine-Tuning Job Status ---")
session.sql("""
    SELECT SNOWFLAKE.CORTEX.FINETUNE('DESCRIBE', 'compliance_classifier_ft')
""").show()

# %% List all fine-tuning jobs
print("--- All Fine-Tuning Jobs ---")
session.sql("""
    SHOW CORTEX FINE_TUNING JOBS
""").show()

# %% [markdown]
# ## 5. Use Fine-Tuned Model (After Job Completes)
# 
# Once the job finishes, the fine-tuned model is available for inference.

# %% Test fine-tuned model (only works after job completes)
# The model name follows pattern: <job_name>

print("--- Testing Fine-Tuned Model ---")
print("(This will only work after the fine-tuning job completes)")

try:
    # Use the fine-tuned model
    result = session.sql(f"""
        SELECT 
            EMAIL_ID,
            LEFT(SUBJECT, 40) AS SUBJECT,
            COMPLIANCE_LABEL AS ACTUAL,
            SNOWFLAKE.CORTEX.COMPLETE(
                '{FINETUNE_JOB_NAME}',  -- Fine-tuned model name
                CONCAT(
                    'Classify this hedge fund email for compliance violations. ',
                    'Categories: INSIDER_TRADING, CONFIDENTIALITY_BREACH, PERSONAL_TRADING, INFO_BARRIER_VIOLATION, CLEAN. ',
                    'Email: ', BODY, ' ',
                    'Classification:'
                )
            ) AS FT_PREDICTION
        FROM RAW_DATA.EMAILS
        WHERE COMPLIANCE_LABEL != 'CLEAN'
        LIMIT 5
    """)
    result.show()
except Exception as e:
    print(f"⏳ Fine-tuning job not yet complete: {e}")

# %% [markdown]
# ## 6. Compare Base vs Fine-Tuned Model
# 
# Evaluate improvement from fine-tuning.

# %% Create comparison (when fine-tuning completes)
print("""
┌─────────────────────────────────────────────────────────────────────┐
│                FINE-TUNING EVALUATION FRAMEWORK                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  When fine-tuning completes, compare:                               │
│                                                                     │
│  1. ACCURACY                                                        │
│     Base model vs fine-tuned on held-out test set                   │
│                                                                     │
│  2. DOMAIN UNDERSTANDING                                            │
│     Does it recognize "MNPI", "Chinese wall", "blackout period"?    │
│                                                                     │
│  3. CONSISTENCY                                                     │
│     Same input → same output format?                                │
│                                                                     │
│  4. EDGE CASES                                                      │
│     Ambiguous emails that base model struggles with                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# %% Create evaluation query (template for when job completes)
EVAL_QUERY = f"""
-- Run this after fine-tuning completes to compare models

WITH predictions AS (
    SELECT 
        EMAIL_ID,
        COMPLIANCE_LABEL AS ACTUAL,
        
        -- Base model prediction
        SNOWFLAKE.CORTEX.COMPLETE(
            'mistral-7b',
            CONCAT('Classify: ', LEFT(BODY, 500), ' Category:')
        ) AS BASE_PRED,
        
        -- Fine-tuned model prediction  
        SNOWFLAKE.CORTEX.COMPLETE(
            '{FINETUNE_JOB_NAME}',
            CONCAT('Classify: ', LEFT(BODY, 500), ' Category:')
        ) AS FT_PRED
        
    FROM RAW_DATA.EMAILS
    LIMIT 20
)
SELECT 
    SUM(CASE WHEN BASE_PRED ILIKE '%' || ACTUAL || '%' THEN 1 ELSE 0 END) AS BASE_CORRECT,
    SUM(CASE WHEN FT_PRED ILIKE '%' || ACTUAL || '%' THEN 1 ELSE 0 END) AS FT_CORRECT,
    COUNT(*) AS TOTAL
FROM predictions;
"""

print("--- Evaluation Query (save for later) ---")
print(EVAL_QUERY)

# %% [markdown]
# ## 7. Fine-Tuning Best Practices

# %% Print best practices
print("""
┌─────────────────────────────────────────────────────────────────────┐
│                  FINE-TUNING BEST PRACTICES                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📊 DATA QUALITY                                                    │
│  • Minimum 100 examples (500+ recommended)                          │
│  • Balanced across categories                                       │
│  • Clean, consistent prompt/completion format                       │
│  • Include edge cases and hard examples                             │
│                                                                     │
│  🎯 PROMPT DESIGN                                                   │
│  • Consistent prompt structure across examples                      │
│  • Include context about the task                                   │
│  • Match inference prompt format to training                        │
│                                                                     │
│  ✅ COMPLETION FORMAT                                               │
│  • Structured, parseable output                                     │
│  • Include reasoning (helps model learn patterns)                   │
│  • Keep completions focused and concise                             │
│                                                                     │
│  📈 EVALUATION                                                      │
│  • Hold out 10-20% for testing                                      │
│  • Compare base vs fine-tuned on same test set                      │
│  • Check for overfitting on rare categories                         │
│                                                                     │
│  💰 COST CONSIDERATIONS                                             │
│  • Fine-tuning has one-time training cost                           │
│  • Inference cost similar to base model                             │
│  • ROI: better accuracy = fewer false positives to review           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# %% [markdown]
# ## 8. Cancel or Clean Up (Optional)

# %% Cancel job if needed
# session.sql("""
#     SELECT SNOWFLAKE.CORTEX.FINETUNE('CANCEL', 'compliance_classifier_ft')
# """).collect()
# print("Job cancelled")

# %% [markdown]
# ## ✅ Part 2.3 Complete!
# 
# **What we learned:**
# - Preparing fine-tuning datasets (prompt/completion pairs)
# - Launching async fine-tuning jobs with CORTEX.FINETUNE
# - Monitoring job progress
# - Evaluation framework for comparing base vs fine-tuned
# 
# **Fine-tuning benefits for compliance:**
# - Better understanding of financial terminology
# - Consistent output format
# - Higher accuracy on domain-specific classifications
# - All data stays in Snowflake (privacy/security)
# 
# **Next:** Run `07_vector_search.py` for semantic email search.

