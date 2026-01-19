# Snowflake Notebook / Python Worksheet
# Part 3.1: Email Content Search & Pattern Recognition
#
# This script demonstrates:
# - Generating embeddings with AI_EMBED
# - Creating Cortex Search services
# - Semantic similarity search for compliance patterns
#
# Prerequisites:
# - Run 01-06 scripts first
# - Cortex functions enabled
#
# Time: ~4-5 minutes

# %% [markdown]
# # Part 3.1: Email Content Search & Pattern Recognition
# 
# **Goal:** Enable semantic search across email archives.
# 
# Traditional keyword search misses:
# - Paraphrased violations
# - Contextually similar but lexically different text
# - "Find me emails like this one"
# 
# Vector embeddings + Cortex Search enable semantic understanding.

# %% Setup
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col

session = get_active_session()

print(f"Connected as: {session.get_current_user()}")
session.use_database("ML_COMPLIANCE_DEMO")
session.use_schema("SEARCH")

# %% [markdown]
# ## 1. Generate Email Embeddings
# 
# Use AI_EMBED to create vector representations of email content.
# Similar emails will have similar vectors.

# %% Generate embeddings for all emails
print("🔢 Generating embeddings with AI_EMBED...")

session.sql("""
CREATE OR REPLACE TABLE SEARCH.EMAIL_EMBEDDINGS AS
SELECT 
    EMAIL_ID,
    SENDER,
    RECIPIENT,
    SUBJECT,
    LEFT(BODY, 500) AS BODY_PREVIEW,
    COMPLIANCE_LABEL,
    SENT_AT,
    
    -- Generate embedding from subject + body
    SNOWFLAKE.CORTEX.EMBED_TEXT_768(
        'e5-base-v2',
        CONCAT(SUBJECT, ' ', BODY)
    ) AS EMBEDDING
    
FROM RAW_DATA.EMAILS
""").collect()

print("✅ Embeddings generated and stored")

# %% Verify embeddings
print("--- Embeddings Preview ---")
session.sql("""
    SELECT 
        EMAIL_ID,
        LEFT(SUBJECT, 40) AS SUBJECT,
        COMPLIANCE_LABEL,
        ARRAY_SIZE(EMBEDDING) AS EMBEDDING_DIM
    FROM SEARCH.EMAIL_EMBEDDINGS
    LIMIT 5
""").show()

# %% Check embedding dimensions
session.sql("""
    SELECT 
        COUNT(*) AS total_emails,
        ARRAY_SIZE(EMBEDDING) AS embedding_dimensions
    FROM SEARCH.EMAIL_EMBEDDINGS
    GROUP BY 2
""").show()

# %% [markdown]
# ## 2. Create Cortex Search Service
# 
# Cortex Search provides optimized semantic search over embeddings.

# %% Create the search service
print("🔍 Creating Cortex Search service...")

session.sql("""
CREATE OR REPLACE CORTEX SEARCH SERVICE SEARCH.EMAIL_SEARCH_SERVICE
ON BODY_PREVIEW
ATTRIBUTES SUBJECT, COMPLIANCE_LABEL, SENDER, RECIPIENT
WAREHOUSE = ML_COMPLIANCE_WH
TARGET_LAG = '1 hour'
AS (
    SELECT 
        EMAIL_ID,
        SUBJECT,
        BODY_PREVIEW,
        COMPLIANCE_LABEL,
        SENDER,
        RECIPIENT,
        SENT_AT::STRING AS SENT_AT
    FROM SEARCH.EMAIL_EMBEDDINGS
)
""").collect()

print("✅ Cortex Search service created: SEARCH.EMAIL_SEARCH_SERVICE")

# %% [markdown]
# ## 3. Semantic Search Demo
# 
# Search for emails semantically similar to a query.

# %% Search for insider trading patterns
print("--- Search: 'confidential merger announcement trading tip' ---")

session.sql("""
SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'ML_COMPLIANCE_DEMO.SEARCH.EMAIL_SEARCH_SERVICE',
    '{
        "query": "confidential merger announcement trading tip",
        "columns": ["EMAIL_ID", "SUBJECT", "BODY_PREVIEW", "COMPLIANCE_LABEL"],
        "limit": 5
    }'
) AS RESULTS
""").show()

# %% Search for data exfiltration patterns
print("--- Search: 'sending client information to external email' ---")

session.sql("""
SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'ML_COMPLIANCE_DEMO.SEARCH.EMAIL_SEARCH_SERVICE',
    '{
        "query": "sending client information to external email",
        "columns": ["EMAIL_ID", "SUBJECT", "BODY_PREVIEW", "COMPLIANCE_LABEL"],
        "limit": 5
    }'
) AS RESULTS
""").show()

# %% [markdown]
# ## 4. Find Similar Emails
# 
# Given a known violation, find similar patterns in the archive.

# %% Get a known insider trading email
print("--- Finding emails similar to a known violation ---")

# First, get a violation example
example_email = session.sql("""
    SELECT EMAIL_ID, SUBJECT, LEFT(BODY_PREVIEW, 200) AS PREVIEW
    FROM SEARCH.EMAIL_EMBEDDINGS
    WHERE COMPLIANCE_LABEL = 'INSIDER_TRADING'
    LIMIT 1
""").collect()

if example_email:
    print(f"Example violation:")
    print(f"  Subject: {example_email[0]['SUBJECT']}")
    print(f"  Preview: {example_email[0]['PREVIEW'][:100]}...")
    
    # Search for similar emails
    print("\n--- Similar emails in archive ---")
    session.sql(f"""
    SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
        'ML_COMPLIANCE_DEMO.SEARCH.EMAIL_SEARCH_SERVICE',
        '{{
            "query": "{example_email[0]['PREVIEW'][:200].replace('"', '')}",
            "columns": ["EMAIL_ID", "SUBJECT", "COMPLIANCE_LABEL"],
            "limit": 5
        }}'
    ) AS SIMILAR_EMAILS
    """).show()

# %% [markdown]
# ## 5. Vector Similarity Search (Manual)
# 
# For more control, compute similarity directly using embeddings.

# %% Find most similar emails using cosine similarity
print("--- Manual Vector Similarity Search ---")

session.sql("""
-- Get embedding for a known insider trading email
WITH reference_email AS (
    SELECT EMBEDDING AS ref_embedding
    FROM SEARCH.EMAIL_EMBEDDINGS
    WHERE COMPLIANCE_LABEL = 'INSIDER_TRADING'
    LIMIT 1
),
-- Calculate cosine similarity with all other emails
similarities AS (
    SELECT 
        e.EMAIL_ID,
        e.SUBJECT,
        e.COMPLIANCE_LABEL,
        VECTOR_COSINE_SIMILARITY(e.EMBEDDING, r.ref_embedding) AS similarity
    FROM SEARCH.EMAIL_EMBEDDINGS e
    CROSS JOIN reference_email r
    WHERE e.COMPLIANCE_LABEL != 'INSIDER_TRADING'  -- Exclude the reference itself
)
SELECT 
    EMAIL_ID,
    LEFT(SUBJECT, 50) AS SUBJECT,
    COMPLIANCE_LABEL,
    ROUND(similarity, 4) AS SIMILARITY_SCORE
FROM similarities
ORDER BY similarity DESC
LIMIT 10
""").show()

# %% [markdown]
# ## 6. Cluster Analysis
# 
# Group similar emails to find patterns.

# %% Find potential clusters by compliance label
print("--- Average Similarity Within vs Between Labels ---")

session.sql("""
WITH sample_pairs AS (
    SELECT 
        a.EMAIL_ID AS email_a,
        b.EMAIL_ID AS email_b,
        a.COMPLIANCE_LABEL AS label_a,
        b.COMPLIANCE_LABEL AS label_b,
        VECTOR_COSINE_SIMILARITY(a.EMBEDDING, b.EMBEDDING) AS similarity
    FROM SEARCH.EMAIL_EMBEDDINGS a
    CROSS JOIN SEARCH.EMAIL_EMBEDDINGS b
    WHERE a.EMAIL_ID < b.EMAIL_ID  -- Avoid duplicates
    LIMIT 1000  -- Sample for performance
)
SELECT 
    CASE 
        WHEN label_a = label_b THEN 'Same Label'
        ELSE 'Different Label'
    END AS comparison_type,
    ROUND(AVG(similarity), 4) AS avg_similarity,
    ROUND(MIN(similarity), 4) AS min_similarity,
    ROUND(MAX(similarity), 4) AS max_similarity,
    COUNT(*) AS pair_count
FROM sample_pairs
GROUP BY 1
""").show()

# %% [markdown]
# ## 7. Anomaly Detection via Embeddings
# 
# Find emails that don't fit typical patterns.

# %% Find outlier emails (low similarity to centroid)
print("--- Potential Anomalies (emails unlike their category) ---")

session.sql("""
WITH label_centroids AS (
    -- Approximate centroid as average of vectors in each category
    SELECT 
        COMPLIANCE_LABEL,
        ARRAY_AGG(EMBEDDING) AS embeddings_list
    FROM SEARCH.EMAIL_EMBEDDINGS
    GROUP BY COMPLIANCE_LABEL
),
email_similarities AS (
    SELECT 
        e.EMAIL_ID,
        e.SUBJECT,
        e.COMPLIANCE_LABEL,
        -- Similarity to own category (using first email in category as proxy)
        (
            SELECT VECTOR_COSINE_SIMILARITY(e.EMBEDDING, e2.EMBEDDING)
            FROM SEARCH.EMAIL_EMBEDDINGS e2
            WHERE e2.COMPLIANCE_LABEL = e.COMPLIANCE_LABEL
              AND e2.EMAIL_ID != e.EMAIL_ID
            LIMIT 1
        ) AS similarity_to_category
    FROM SEARCH.EMAIL_EMBEDDINGS e
)
SELECT 
    EMAIL_ID,
    LEFT(SUBJECT, 50) AS SUBJECT,
    COMPLIANCE_LABEL,
    ROUND(similarity_to_category, 4) AS CATEGORY_FIT
FROM email_similarities
WHERE similarity_to_category IS NOT NULL
ORDER BY similarity_to_category ASC  -- Lowest = most unusual
LIMIT 10
""").show()

# %% [markdown]
# ## 8. Search Use Cases Summary

# %% Print use cases
print("""
┌─────────────────────────────────────────────────────────────────────┐
│                VECTOR SEARCH USE CASES FOR COMPLIANCE               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  🔍 INVESTIGATION SUPPORT                                           │
│  "Find all emails similar to this known violation"                  │
│  → Discover related communications in an investigation              │
│                                                                     │
│  🎯 PATTERN DETECTION                                               │
│  "Find emails about merger activity"                                │
│  → Catches paraphrased content keyword search misses                │
│                                                                     │
│  ⚠️  ANOMALY DETECTION                                              │
│  "Which emails don't fit normal patterns?"                          │
│  → Flag unusual communications for review                           │
│                                                                     │
│  📊 CLUSTERING                                                      │
│  "Group similar emails together"                                    │
│  → Identify communication patterns, social networks                 │
│                                                                     │
│  🔄 NEAR-DUPLICATE DETECTION                                        │
│  "Find emails with same content to different recipients"            │
│  → Detect potential information spreading                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# %% [markdown]
# ## ✅ Part 3.1 Complete!
# 
# **What we learned:**
# - Generating embeddings with `AI_EMBED` / `EMBED_TEXT_768`
# - Creating Cortex Search services for semantic search
# - Manual vector similarity with `VECTOR_COSINE_SIMILARITY`
# - Clustering and anomaly detection patterns
# 
# **Key capabilities:**
# - "Find similar" searches that keyword search can't do
# - Pattern discovery across large email archives
# - Investigation support for compliance teams
# 
# **Next:** Run `08_pattern_evolution.py` for model maintenance strategies.

