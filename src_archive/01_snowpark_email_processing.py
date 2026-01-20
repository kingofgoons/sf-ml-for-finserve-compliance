# Snowflake Notebook / Python Worksheet
# Part 1.1: Snowpark Basics for Email Data Processing
#
# This script demonstrates:
# - DataFrame operations for email analysis
# - Creating UDFs for message processing
#
# Prerequisites:
# - Run 00_setup.sql first
# - Upload emails_synthetic.csv to @RAW_DATA.EMAIL_DATA_STAGE
#
# Time: ~4-5 minutes

# %% [markdown]
# # Part 1.1: Snowpark Basics for Email Data Processing
# 
# **Goal:** Load and process email archive data using Snowpark DataFrames.
# 
# We'll demonstrate:
# - Loading CSV data from a stage
# - DataFrame operations: filtering, grouping, aggregations
# - Pattern analysis for compliance signals
# - Creating UDFs for text feature extraction

# %% Setup - Get Active Session
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import (
    col,
    count,
    avg,
    sum as sum_,
    when,
    hour,
    length,
    lit,
    call_udf,
)
from snowflake.snowpark.types import StringType, IntegerType

session = get_active_session()

# Verify connection
print(f"Connected as: {session.get_current_user()}")
print(f"Role: {session.get_current_role()}")
print(f"Database: {session.get_current_database()}")
print(f"Warehouse: {session.get_current_warehouse()}")

# %% [markdown]
# ## 1. Load Email Data from Stage
# 
# First, upload `data/emails_synthetic.csv` to the stage using Snowsight:
# 1. Navigate to **Data → Databases → ML_COMPLIANCE_DEMO → RAW_DATA → Stages**
# 2. Click on `EMAIL_DATA_STAGE`
# 3. Click **+ Files** and upload `emails_synthetic.csv`
# 
# Or use this PUT command in a SQL worksheet:
# ```sql
# PUT file:///path/to/emails_synthetic.csv @RAW_DATA.EMAIL_DATA_STAGE AUTO_COMPRESS=FALSE;
# ```

# %% Load data from stage into table
session.use_schema("RAW_DATA")

# Read CSV from stage
df = session.read.options({
    "FIELD_OPTIONALLY_ENCLOSED_BY": '"',
    "SKIP_HEADER": 1,
    "NULL_IF": "('')",
}).schema(
    "EMAIL_ID STRING, SENDER STRING, RECIPIENT STRING, CC STRING, "
    "SUBJECT STRING, BODY STRING, SENT_AT STRING, "
    "SENDER_DEPT STRING, RECIPIENT_DEPT STRING, COMPLIANCE_LABEL STRING"
).csv("@EMAIL_DATA_STAGE/emails_synthetic.csv")

print(f"Loaded {df.count()} emails from stage")
df.show(5)

# %% Write to EMAILS table (idempotent)
session.sql("TRUNCATE TABLE IF EXISTS RAW_DATA.EMAILS").collect()

df.write.mode("append").save_as_table("RAW_DATA.EMAILS")
print("✅ Data written to RAW_DATA.EMAILS")

# Verify
session.table("EMAILS").count()

# %% [markdown]
# ## 2. DataFrame Operations: Email Analysis
# 
# Snowpark DataFrames let us write Pythonic data transformations that execute in Snowflake.

# %% 2.1 Email volume by department
emails = session.table("EMAILS")

print("--- Email Volume by Department ---")
dept_stats = (
    emails
    .group_by("SENDER_DEPT")
    .agg(
        count("*").alias("EMAILS_SENT"),
        avg(length("BODY")).alias("AVG_BODY_LENGTH"),
    )
    .order_by(col("EMAILS_SENT").desc())
)
dept_stats.show()

# %% 2.2 Compliance label distribution
print("--- Compliance Label Distribution ---")
label_dist = (
    emails
    .group_by("COMPLIANCE_LABEL")
    .count()
    .order_by(col("COUNT").desc())
)
label_dist.show()

# %% 2.3 Cross-department communication (info barrier risk)
print("--- Cross-Department Communication ---")
cross_dept = (
    emails
    .filter(col("SENDER_DEPT") != col("RECIPIENT_DEPT"))
    .group_by("SENDER_DEPT", "RECIPIENT_DEPT")
    .count()
    .order_by(col("COUNT").desc())
    .limit(10)
)
cross_dept.show()

# %% 2.4 After-hours email activity (risk signal)
print("--- After-Hours Activity (before 8am or after 6pm) ---")

# Cast SENT_AT to timestamp if needed, then extract hour
emails_with_hour = emails.with_column(
    "HOUR", 
    hour(col("SENT_AT").cast("TIMESTAMP"))
)

after_hours = (
    emails_with_hour
    .with_column(
        "IS_AFTER_HOURS",
        when((col("HOUR") < 8) | (col("HOUR") >= 18), lit(1)).otherwise(lit(0))
    )
    .group_by("SENDER_DEPT")
    .agg(
        count("*").alias("TOTAL_EMAILS"),
        sum_("IS_AFTER_HOURS").alias("AFTER_HOURS_COUNT"),
    )
    .with_column(
        "AFTER_HOURS_PCT",
        (col("AFTER_HOURS_COUNT") / col("TOTAL_EMAILS") * 100)
    )
    .order_by(col("AFTER_HOURS_PCT").desc())
)
after_hours.show()

# %% 2.5 Research <-> Trading barrier monitoring
print("--- Research <-> Trading Communications (Info Barrier) ---")

barrier_comms = (
    emails
    .filter(
        ((col("SENDER_DEPT") == "Research") & (col("RECIPIENT_DEPT") == "Trading")) |
        ((col("SENDER_DEPT") == "Trading") & (col("RECIPIENT_DEPT") == "Research"))
    )
    .select("EMAIL_ID", "SENDER", "RECIPIENT", "SUBJECT", "COMPLIANCE_LABEL")
)

print(f"Found {barrier_comms.count()} emails crossing Research/Trading barrier:")
barrier_comms.show()

# %% [markdown]
# ## 3. User-Defined Functions (UDFs)
# 
# Create Python UDFs for text feature extraction. These run inside Snowflake's Python runtime.

# %% 3.1 Create urgency detection UDF
session.sql("""
CREATE OR REPLACE FUNCTION RAW_DATA.DETECT_URGENCY(text STRING)
RETURNS INT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
HANDLER = 'detect_urgency'
AS $$
def detect_urgency(text: str) -> int:
    '''Return urgency score based on keyword presence (0-5).'''
    if text is None:
        return 0
    text_upper = text.upper()
    urgency_keywords = ["URGENT", "ASAP", "IMMEDIATELY", "TIME SENSITIVE", "ACT NOW"]
    return sum(1 for kw in urgency_keywords if kw in text_upper)
$$
""").collect()
print("✅ Created UDF: RAW_DATA.DETECT_URGENCY")

# %% 3.2 Create secrecy detection UDF
session.sql("""
CREATE OR REPLACE FUNCTION RAW_DATA.DETECT_SECRECY(text STRING)
RETURNS INT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
HANDLER = 'detect_secrecy'
AS $$
def detect_secrecy(text: str) -> int:
    '''Return secrecy score based on suspicious phrases (0-6).'''
    if text is None:
        return 0
    text_upper = text.upper()
    secrecy_phrases = [
        "DELETE", "DON'T TELL", "KEEP THIS BETWEEN", 
        "OFF THE RECORD", "CONFIDENTIAL", "SECRET"
    ]
    return sum(1 for phrase in secrecy_phrases if phrase in text_upper)
$$
""").collect()
print("✅ Created UDF: RAW_DATA.DETECT_SECRECY")

# %% [markdown]
# ## 4. Apply UDFs for Risk Scoring
# 
# Use our UDFs to score emails and compare against ground truth labels.

# %% 4.1 Score all emails
print("--- Top 10 Emails by Risk Score (Urgency + Secrecy) ---")

scored_emails = (
    emails
    .select(
        "EMAIL_ID",
        "SUBJECT",
        "COMPLIANCE_LABEL",
        call_udf("RAW_DATA.DETECT_URGENCY", col("BODY")).alias("URGENCY_SCORE"),
        call_udf("RAW_DATA.DETECT_SECRECY", col("BODY")).alias("SECRECY_SCORE"),
    )
    .with_column("RISK_SCORE", col("URGENCY_SCORE") + col("SECRECY_SCORE"))
    .filter(col("RISK_SCORE") > 0)
    .order_by(col("RISK_SCORE").desc())
    .limit(10)
)
scored_emails.show()

# %% 4.2 Compare risk scores by compliance label
print("--- Average Risk Score by Compliance Label ---")

risk_by_label = (
    emails
    .select(
        "COMPLIANCE_LABEL",
        call_udf("RAW_DATA.DETECT_URGENCY", col("BODY")).alias("URGENCY"),
        call_udf("RAW_DATA.DETECT_SECRECY", col("BODY")).alias("SECRECY"),
    )
    .group_by("COMPLIANCE_LABEL")
    .agg(
        avg("URGENCY").alias("AVG_URGENCY"),
        avg("SECRECY").alias("AVG_SECRECY"),
    )
    .with_column("AVG_TOTAL_RISK", col("AVG_URGENCY") + col("AVG_SECRECY"))
    .order_by(col("AVG_TOTAL_RISK").desc())
)
risk_by_label.show()

# %% [markdown]
# ## ✅ Part 1.1 Complete!
# 
# **What we learned:**
# - Snowpark DataFrames for SQL-like operations in Python
# - Loading data from stages
# - Creating permanent Python UDFs
# - Basic risk signal extraction from text
# 
# **Next:** Run `02_feature_store_setup.py` to build ML features from these patterns.
