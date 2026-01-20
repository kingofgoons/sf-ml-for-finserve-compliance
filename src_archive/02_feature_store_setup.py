# Snowflake Notebook / Python Worksheet
# Part 1.2: Feature Store for Communication Risk Features
#
# This script demonstrates:
# - Creating Feature Store entities
# - Defining feature views from email data
# - Generating training datasets
#
# Prerequisites:
# - Run 00_setup.sql and 01_snowpark_email_processing.py first
#
# Time: ~4-5 minutes

# %% [markdown]
# # Part 1.2: Feature Store for Communication Risk Features
# 
# **Goal:** Build reusable ML features for email surveillance models.
# 
# The Snowflake Feature Store lets us:
# - Define features once, reuse across models
# - Track feature lineage and versioning
# - Generate point-in-time correct training data
# 
# We'll create communication pattern features that signal compliance risk.

# %% Setup
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, count, avg, sum as sum_, when, hour, length, lit, datediff
from snowflake.ml.feature_store import FeatureStore, Entity, FeatureView

session = get_active_session()

print(f"Connected as: {session.get_current_user()}")
print(f"Role: {session.get_current_role()}")

# Use the ML schema for feature store objects
session.use_database("ML_COMPLIANCE_DEMO")
session.use_schema("ML")

# %% [markdown]
# ## 1. Initialize Feature Store
# 
# The Feature Store uses a schema to store metadata about entities and features.

# %% Create Feature Store
fs = FeatureStore(
    session=session,
    database="ML_COMPLIANCE_DEMO",
    name="ML",  # Schema name
    default_warehouse="ML_COMPLIANCE_WH",
)
print("✅ Feature Store initialized")

# %% [markdown]
# ## 2. Define Entities
# 
# Entities are the "join keys" for features — typically business objects like users, accounts, or in our case, employees.

# %% Create Employee entity
employee_entity = Entity(
    name="EMPLOYEE",
    join_keys=["EMPLOYEE_EMAIL"],
    desc="Employee identified by email address"
)

fs.register_entity(employee_entity)
print("✅ Registered entity: EMPLOYEE")

# List all entities
fs.list_entities().show()

# %% [markdown]
# ## 3. Create Feature Views
# 
# Feature Views define how features are computed from source data.
# We'll create communication pattern features for each employee.

# %% 3.1 Build the feature DataFrame
# First, compute features from the raw emails table

emails = session.table("RAW_DATA.EMAILS")

# Compute per-employee features
employee_features_df = (
    emails
    # Aggregate by sender
    .group_by(col("SENDER").alias("EMPLOYEE_EMAIL"))
    .agg(
        # Volume features
        count("*").alias("TOTAL_EMAILS_SENT"),
        avg(length("BODY")).alias("AVG_EMAIL_LENGTH"),
        
        # Cross-department ratio
        sum_(
            when(col("SENDER_DEPT") != col("RECIPIENT_DEPT"), lit(1)).otherwise(lit(0))
        ).alias("CROSS_DEPT_COUNT"),
        
        # After-hours activity
        sum_(
            when(
                (hour(col("SENT_AT").cast("TIMESTAMP")) < 8) | 
                (hour(col("SENT_AT").cast("TIMESTAMP")) >= 18), 
                lit(1)
            ).otherwise(lit(0))
        ).alias("AFTER_HOURS_COUNT"),
        
        # Research <-> Trading barrier contacts (high risk signal)
        sum_(
            when(
                ((col("SENDER_DEPT") == "Research") & (col("RECIPIENT_DEPT") == "Trading")) |
                ((col("SENDER_DEPT") == "Trading") & (col("RECIPIENT_DEPT") == "Research")),
                lit(1)
            ).otherwise(lit(0))
        ).alias("BARRIER_VIOLATION_COUNT"),
    )
    # Compute ratios
    .with_column(
        "CROSS_DEPT_RATIO",
        col("CROSS_DEPT_COUNT") / col("TOTAL_EMAILS_SENT")
    )
    .with_column(
        "AFTER_HOURS_RATIO",
        col("AFTER_HOURS_COUNT") / col("TOTAL_EMAILS_SENT")
    )
)

print("--- Employee Feature Preview ---")
employee_features_df.show(10)

# %% 3.2 Save features to a table (Feature View source)
employee_features_df.write.mode("overwrite").save_as_table("ML.EMPLOYEE_COMM_FEATURES")
print("✅ Saved features to ML.EMPLOYEE_COMM_FEATURES")

# %% 3.3 Create Feature View
# Register the feature view with the Feature Store

employee_comm_fv = FeatureView(
    name="EMPLOYEE_COMMUNICATION_FEATURES",
    entities=[employee_entity],
    feature_df=session.table("ML.EMPLOYEE_COMM_FEATURES"),
    desc="Communication pattern features for compliance risk scoring"
)

# Register with Feature Store
employee_comm_fv = fs.register_feature_view(
    feature_view=employee_comm_fv,
    version="V1",
    block=True,  # Wait for registration to complete
)
print("✅ Registered FeatureView: EMPLOYEE_COMMUNICATION_FEATURES/V1")

# %% List registered feature views
print("--- Registered Feature Views ---")
fs.list_feature_views().show()

# %% [markdown]
# ## 4. Add Email-Level Features
# 
# Create a second feature view for per-email risk signals (from our UDFs).

# %% 4.1 Create Email entity
email_entity = Entity(
    name="EMAIL",
    join_keys=["EMAIL_ID"],
    desc="Individual email message"
)

fs.register_entity(email_entity)
print("✅ Registered entity: EMAIL")

# %% 4.2 Compute email-level features using our UDFs
email_features_df = (
    emails
    .select(
        col("EMAIL_ID"),
        col("COMPLIANCE_LABEL"),
        # Use UDFs from Part 1.1
        session.sql("SELECT RAW_DATA.DETECT_URGENCY(BODY) FROM RAW_DATA.EMAILS LIMIT 1").collect(),  # Verify UDF exists
    )
)

# Use SQL to compute features with UDFs (cleaner approach)
email_features_df = session.sql("""
    SELECT 
        EMAIL_ID,
        COMPLIANCE_LABEL,
        RAW_DATA.DETECT_URGENCY(BODY) AS URGENCY_SCORE,
        RAW_DATA.DETECT_SECRECY(BODY) AS SECRECY_SCORE,
        RAW_DATA.DETECT_URGENCY(BODY) + RAW_DATA.DETECT_SECRECY(BODY) AS TOTAL_RISK_SCORE,
        LENGTH(BODY) AS BODY_LENGTH,
        CASE 
            WHEN SENDER_DEPT != RECIPIENT_DEPT THEN 1 
            ELSE 0 
        END AS IS_CROSS_DEPT,
        CASE 
            WHEN (SENDER_DEPT = 'Research' AND RECIPIENT_DEPT = 'Trading')
              OR (SENDER_DEPT = 'Trading' AND RECIPIENT_DEPT = 'Research')
            THEN 1 
            ELSE 0 
        END AS IS_BARRIER_CROSSING
    FROM RAW_DATA.EMAILS
""")

print("--- Email Feature Preview ---")
email_features_df.show(10)

# %% 4.3 Save and register email features
email_features_df.write.mode("overwrite").save_as_table("ML.EMAIL_RISK_FEATURES")
print("✅ Saved features to ML.EMAIL_RISK_FEATURES")

email_risk_fv = FeatureView(
    name="EMAIL_RISK_FEATURES",
    entities=[email_entity],
    feature_df=session.table("ML.EMAIL_RISK_FEATURES"),
    desc="Per-email risk signals from text analysis"
)

email_risk_fv = fs.register_feature_view(
    feature_view=email_risk_fv,
    version="V1",
    block=True,
)
print("✅ Registered FeatureView: EMAIL_RISK_FEATURES/V1")

# %% [markdown]
# ## 5. Generate Training Dataset
# 
# Use the Feature Store to create a point-in-time correct training dataset.

# %% 5.1 Create a spine (list of entities to get features for)
# The spine defines which entities and timestamps we want features for

training_spine = session.sql("""
    SELECT 
        EMAIL_ID,
        COMPLIANCE_LABEL AS LABEL
    FROM RAW_DATA.EMAILS
""")

print(f"Training spine has {training_spine.count()} records")

# %% 5.2 Generate training dataset with features
training_dataset = fs.generate_dataset(
    name="EMAIL_COMPLIANCE_TRAINING",
    spine_df=training_spine,
    features=[email_risk_fv],
    output_type="table",
    desc="Training dataset for email compliance classification"
)

print("✅ Generated training dataset")
print(f"Dataset: {training_dataset.fully_qualified_name()}")

# Preview the dataset
training_dataset.read.df().show(10)

# %% [markdown]
# ## 6. Explore Feature Store Metadata
# 
# The Feature Store tracks lineage and metadata for all features.

# %% List all feature views with details
print("--- All Feature Views ---")
for fv in fs.list_feature_views().collect():
    print(f"  • {fv['NAME']}/{fv['VERSION']}: {fv['DESC']}")

# %% Get feature view details
fv_details = fs.get_feature_view("EMAIL_RISK_FEATURES", "V1")
print(f"\n--- Feature View Details: {fv_details.name} ---")
print(f"Version: {fv_details.version}")
print(f"Entities: {[e.name for e in fv_details.entities]}")
print(f"Description: {fv_details.desc}")

# %% [markdown]
# ## ✅ Part 1.2 Complete!
# 
# **What we learned:**
# - Creating Feature Store entities (EMPLOYEE, EMAIL)
# - Building feature DataFrames from raw data
# - Registering Feature Views for reuse
# - Generating training datasets
# 
# **Features created:**
# | Feature | Description |
# |---------|-------------|
# | `TOTAL_EMAILS_SENT` | Email volume per employee |
# | `CROSS_DEPT_RATIO` | % of cross-department emails |
# | `AFTER_HOURS_RATIO` | % of emails sent outside 8am-6pm |
# | `BARRIER_VIOLATION_COUNT` | Research↔Trading contacts |
# | `URGENCY_SCORE` | Keyword-based urgency signal |
# | `SECRECY_SCORE` | Suspicious phrase detection |
# 
# **Next:** Run `03_model_registry.py` to train and register a compliance classifier.

