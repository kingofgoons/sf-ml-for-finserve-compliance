# Snowflake Notebook / Python Worksheet
# Part 1.3: Model Registry & Custom Classification Models
#
# This script demonstrates:
# - Training a compliance classifier with Snowpark ML
# - Registering models in the Model Registry
# - Running inference with registered models
#
# Prerequisites:
# - Run 01 and 02 scripts first (data + features)
#
# Time: ~4-5 minutes

# %% [markdown]
# # Part 1.3: Model Registry & Custom Classification Models
# 
# **Goal:** Train and register an email compliance classifier.
# 
# The Snowflake Model Registry lets us:
# - Version and track models centrally
# - Deploy models for SQL-based inference
# - Maintain audit trails for compliance
# 
# We'll train a simple classifier to detect compliance violations.

# %% Setup
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col
from snowflake.ml.registry import Registry
from snowflake.ml.modeling.preprocessing import LabelEncoder
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.modeling.metrics import accuracy_score, precision_score, recall_score
import pandas as pd

session = get_active_session()

print(f"Connected as: {session.get_current_user()}")
print(f"Role: {session.get_current_role()}")

session.use_database("ML_COMPLIANCE_DEMO")
session.use_schema("ML")

# %% [markdown]
# ## 1. Prepare Training Data
# 
# Load the features we created in Part 1.2.

# %% Load email risk features
training_df = session.table("ML.EMAIL_RISK_FEATURES")

print("--- Training Data Preview ---")
training_df.show(5)
print(f"Total records: {training_df.count()}")

# Check label distribution
print("\n--- Label Distribution ---")
training_df.group_by("COMPLIANCE_LABEL").count().show()

# %% [markdown]
# ## 2. Feature Engineering for Classification
# 
# Convert the multi-class compliance labels to binary (VIOLATION vs CLEAN).

# %% Create binary labels
# For this demo: anything that's not CLEAN is a violation
training_df = training_df.with_column(
    "IS_VIOLATION",
    (col("COMPLIANCE_LABEL") != "CLEAN").cast("INT")
)

print("--- Binary Label Distribution ---")
training_df.group_by("IS_VIOLATION").count().show()

# %% Select features for training
feature_cols = [
    "URGENCY_SCORE",
    "SECRECY_SCORE", 
    "TOTAL_RISK_SCORE",
    "BODY_LENGTH",
    "IS_CROSS_DEPT",
    "IS_BARRIER_CROSSING",
]

label_col = "IS_VIOLATION"

# Keep only needed columns
model_df = training_df.select(
    "EMAIL_ID",
    *feature_cols,
    label_col,
    "COMPLIANCE_LABEL"  # Keep for evaluation
)

print("--- Model Features ---")
model_df.show(5)

# %% [markdown]
# ## 3. Train/Test Split
# 
# Split data for model evaluation.

# %% Random split
train_df, test_df = model_df.random_split([0.8, 0.2], seed=42)

print(f"Training set: {train_df.count()} records")
print(f"Test set: {test_df.count()} records")

# %% [markdown]
# ## 4. Train XGBoost Classifier
# 
# Use Snowpark ML's XGBClassifier — training runs inside Snowflake.

# %% Train model
print("🏋️ Training XGBoost classifier...")

xgb_model = XGBClassifier(
    input_cols=feature_cols,
    label_cols=[label_col],
    output_cols=["PREDICTED_VIOLATION"],
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
)

xgb_model.fit(train_df)
print("✅ Model training complete")

# %% [markdown]
# ## 5. Evaluate Model Performance

# %% Generate predictions on test set
predictions_df = xgb_model.predict(test_df)

print("--- Predictions Preview ---")
predictions_df.select(
    "EMAIL_ID", 
    "IS_VIOLATION", 
    "PREDICTED_VIOLATION",
    "COMPLIANCE_LABEL"
).show(10)

# %% Calculate metrics
# Convert to pandas for metric calculation
predictions_pd = predictions_df.select(
    "IS_VIOLATION", 
    "PREDICTED_VIOLATION"
).to_pandas()

y_true = predictions_pd["IS_VIOLATION"]
y_pred = predictions_pd["PREDICTED_VIOLATION"]

accuracy = (y_true == y_pred).mean()
# Calculate precision/recall for violation class
tp = ((y_true == 1) & (y_pred == 1)).sum()
fp = ((y_true == 0) & (y_pred == 1)).sum()
fn = ((y_true == 1) & (y_pred == 0)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print("\n--- Model Performance ---")
print(f"  Accuracy:  {accuracy:.2%}")
print(f"  Precision: {precision:.2%} (of flagged emails, how many are violations)")
print(f"  Recall:    {recall:.2%} (of actual violations, how many did we catch)")

# %% Confusion matrix
print("\n--- Confusion Matrix ---")
print(f"                    Predicted")
print(f"                 CLEAN   VIOLATION")
print(f"Actual CLEAN     {((y_true == 0) & (y_pred == 0)).sum():5d}   {fp:5d}")
print(f"Actual VIOLATION {fn:5d}   {tp:5d}")

# %% [markdown]
# ## 6. Register Model in Model Registry
# 
# Save the trained model to Snowflake's Model Registry for versioning and deployment.

# %% Initialize Model Registry
reg = Registry(session=session, database_name="ML_COMPLIANCE_DEMO", schema_name="ML")

print("✅ Model Registry initialized")

# %% Register the model
model_name = "EMAIL_COMPLIANCE_CLASSIFIER"

mv = reg.log_model(
    model=xgb_model,
    model_name=model_name,
    version_name="V1",
    metrics={
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "training_samples": train_df.count(),
        "test_samples": test_df.count(),
    },
    comment="XGBoost classifier for email compliance violation detection. Binary classification: CLEAN vs VIOLATION."
)

print(f"✅ Registered model: {model_name}/V1")

# %% [markdown]
# ## 7. Explore Model Registry

# %% List all registered models
print("--- Registered Models ---")
reg.show_models()

# %% Get model details
model_ref = reg.get_model(model_name)
print(f"\n--- Model: {model_name} ---")
print(f"Versions: {[v.version_name for v in model_ref.versions()]}")

# Get version details
v1 = model_ref.version("V1")
print(f"\nVersion V1 Metrics:")
for k, v in v1.get_metric("accuracy").items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 8. Run Inference with Registered Model
# 
# Use the registered model to score new emails.

# %% Load model from registry and run inference
loaded_model = reg.get_model(model_name).version("V1")

# Score all emails
all_emails_df = session.table("ML.EMAIL_RISK_FEATURES").select(
    "EMAIL_ID",
    *feature_cols,
)

scored_df = loaded_model.run(all_emails_df, function_name="predict")

print("--- Scored Emails ---")
scored_df.show(10)

# %% Save predictions to table
scored_with_labels = (
    scored_df
    .join(
        session.table("ML.EMAIL_RISK_FEATURES").select("EMAIL_ID", "COMPLIANCE_LABEL"),
        on="EMAIL_ID"
    )
)

scored_with_labels.write.mode("overwrite").save_as_table("ML.MODEL_PREDICTIONS_V1")
print("✅ Saved predictions to ML.MODEL_PREDICTIONS_V1")

# %% Compare predictions vs ground truth
print("\n--- Prediction Summary by Actual Label ---")
session.sql("""
    SELECT 
        COMPLIANCE_LABEL,
        COUNT(*) AS TOTAL,
        SUM(PREDICTED_VIOLATION) AS FLAGGED_AS_VIOLATION,
        ROUND(SUM(PREDICTED_VIOLATION) / COUNT(*) * 100, 1) AS FLAG_RATE_PCT
    FROM ML.MODEL_PREDICTIONS_V1
    GROUP BY COMPLIANCE_LABEL
    ORDER BY FLAG_RATE_PCT DESC
""").show()

# %% [markdown]
# ## 9. Custom Embedding Models from Hugging Face
# 
# Upload and use your own embedding model from Hugging Face.
# This is useful when you need domain-specific embeddings.

# %% Create stage for custom models
session.sql("""
CREATE STAGE IF NOT EXISTS ML.CUSTOM_MODELS
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    COMMENT = 'Stage for custom ML models (e.g., Hugging Face)'
""").collect()
print("✅ Created custom models stage")

# %% [markdown]
# ### Uploading a Custom Model
# 
# To use a custom Hugging Face model:
# 
# 1. **Download model locally:**
# ```python
# from transformers import AutoModel, AutoTokenizer
# model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# model.save_pretrained("./my_model")
# tokenizer.save_pretrained("./my_model")
# ```
# 
# 2. **Upload to Snowflake stage:**
# ```sql
# PUT file://./my_model/* @ML.CUSTOM_MODELS/sentence_transformer/ AUTO_COMPRESS=FALSE;
# ```
# 
# 3. **Create UDF using the model:**

# %% Create custom embedding UDF (template)
print("--- Custom Embedding UDF Template ---")

CUSTOM_EMBED_UDF = """
-- This UDF loads a custom model from stage and generates embeddings
-- Requires: Model files uploaded to @ML.CUSTOM_MODELS/sentence_transformer/

CREATE OR REPLACE FUNCTION ML.CUSTOM_EMBED(text STRING)
RETURNS ARRAY
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('transformers', 'torch', 'snowflake-snowpark-python')
IMPORTS = ('@ML.CUSTOM_MODELS/sentence_transformer/')
HANDLER = 'embed_text'
AS $$
import sys
import os
import torch
from transformers import AutoModel, AutoTokenizer

# Load model from stage (cached after first load)
MODEL_PATH = sys._xoptions.get("snowflake_import_directory") + "/sentence_transformer"
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModel.from_pretrained(MODEL_PATH)
        model.eval()

def embed_text(text: str) -> list:
    load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings
$$;
"""

print(CUSTOM_EMBED_UDF)
print("\n⚠️ Note: Upload model files to stage before creating this UDF")

# %% Show stage contents (for verification)
print("\n--- Custom Models Stage Contents ---")
session.sql("LIST @ML.CUSTOM_MODELS").show()

# %% [markdown]
# ### When to Use Custom Models
# 
# | Use Case | Recommendation |
# |----------|----------------|
# | General text | Use Snowflake's `AI_EMBED` (optimized, no setup) |
# | Domain-specific | Custom model trained on your domain |
# | Regulatory requirement | Specific approved model |
# | Reproducibility | Pinned model version |

# %% [markdown]
# ## ✅ Part 1.3 Complete!
# 
# **What we learned:**
# - Training ML models with Snowpark ML (XGBClassifier)
# - Evaluating binary classification performance
# - Registering models in the Model Registry
# - Running inference with registered models
# - **Uploading custom Hugging Face models to stages**
# - **Creating UDFs with custom embedding models**
# 
# **Model registered:**
# - `EMAIL_COMPLIANCE_CLASSIFIER/V1`
# - Binary classification: CLEAN vs VIOLATION
# - Features: urgency, secrecy, risk scores, cross-dept, barrier-crossing
# 
# **Next:** Run `04_model_benchmarking.py` to compare this ML approach vs keyword-based rules.

