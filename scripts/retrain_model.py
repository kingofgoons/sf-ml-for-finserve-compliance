#!/usr/bin/env python3
"""Retrain ML model on new LLM-generated email data."""

from snowflake.snowpark import Session
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.registry import Registry

session = Session.builder.getOrCreate()
session.sql("USE WAREHOUSE COMPLIANCE_DEMO_WH").collect()
session.sql("USE DATABASE COMPLIANCE_DEMO").collect()
session.sql("USE SCHEMA ML").collect()

print("Training XGBoost model on new data...")

FEATURE_COLS = [
    "BASELINE_SIMILARITY",
    "MNPI_RISK_SCORE",
    "CONFIDENTIALITY_RISK_SCORE",
    "PERSONAL_TRADING_RISK_SCORE",
    "INFO_BARRIER_RISK_SCORE",
    "CROSS_BARRIER_FLAG",
]
TARGET_COL = "IS_VIOLATION"

features_df = session.table("EMAIL_SEMANTIC_FEATURES")

train_df, test_df = features_df.random_split([0.8, 0.2], seed=42)

print(f"Training samples: {train_df.count()}")
print(f"Test samples: {test_df.count()}")

model = XGBClassifier(
    input_cols=FEATURE_COLS,
    label_cols=[TARGET_COL],
    output_cols=["PREDICTION"],
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
)

model.fit(train_df)
print("Model trained!")

predictions = model.predict_proba(test_df)
predictions.write.mode("overwrite").save_as_table("MODEL_TEST_PREDICTIONS")

results = session.sql("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN PREDICT_PROBA_1 >= 0.5 AND IS_VIOLATION = 1 THEN 1 ELSE 0 END) as tp,
        SUM(CASE WHEN PREDICT_PROBA_1 >= 0.5 AND IS_VIOLATION = 0 THEN 1 ELSE 0 END) as fp,
        SUM(CASE WHEN PREDICT_PROBA_1 < 0.5 AND IS_VIOLATION = 1 THEN 1 ELSE 0 END) as fn,
        SUM(CASE WHEN PREDICT_PROBA_1 < 0.5 AND IS_VIOLATION = 0 THEN 1 ELSE 0 END) as tn
    FROM MODEL_TEST_PREDICTIONS
""").collect()[0]

tp, fp, fn, tn = results["TP"], results["FP"], results["FN"], results["TN"]
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Test metrics: Precision={precision:.2%}, Recall={recall:.2%}, F1={f1:.2%}")

reg = Registry(session=session)
model_version = reg.log_model(
    model,
    model_name="EMAIL_COMPLIANCE_CLASSIFIER",
    version_name="v3_llm_data",
    sample_input_data=train_df.select(FEATURE_COLS),
    comment="XGBoost classifier trained on LLM-generated emails"
)
print("Model registered as EMAIL_COMPLIANCE_CLASSIFIER v3_llm_data")

print("Generating full predictions with three-way classification...")

all_predictions = model.predict_proba(features_df)
all_predictions.write.mode("overwrite").save_as_table("MODEL_PREDICTIONS_RAW")

session.sql("""
    CREATE OR REPLACE TABLE MODEL_PREDICTIONS_V1 AS
    SELECT 
        *,
        PREDICT_PROBA_1 as VIOLATION_PROBABILITY,
        CASE 
            WHEN PREDICT_PROBA_1 >= 0.7 THEN 'HIGH_RISK'
            WHEN PREDICT_PROBA_1 <= 0.3 THEN 'LOW_RISK'
            ELSE 'NEEDS_REVIEW'
        END as ML_DECISION
    FROM MODEL_PREDICTIONS_RAW
""").collect()

dist = session.sql("""
    SELECT 
        ML_DECISION,
        COUNT(*) as cnt,
        SUM(IS_VIOLATION) as actual_violations
    FROM MODEL_PREDICTIONS_V1
    GROUP BY ML_DECISION
    ORDER BY cnt DESC
""").collect()

print("Three-way classification distribution:")
for r in dist:
    print(f"  {r['ML_DECISION']}: {r['CNT']} emails, {r['ACTUAL_VIOLATIONS']} actual violations")

print("Done!")
