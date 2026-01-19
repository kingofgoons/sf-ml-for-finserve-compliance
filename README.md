# Hedge Fund Email Compliance Intelligence Platform

An end-to-end demonstration of Snowflake ML capabilities for email surveillance and compliance in financial services.

---

## Overview

This hands-on lab walks through building an intelligent email compliance system using Snowflake's ML infrastructure. The demo covers:

- **Email data processing** with Snowpark
- **Feature engineering** for communication risk signals
- **Model management** via Feature Store & Model Registry
- **LLM integration** for automated compliance scoring
- **Vector search** for semantic pattern detection

**Total demo time:** ~35-40 minutes of code execution + discussion

---

## Prerequisites

- Snowflake account with:
  - Snowpark-enabled warehouse
  - Cortex LLM functions enabled
  - `ACCOUNTADMIN` or appropriate ML privileges
- Python 3.9+

---

## Project Structure

```
├── src/
│   ├── 00_setup.sql                         # Snowflake object setup
│   ├── 01_snowpark_email_processing.py      # Part 1.1
│   ├── 02_feature_store_setup.py            # Part 1.2
│   ├── 03_model_registry.py                 # Part 1.3
│   ├── 04_model_benchmarking.py             # Part 2.1
│   ├── 05_llm_compliance_pipeline.py        # Part 2.2
│   ├── 06_fine_tuning.py                    # Part 2.3
│   ├── 07_vector_search.py                  # Part 3.1
│   ├── 08_pattern_evolution.py              # Part 3.2
│   └── 99_reset.sql                         # Cleanup for re-runs
├── data/
│   └── emails_synthetic.csv                 # Generated mock email data
├── scripts/
│   └── generate_data.py                     # Synthetic data generator
└── assets/
    └── (diagrams, screenshots)
```

---

## Part 1: ML Infrastructure & Email Compliance Workflows

> **Time:** 12-15 minutes

### Step 1.1 — Snowpark Basics for Email Data Processing

**Goal:** Load and process email archive data using Snowpark DataFrames.

- Connect to Snowflake, load synthetic email data
- DataFrame operations: sender/recipient analysis, temporal patterns
- Create processing UDFs for message metadata extraction

**Key Snowflake features:** Snowpark Python, UDFs

### Step 1.2 — Feature Store for Communication Risk Features

**Goal:** Build reusable features for email surveillance models.

- Define communication risk features (message frequency, after-hours activity, cross-department communication)
- Create Feature Store entities and feature views
- Generate training datasets from feature pipelines

**Key Snowflake features:** Feature Store, Feature Views

### Step 1.3 — Model Registry & Custom Classification Models

**Goal:** Register and version email classification models.

- Train a simple compliance classifier (scikit-learn or XGBoost)
- Register model in Snowflake Model Registry
- Deploy model as a UDF for inference

**Key Snowflake features:** Model Registry, Custom UDFs

---

## Part 2: Model Performance & Email Intelligence Integration

> **Time:** 12-15 minutes

### Step 2.1 — Email Surveillance Model Benchmarking

**Goal:** Compare keyword-based vs ML-based detection approaches.

- Run both approaches on test dataset
- Compare precision, recall, false positive rates
- Visualize investigation efficiency gains

**Key Snowflake features:** Cortex Playground, model evaluation

### Step 2.2 — LLM-to-Email Compliance Integration

**Goal:** Chain LLM analysis into structured compliance workflows.

- Use `AI_CLASSIFY` for email categorization
- Use `AI_COMPLETE` for extracting structured compliance signals
- Build end-to-end scoring pipeline

**Key Snowflake features:** AI_CLASSIFY, AI_COMPLETE, Cortex LLM functions

### Step 2.3 — Fine-Tuning for Financial Communication Language

**Goal:** Customize models for hedge fund terminology.

- Prepare fine-tuning dataset with fund-specific examples
- Run `CORTEX.FINETUNE` job
- Evaluate fine-tuned model vs base model

**Key Snowflake features:** Cortex Fine-tuning

---

## Part 3: Vector Search for Email Intelligence

> **Time:** 8-10 minutes

### Step 3.1 — Email Content Search & Pattern Recognition

**Goal:** Enable semantic search across email archives.

- Generate embeddings with `AI_EMBED`
- Create Cortex Search service
- Demo: "Find emails similar to this compliance violation"

**Key Snowflake features:** AI_EMBED, Cortex Search

### Step 3.2 — Communication Pattern Evolution & Model Updates

**Goal:** Maintain models as communication patterns change.

- Batch re-embedding workflow
- A/B testing between model versions
- Performance validation framework

**Key Snowflake features:** Cortex Search multi-indexing

---

## Compliance Use Cases Demonstrated

| Use Case | What We Detect |
|----------|----------------|
| **Insider Trading Prevention** | MNPI sharing, trading discussions before blackouts |
| **Information Barriers** | Cross-departmental leaks, conflict of interest |
| **Personal Trading Compliance** | Personal investment discussions, pre-clearance violations |
| **Client Confidentiality** | Unauthorized client info sharing, data leakage |

---

## Quick Start

```bash
# 1. Set up Snowflake objects
snowsql -f src/00_setup.sql

# 2. Generate synthetic data (if not present)
python scripts/generate_data.py

# 3. Walk through each step in order
python src/01_snowpark_email_processing.py
# ... continue through 08

# 4. Reset environment for next demo
snowsql -f src/99_reset.sql
```

---

## Synthetic Data Schema

The demo uses generated email data with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `email_id` | STRING | Unique identifier |
| `sender` | STRING | Sender email address |
| `recipient` | STRING | Recipient email address |
| `cc` | STRING | CC recipients (nullable) |
| `subject` | STRING | Email subject line |
| `body` | STRING | Email body text |
| `sent_at` | TIMESTAMP | Send timestamp |
| `sender_dept` | STRING | Sender's department |
| `recipient_dept` | STRING | Recipient's department |
| `compliance_label` | STRING | Ground truth label (for training) |

**Labels:** `CLEAN`, `INSIDER_TRADING`, `CONFIDENTIALITY_BREACH`, `PERSONAL_TRADING`, `INFO_BARRIER_VIOLATION`

---

## References

- [Snowpark Python UDFs](https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-udfs)
- [Feature Store Overview](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview)
- [Model Registry](https://docs.snowflake.com/developer-guide/snowflake-ml/overview)
- [AI_CLASSIFY](https://docs.snowflake.com/en/sql-reference/functions/ai_classify)
- [AI_EMBED](https://docs.snowflake.com/en/sql-reference/functions/ai_embed)
- [Cortex Fine-tuning](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-finetuning)
- [Cortex Search](https://docs.snowflake.com/en/sql-reference/sql/create-cortex-search)

---

## License

Internal demo use only. All data is synthetic.
