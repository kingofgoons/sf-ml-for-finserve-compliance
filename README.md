# ML for Financial Services Compliance

## Email Surveillance Intelligence Demo

A hands-on demonstration of Snowflake's ML capabilities for hedge fund compliance teams. This demo shows how to build an intelligent email surveillance system that combines traditional ML, LLMs, fine-tuning, and vector search.

---

## Quick Start

### Prerequisites

- Snowflake account with Cortex enabled
- Python 3.9+ with Snowpark
- Role with CREATE DATABASE/WAREHOUSE privileges

### Setup

1. **Configure Snowflake connection:**
   ```bash
   # Option A: Use your existing config
   # Your ~/.snowflake/config.toml will be used automatically
   
   # Option B: Create a local config for this demo
   cp snowflake.config.template snowflake.config
   # Edit snowflake.config with your credentials
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate demo data:**
   ```bash
   python scripts/generate_data.py
   ```

4. **Push data to Snowflake:**
   ```bash
   python scripts/setup_snowflake.py
   ```

5. **Run the notebooks in order** (01 through 06)

---

## Repository Structure

```
sf-ml-for-finserve-compliance/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── snowflake.config.template       # Config template (safe to commit)
├── snowflake.config                # Your config (gitignored)
│
├── scripts/
│   ├── generate_data.py            # Generate 10K emails + 500 fine-tune samples
│   └── setup_snowflake.py          # Push data to Snowflake
│
├── notebooks/
│   ├── 00_setup.sql                # Create DB, schema, warehouse
│   ├── 01_data_exploration.ipynb   # Explore email dataset
│   ├── 02_feature_engineering.ipynb # Feature Store setup
│   ├── 03_model_training.ipynb     # XGBoost + Model Registry
│   ├── 04_ml_llm_integration.ipynb # ML + Claude claude-opus-4-5 architecture
│   ├── 05_fine_tuning.ipynb        # Fine-tune LLM for compliance
│   ├── 06_vector_search.ipynb      # Arctic Embed + semantic search
│   └── 99_cleanup.sql              # Remove demo objects
│
├── data/
│   ├── emails_synthetic.csv        # 10K synthetic emails
│   └── finetune_training.jsonl     # 500 labeled training samples
│
└── src_archive/                    # Original files (reference only)
```

---

## Demo Flow

### Part 1: The ML Foundation

| Notebook | Focus | Snowflake Features |
|----------|-------|--------------------|
| 01_data_exploration | Understand the data | Basic SQL, aggregations |
| 02_feature_engineering | Build reusable features | Feature Store, Entity, FeatureView |
| 03_model_training | Train & deploy models | XGBClassifier, Model Registry |

### Part 2: The Integration

| Notebook | Focus | Snowflake Features |
|----------|-------|--------------------|
| 04_ml_llm_integration | Optimal architecture | CORTEX.COMPLETE (Claude claude-opus-4-5), tiered approach |

### Part 3: Advanced Capabilities

| Notebook | Focus | Snowflake Features |
|----------|-------|--------------------|
| 05_fine_tuning | Custom LLM | CORTEX.FINETUNE |
| 06_vector_search | Semantic search | EMBED_TEXT_768 (Arctic), VECTOR_COSINE_SIMILARITY |

---

## The Dataset

**10,000 synthetic hedge fund emails** with compliance labels:

| Label | Count | Description |
|-------|-------|-------------|
| CLEAN | ~70% | Normal business communications |
| INSIDER_TRADING | ~8% | MNPI sharing, trading tips |
| CONFIDENTIALITY_BREACH | ~8% | Client data leaks |
| PERSONAL_TRADING | ~7% | Undisclosed personal trades |
| INFO_BARRIER_VIOLATION | ~7% | Research↔Trading wall breaches |

---

## Key Snowflake Features Demonstrated

### Snowpark ML
- **Feature Store:** Reusable, versioned feature engineering
- **Model Registry:** Version-controlled model deployment
- **XGBClassifier:** Native gradient boosting

### Cortex AI
- **COMPLETE:** LLM inference with Claude claude-opus-4-5
- **FINETUNE:** Custom model training
- **EMBED_TEXT_768:** Vector embeddings with Arctic Embed

### Vector Search
- **VECTOR type:** Native 768-dim vector storage
- **VECTOR_COSINE_SIMILARITY:** Semantic similarity search

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     10,000 emails/day                            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  TIER 1: ML CLASSIFIER (XGBoost)                                 │
│  └── Fast (~5ms), cheap (~$0.0001/email)                         │
│  └── Catches obvious violations                                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
         ~85% CLEAN                    ~15% FLAGGED
         (no action)                        │
                                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  TIER 2: LLM ANALYSIS (Claude claude-opus-4-5)                            │
│  └── Smart (~200ms), more expensive (~$0.01/email)               │
│  └── Deep reasoning, nuanced detection                           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
         LOW RISK                     HIGH RISK
       (auto-close)                  (human review)
```

**Result:** 90% cost reduction vs all-LLM while maintaining accuracy.

---

## Cleanup

After the demo, remove all objects:

```sql
-- Run notebooks/99_cleanup.sql
-- Or manually:
DROP DATABASE IF EXISTS COMPLIANCE_DEMO CASCADE;
DROP WAREHOUSE IF EXISTS COMPLIANCE_DEMO_WH;
```

---

## Notes

- **Fine-tuning:** The FINETUNE job in notebook 05 is async and may take 30+ minutes
- **Embeddings:** Generating 10K embeddings in notebook 06 takes a few minutes
- **Costs:** Demo uses MEDIUM warehouse; adjust as needed for your account

---

## License

Internal demo - not for distribution.
