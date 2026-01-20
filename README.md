# ML for Financial Services Compliance

## Email Surveillance Intelligence Demo

A hands-on demonstration of Snowflake's ML capabilities for hedge fund compliance teams. This demo shows how to build an intelligent email surveillance system that combines semantic embeddings, ML classification, LLMs, fine-tuning, and vector search.

---

## Performance Results

| Metric | Keyword Baseline | ML Only | Hybrid (ML + LLM) |
|--------|-----------------|---------|-------------------|
| **Precision** | 32% | 93% | **93%** |
| **Recall** | 16% | 78% | **82%** |
| **F1 Score** | 21% | 85% | **87%** |

*The hybrid system improves recall by 4% by having the LLM analyze uncertain cases, while only processing 2.3% of emails through the LLM.*

---

## Quick Start

### Prerequisites

- Snowflake account with Cortex enabled
- Python 3.9+ with Snowpark
- Role with CREATE DATABASE/WAREHOUSE privileges

### Setup

1. **Configure Snowflake connection:**
   ```bash
   # Your ~/.snowflake/config.toml will be used automatically
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

5. **Run the demo notebooks in order** (DEMO_00 through DEMO_08)

---

## Repository Structure

```
sf-ml-for-finserve-compliance/
├── README.md                       
├── requirements.txt                
│
├── scripts/
│   ├── generate_data.py            # Generate 10K emails + 500 fine-tune samples
│   └── setup_snowflake.py          # Push data to Snowflake
│
├── notebooks/
│   ├── DEMO_00_the_pain.ipynb      # Keyword baseline (the problem)
│   ├── DEMO_01_blueprint.ipynb     # Architecture overview
│   ├── DEMO_02_layer1_features.ipynb   # Semantic feature engineering
│   ├── DEMO_03_layer2_ml_model.ipynb   # XGBoost + Model Registry
│   ├── DEMO_04_layer3_llm_analysis.ipynb   # Claude deep analysis
│   ├── DEMO_05_layer4_hybrid.ipynb     # ML + LLM pipeline
│   ├── DEMO_06_layer5_finetuning.ipynb # Fine-tune for compliance
│   ├── DEMO_07_layer6_semantic_search.ipynb # Cortex Search
│   └── DEMO_08_resolution.ipynb    # Summary & results
│
├── data/
│   ├── emails_synthetic.csv        # 10K synthetic emails
│   └── finetune_training.jsonl     # 500 labeled training samples
│
└── src_archive/                    # Original reference files
```

---

## Demo Flow

### The Pain (DEMO_00)
Shows why keyword-based compliance fails:
- 32% precision (68% false alarms)
- 16% recall (misses 84% of violations)
- Compliance teams drowning in noise

### The Solution (DEMO_01-08)

| Demo | Layer | Focus | Snowflake Features |
|------|-------|-------|--------------------|
| 01 | - | Architecture blueprint | Conceptual overview |
| 02 | 1 | Semantic features | Cortex Embeddings, VECTOR type |
| 03 | 2 | ML classification | Feature Store, Model Registry, XGBoost |
| 04 | 3 | LLM deep analysis | CORTEX.COMPLETE (Claude) |
| 05 | 4 | Hybrid pipeline | ML filter + LLM reasoning |
| 06 | 5 | Fine-tuning | CORTEX.FINETUNE |
| 07 | 6 | Semantic search | Cortex Search Service |
| 08 | - | Resolution | Full system summary |

---

## The Dataset

**10,000 synthetic hedge fund emails** with realistic noise:

| Label | Distribution | Description |
|-------|-------------|-------------|
| CLEAN | ~67% | Normal business communications |
| INSIDER_TRADING | ~8% | MNPI sharing, trading tips |
| CONFIDENTIALITY_BREACH | ~9% | Client data leaks |
| PERSONAL_TRADING | ~8% | Undisclosed personal trades |
| INFO_BARRIER_VIOLATION | ~8% | Research/Trading wall breaches |

**Data includes:**
- Subtle violations (not obvious language)
- Borderline clean emails (discuss sensitive topics legitimately)
- ~8% label noise (simulates real-world labeling disagreements)

---

## Key Innovation: Semantic Risk Features

Instead of keyword matching, we measure **semantic distance** from risk concepts:

```python
BASELINE_CONCEPT = "quarterly report meeting schedule project update..."

RISK_CONCEPTS = {
    'SECRECY': "keep this secret between us, do not tell anyone...",
    'URGENCY': "act before the announcement, move now before news...",
    'INSIDER': "inside information about merger, non-public material...",
    'EVASION': "delete this email, destroy evidence, cover our tracks...",
    'TIPPING': "buy this stock now, guaranteed profit, act on this tip..."
}

# Relative risk score = risk_similarity - baseline_similarity
# Negative = normal business email
# Positive = elevated risk
```

This captures **meaning, not keywords** - violators can change vocabulary but meaning still clusters near risk concepts.

---

## Snowflake Technologies Used

| Technology | Purpose |
|------------|---------|
| **Cortex Embeddings** | EMBED_TEXT_768 with Arctic Embed M |
| **VECTOR type** | Native 768-dim vector storage |
| **VECTOR_COSINE_SIMILARITY** | Semantic similarity calculation |
| **Feature Store** | Versioned, auto-refreshing features |
| **Model Registry** | Versioned ML model deployment |
| **XGBClassifier** | Gradient boosting classification |
| **CORTEX.COMPLETE** | LLM inference (Claude) |
| **CORTEX.FINETUNE** | Custom model training |
| **Cortex Search Service** | Self-maintaining semantic search |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     10,000 emails/day                            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1: SEMANTIC FEATURES                                      │
│  └── Embed emails with Arctic Embed M                            │
│  └── Compute relative risk scores vs baseline                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 2: ML CLASSIFIER (XGBoost)                                │
│  └── 93% precision, 80% recall, 86% F1                           │
│  └── Fast (~5ms), cheap (~$0.0001/email)                         │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
         ~55% AUTO-CLEARED             ~45% FLAGGED
         (low risk)                         │
                                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 3: LLM ANALYSIS (Claude)                                  │
│  └── Deep reasoning with natural language explanation            │
│  └── Categorizes violation type with confidence                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
         LOW RISK                     HIGH RISK
       (auto-close)                  (human review)
```

**Result:** ~55% cost reduction vs all-LLM while maintaining accuracy.

---

## Cleanup

After the demo, remove all objects:

```sql
DROP DATABASE IF EXISTS COMPLIANCE_DEMO CASCADE;
DROP WAREHOUSE IF EXISTS COMPLIANCE_DEMO_WH;
```

---

## Notes

- **Semantic features:** Computing embeddings for 10K emails takes ~30 seconds
- **Fine-tuning:** The FINETUNE job is async and takes 30-60 minutes
- **Cortex Search:** Service indexing may take a few minutes after creation
- **Label noise:** 8% of labels are intentionally flipped to create realistic ML performance

---

## License

Internal demo - not for distribution.
