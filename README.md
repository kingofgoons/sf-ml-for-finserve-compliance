# ML for Financial Services Compliance

## Email Surveillance Intelligence Demo

A hands-on demonstration of Snowflake's ML capabilities for hedge fund compliance teams. This demo shows how to build an intelligent email surveillance system that combines semantic embeddings, ML classification, LLMs, fine-tuning, and vector search.

---

## Performance Results

| Metric | Keyword Baseline | ML Only | Hybrid (ML + LLM) |
|--------|------------------|---------|-------------------|
| **Precision** | ~38% | ~91% | **~91%** |
| **Recall** | ~67% | ~54% | **~85%** |
| **F1 Score** | ~49% | ~68% | **~88%** |

*The key insight: ML handles clear-cut cases (HIGH_RISK and LOW_RISK) while the LLM focuses on the uncertain NEEDS_REVIEW bucket where it adds the most value. This targeted approach improves recall by ~31% while only processing 21.2% of emails through the LLM.*

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

3. **Generate demo data (LLM-powered):**
   ```bash
   python scripts/generate_data_llm.py
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
├── TALK_TRACK.md               # Demo presentation notes
├── requirements.txt                
├── snowflake.config.template   # Config template
│
├── scripts/
│   ├── generate_data_llm.py    # Generate 10K emails via Cortex LLM
│   ├── setup_snowflake.py      # Create DB/schemas and load data
│   └── retrain_model.py        # Retrain ML model if needed
│
├── notebooks/
│   ├── DEMO_00_the_pain.ipynb          # Keyword baseline (the problem)
│   ├── DEMO_01_blueprint.ipynb         # Architecture overview
│   ├── DEMO_02_layer1_features.ipynb   # Semantic feature engineering
│   ├── DEMO_03_layer2_ml_model.ipynb   # XGBoost + Model Registry
│   ├── DEMO_04_layer3_llm_analysis.ipynb   # Claude deep analysis
│   ├── DEMO_05_layer4_hybrid.ipynb     # ML + LLM pipeline metrics
│   ├── DEMO_06_layer5_finetuning.ipynb # Fine-tune for compliance
│   ├── DEMO_07_layer6_semantic_search.ipynb # Cortex Search
│   └── DEMO_08_resolution.ipynb        # Summary & results
│
└── data/
    ├── emails_synthetic.csv        # 10K synthetic emails
    └── finetune_training.jsonl     # Labeled training samples
```

---

## Demo Flow

### The Pain (DEMO_00)
Shows why keyword-based compliance fails:
- ~38% precision (lots of false alarms)
- ~67% recall (misses violations with subtle language)
- Compliance teams drowning in noise

### The Solution (DEMO_01-08)

| Demo | Layer | Focus | Snowflake Features |
|------|-------|-------|--------------------|
| 01 | - | Architecture blueprint | Conceptual overview |
| 02 | 1 | Semantic features | Cortex Embeddings, Feature Store |
| 03 | 2 | ML classification | Model Registry, XGBoost |
| 04 | 3 | LLM deep analysis | CORTEX.COMPLETE (Claude) |
| 05 | 4 | Hybrid pipeline | Three-way classification metrics |
| 06 | 5 | Fine-tuning | CORTEX.FINETUNE |
| 07 | 6 | Semantic search | Cortex Search Service |
| 08 | - | Resolution | Full system summary |

---

## The Dataset

**10,000 synthetic hedge fund emails** generated via Cortex LLM with realistic variation:

| Label | Distribution | Description |
|-------|-------------|-------------|
| CLEAN | ~67% | Normal business communications |
| INSIDER_TRADING | ~8% | MNPI sharing, trading tips |
| CONFIDENTIALITY_BREACH | ~9% | Client data leaks |
| PERSONAL_TRADING | ~8% | Undisclosed personal trades |
| INFO_BARRIER_VIOLATION | ~8% | Research/Trading wall breaches |

**Data characteristics:**
- LLM-generated with unique variation seeds (93%+ unique subjects)
- Subtle violations (not obvious language)
- Borderline clean emails (discuss sensitive topics legitimately)

---

## Key Innovation: Semantic Risk Features

Instead of keyword matching, we measure **semantic distance** from risk concepts:

| Feature | Description |
|---------|-------------|
| BASELINE_SIMILARITY | Distance to normal business language |
| MNPI_RISK_SCORE | Similarity to insider trading language |
| CONFIDENTIALITY_RISK_SCORE | Similarity to data leak language |
| PERSONAL_TRADING_RISK_SCORE | Similarity to personal trading language |
| INFO_BARRIER_RISK_SCORE | Similarity to wall-crossing language |
| CROSS_BARRIER_FLAG | Research↔Trading department indicator |

This captures **meaning, not keywords** - violators can change vocabulary but meaning still clusters near risk concepts.

---

## Three-Way Classification

The ML model outputs probability scores that drive a three-way decision:

| ML Decision | Threshold | Action |
|-------------|-----------|--------|
| **HIGH_RISK** | probability >= 0.7 | Auto-flag for review |
| **NEEDS_REVIEW** | 0.3 < probability < 0.7 | Send to LLM for analysis |
| **LOW_RISK** | probability <= 0.3 | Auto-clear |

This tiered approach means the LLM only processes ~21% of emails (the uncertain middle), dramatically reducing cost while improving recall.

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

## Cleanup

After the demo, remove all objects:

```sql
DROP DATABASE IF EXISTS COMPLIANCE_DEMO CASCADE;
DROP WAREHOUSE IF EXISTS COMPLIANCE_DEMO_WH;
```

---

## Notes

- **Data generation:** Uses Cortex LLM batch SQL for realistic, unique emails
- **Semantic features:** Computing embeddings for 10K emails takes ~2-3 minutes
- **Fine-tuning:** The FINETUNE job is async and takes 30-60 minutes
- **Cortex Search:** Service indexing may take a few minutes after creation

---

## License

Internal demo - not for distribution.
