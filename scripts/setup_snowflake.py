#!/usr/bin/env python3
"""
Setup Snowflake environment and load data for the compliance demo.

This script:
1. Creates the demo database, schemas, and warehouse
2. Creates the required tables
3. Uploads data files to a stage
4. Loads data into tables

Prerequisites:
- Run generate_data.py first to create the CSV files
- Snowflake connection configured via ~/.snowflake/config.toml
  OR snowflake.config in this repo

Usage:
    python scripts/setup_snowflake.py
"""

from pathlib import Path

from snowflake.snowpark import Session


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
EMAILS_FILE = DATA_DIR / "emails_synthetic.csv"
FINETUNE_FILE = DATA_DIR / "finetune_training.jsonl"

# Demo environment names
DATABASE_NAME = "COMPLIANCE_DEMO"
WAREHOUSE_NAME = "COMPLIANCE_DEMO_WH"
SCHEMA_EMAIL = "EMAIL_SURVEILLANCE"
SCHEMA_ML = "ML"
SCHEMA_SEARCH = "SEARCH"


# ============================================================================
# Snowflake Connection
# ============================================================================


def get_session() -> Session:
    """
    Create a Snowpark session using the default config.
    
    Uses ~/.snowflake/config.toml by default (Snowpark session builder).
    """
    print("Connecting to Snowflake...")
    
    # Use the Snowpark session builder with default config
    # This automatically reads from ~/.snowflake/config.toml
    session = Session.builder.getOrCreate()
    
    print(f"  Connected as: {session.get_current_user()}")
    print(f"  Account: {session.get_current_account()}")
    print(f"  Role: {session.get_current_role()}")
    
    return session


# ============================================================================
# Setup Functions
# ============================================================================


def create_warehouse(session: Session) -> None:
    """Create the demo warehouse."""
    print(f"\nCreating warehouse: {WAREHOUSE_NAME}")
    session.sql(f"""
        CREATE WAREHOUSE IF NOT EXISTS {WAREHOUSE_NAME}
            WAREHOUSE_SIZE = 'MEDIUM'
            AUTO_SUSPEND = 60
            AUTO_RESUME = TRUE
            INITIALLY_SUSPENDED = TRUE
            COMMENT = 'Warehouse for ML compliance demo'
    """).collect()
    
    session.sql(f"USE WAREHOUSE {WAREHOUSE_NAME}").collect()
    print(f"  ✓ Warehouse created and activated")


def create_database_and_schemas(session: Session) -> None:
    """Create the demo database and schemas."""
    print(f"\nCreating database: {DATABASE_NAME}")
    session.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}").collect()
    session.sql(f"USE DATABASE {DATABASE_NAME}").collect()
    print(f"  ✓ Database created")
    
    print(f"\nCreating schemas...")
    for schema in [SCHEMA_EMAIL, SCHEMA_ML, SCHEMA_SEARCH]:
        session.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}").collect()
        print(f"  ✓ Schema {schema} created")


def create_tables(session: Session) -> None:
    """Create the required tables."""
    print(f"\nCreating tables...")
    
    # Main emails table
    session.sql(f"""
        CREATE OR REPLACE TABLE {DATABASE_NAME}.{SCHEMA_EMAIL}.EMAILS (
            EMAIL_ID        VARCHAR(36) PRIMARY KEY,
            SENDER          VARCHAR(100) NOT NULL,
            RECIPIENT       VARCHAR(100) NOT NULL,
            CC              VARCHAR(500),
            SUBJECT         VARCHAR(500),
            BODY            VARCHAR(16777216),
            SENT_AT         TIMESTAMP_NTZ NOT NULL,
            SENDER_DEPT     VARCHAR(50),
            RECIPIENT_DEPT  VARCHAR(50),
            COMPLIANCE_LABEL VARCHAR(50),
            LOADED_AT       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """).collect()
    print(f"  ✓ Table {SCHEMA_EMAIL}.EMAILS created")
    
    # Fine-tuning training data table
    session.sql(f"""
        CREATE OR REPLACE TABLE {DATABASE_NAME}.{SCHEMA_ML}.FINETUNE_TRAINING (
            SAMPLE_ID       NUMBER AUTOINCREMENT,
            PROMPT          VARCHAR(16777216),
            COMPLETION      VARCHAR(16777216),
            LOADED_AT       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """).collect()
    print(f"  ✓ Table {SCHEMA_ML}.FINETUNE_TRAINING created")
    
    # Embeddings table (for vector search)
    session.sql(f"""
        CREATE OR REPLACE TABLE {DATABASE_NAME}.{SCHEMA_SEARCH}.EMAIL_EMBEDDINGS (
            EMAIL_ID        VARCHAR(36) PRIMARY KEY,
            SUBJECT         VARCHAR(500),
            BODY_PREVIEW    VARCHAR(1000),
            COMPLIANCE_LABEL VARCHAR(50),
            EMBEDDING       VECTOR(FLOAT, 768),
            EMBEDDED_AT     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """).collect()
    print(f"  ✓ Table {SCHEMA_SEARCH}.EMAIL_EMBEDDINGS created")


def create_stages(session: Session) -> None:
    """Create internal stages for data loading."""
    print(f"\nCreating stages...")
    
    session.sql(f"""
        CREATE OR REPLACE STAGE {DATABASE_NAME}.{SCHEMA_EMAIL}.DATA_STAGE
            DIRECTORY = (ENABLE = TRUE)
            COMMENT = 'Stage for loading demo data files'
    """).collect()
    print(f"  ✓ Stage {SCHEMA_EMAIL}.DATA_STAGE created")


def upload_files(session: Session) -> None:
    """Upload data files to the stage."""
    print(f"\nUploading data files to stage...")
    
    # Check if files exist
    if not EMAILS_FILE.exists():
        raise FileNotFoundError(
            f"Email data file not found: {EMAILS_FILE}\n"
            "Run 'python scripts/generate_data.py' first."
        )
    
    if not FINETUNE_FILE.exists():
        raise FileNotFoundError(
            f"Fine-tuning data file not found: {FINETUNE_FILE}\n"
            "Run 'python scripts/generate_data.py' first."
        )
    
    # Upload emails CSV
    session.file.put(
        str(EMAILS_FILE),
        f"@{DATABASE_NAME}.{SCHEMA_EMAIL}.DATA_STAGE",
        auto_compress=False,
        overwrite=True,
    )
    print(f"  ✓ Uploaded {EMAILS_FILE.name}")
    
    # Upload fine-tuning JSONL
    session.file.put(
        str(FINETUNE_FILE),
        f"@{DATABASE_NAME}.{SCHEMA_EMAIL}.DATA_STAGE",
        auto_compress=False,
        overwrite=True,
    )
    print(f"  ✓ Uploaded {FINETUNE_FILE.name}")


def load_emails(session: Session) -> None:
    """Load email data from stage into table."""
    print(f"\nLoading emails into table...")
    
    # Create file format
    session.sql(f"""
        CREATE OR REPLACE FILE FORMAT {DATABASE_NAME}.{SCHEMA_EMAIL}.CSV_FORMAT
            TYPE = 'CSV'
            FIELD_OPTIONALLY_ENCLOSED_BY = '"'
            SKIP_HEADER = 1
            NULL_IF = ('')
            FIELD_DELIMITER = ','
            ESCAPE_UNENCLOSED_FIELD = NONE
    """).collect()
    
    # Load data
    session.sql(f"""
        COPY INTO {DATABASE_NAME}.{SCHEMA_EMAIL}.EMAILS (
            EMAIL_ID, SENDER, RECIPIENT, CC, SUBJECT, BODY, 
            SENT_AT, SENDER_DEPT, RECIPIENT_DEPT, COMPLIANCE_LABEL
        )
        FROM @{DATABASE_NAME}.{SCHEMA_EMAIL}.DATA_STAGE/emails_synthetic.csv
        FILE_FORMAT = {DATABASE_NAME}.{SCHEMA_EMAIL}.CSV_FORMAT
        ON_ERROR = 'CONTINUE'
    """).collect()
    
    # Verify count
    count = session.sql(f"""
        SELECT COUNT(*) as cnt FROM {DATABASE_NAME}.{SCHEMA_EMAIL}.EMAILS
    """).collect()[0]["CNT"]
    print(f"  ✓ Loaded {count:,} emails")


def load_finetune_data(session: Session) -> None:
    """Load fine-tuning data from stage into table."""
    print(f"\nLoading fine-tuning samples into table...")
    
    # Create file format for JSONL
    session.sql(f"""
        CREATE OR REPLACE FILE FORMAT {DATABASE_NAME}.{SCHEMA_EMAIL}.JSONL_FORMAT
            TYPE = 'JSON'
            STRIP_OUTER_ARRAY = FALSE
    """).collect()
    
    # Load data (JSONL needs special handling)
    session.sql(f"""
        COPY INTO {DATABASE_NAME}.{SCHEMA_ML}.FINETUNE_TRAINING (PROMPT, COMPLETION)
        FROM (
            SELECT 
                $1:prompt::STRING,
                $1:completion::STRING
            FROM @{DATABASE_NAME}.{SCHEMA_EMAIL}.DATA_STAGE/finetune_training.jsonl
        )
        FILE_FORMAT = {DATABASE_NAME}.{SCHEMA_EMAIL}.JSONL_FORMAT
        ON_ERROR = 'CONTINUE'
    """).collect()
    
    # Verify count
    count = session.sql(f"""
        SELECT COUNT(*) as cnt FROM {DATABASE_NAME}.{SCHEMA_ML}.FINETUNE_TRAINING
    """).collect()[0]["CNT"]
    print(f"  ✓ Loaded {count} fine-tuning samples")


def verify_setup(session: Session) -> None:
    """Verify the setup completed successfully."""
    print(f"\n{'='*60}")
    print("SETUP VERIFICATION")
    print(f"{'='*60}")
    
    # Check email distribution
    print("\nEmail label distribution:")
    dist = session.sql(f"""
        SELECT COMPLIANCE_LABEL, COUNT(*) as CNT
        FROM {DATABASE_NAME}.{SCHEMA_EMAIL}.EMAILS
        GROUP BY 1
        ORDER BY 2 DESC
    """).collect()
    
    for row in dist:
        print(f"  {row['COMPLIANCE_LABEL']}: {row['CNT']:,}")
    
    # Check fine-tuning samples
    print("\nFine-tuning sample count:")
    ft_count = session.sql(f"""
        SELECT COUNT(*) as cnt FROM {DATABASE_NAME}.{SCHEMA_ML}.FINETUNE_TRAINING
    """).collect()[0]["CNT"]
    print(f"  {ft_count} samples ready for fine-tuning")
    
    # List stage contents
    print("\nStage contents:")
    files = session.sql(f"""
        LIST @{DATABASE_NAME}.{SCHEMA_EMAIL}.DATA_STAGE
    """).collect()
    for f in files:
        print(f"  {f['name']}: {f['size']:,} bytes")
    
    print(f"\n{'='*60}")
    print("✅ Setup complete! Ready for demo.")
    print(f"{'='*60}")
    print(f"\nTo use in Snowsight:")
    print(f"  USE WAREHOUSE {WAREHOUSE_NAME};")
    print(f"  USE DATABASE {DATABASE_NAME};")
    print(f"  USE SCHEMA {SCHEMA_EMAIL};")


# ============================================================================
# Main
# ============================================================================


def main():
    """Run the complete setup."""
    print("=" * 60)
    print("Snowflake Compliance Demo Setup")
    print("=" * 60)
    
    # Connect
    session = get_session()
    
    try:
        # Setup environment
        create_warehouse(session)
        create_database_and_schemas(session)
        create_tables(session)
        create_stages(session)
        
        # Load data
        upload_files(session)
        load_emails(session)
        load_finetune_data(session)
        
        # Verify
        verify_setup(session)
        
    finally:
        session.close()


if __name__ == "__main__":
    main()
