#!/usr/bin/env python3
"""
Cleanup Snowflake environment for the compliance demo.

This script removes all objects created by setup_snowflake.py:
- Database: COMPLIANCE_DEMO (and all schemas/tables within)
- Warehouse: COMPLIANCE_DEMO_WH

Usage:
    python scripts/cleanup_snowflake.py
"""

from snowflake.snowpark import Session


DATABASE_NAME = "COMPLIANCE_DEMO"
WAREHOUSE_NAME = "COMPLIANCE_DEMO_WH"


def get_session() -> Session:
    print("Connecting to Snowflake...")
    session = Session.builder.getOrCreate()
    print(f"  Connected as: {session.get_current_user()}")
    return session


def cleanup(session: Session) -> None:
    print(f"\nDropping database: {DATABASE_NAME}")
    session.sql(f"DROP DATABASE IF EXISTS {DATABASE_NAME}").collect()
    print(f"  ✓ Database dropped")

    print(f"\nDropping warehouse: {WAREHOUSE_NAME}")
    session.sql(f"DROP WAREHOUSE IF EXISTS {WAREHOUSE_NAME}").collect()
    print(f"  ✓ Warehouse dropped")


def main():
    print("=" * 60)
    print("Snowflake Compliance Demo Cleanup")
    print("=" * 60)

    session = get_session()

    try:
        cleanup(session)
        print(f"\n{'='*60}")
        print("✅ Cleanup complete!")
        print(f"{'='*60}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
