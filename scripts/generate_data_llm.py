#!/usr/bin/env python3
"""
Generate synthetic email dataset using Snowflake Cortex LLM with batch SQL.
Uses parallelized SQL queries for much faster generation.
"""

import csv
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from snowflake.snowpark import Session

NUM_EMAILS = 10000  # Full dataset
OUTPUT_DIR = Path(__file__).parent.parent / "data"
EMAILS_OUTPUT = OUTPUT_DIR / "emails_synthetic.csv"

LABEL_DISTRIBUTION = {
    "CLEAN": 0.67,
    "INSIDER_TRADING": 0.08,
    "CONFIDENTIALITY_BREACH": 0.09,
    "PERSONAL_TRADING": 0.08,
    "INFO_BARRIER_VIOLATION": 0.08,
}

EMPLOYEES = [
    {"name": "Sarah Chen", "email": "s.chen@acmefund.com", "dept": "Research"},
    {"name": "Marcus Webb", "email": "m.webb@acmefund.com", "dept": "Research"},
    {"name": "Priya Sharma", "email": "p.sharma@acmefund.com", "dept": "Research"},
    {"name": "Daniel Kim", "email": "d.kim@acmefund.com", "dept": "Research"},
    {"name": "James Morrison", "email": "j.morrison@acmefund.com", "dept": "Trading"},
    {"name": "Elena Volkov", "email": "e.volkov@acmefund.com", "dept": "Trading"},
    {"name": "David Park", "email": "d.park@acmefund.com", "dept": "Trading"},
    {"name": "Nicole Brown", "email": "n.brown@acmefund.com", "dept": "Trading"},
    {"name": "Michael Torres", "email": "m.torres@acmefund.com", "dept": "Portfolio Management"},
    {"name": "Amanda Foster", "email": "a.foster@acmefund.com", "dept": "Portfolio Management"},
    {"name": "Robert Hayes", "email": "r.hayes@acmefund.com", "dept": "Compliance"},
    {"name": "Jennifer Liu", "email": "j.liu@acmefund.com", "dept": "Compliance"},
    {"name": "Kevin O'Brien", "email": "k.obrien@acmefund.com", "dept": "Operations"},
    {"name": "Lisa Martinez", "email": "l.martinez@acmefund.com", "dept": "Operations"},
    {"name": "Thomas Grant", "email": "t.grant@acmefund.com", "dept": "Legal"},
    {"name": "Susan Clark", "email": "s.clark@acmefund.com", "dept": "Legal"},
    {"name": "Rachel Kim", "email": "r.kim@acmefund.com", "dept": "Client Relations"},
    {"name": "Andrew Bell", "email": "a.bell@acmefund.com", "dept": "Client Relations"},
    {"name": "Christopher Lee", "email": "c.lee@acmefund.com", "dept": "Risk Management"},
    {"name": "Michelle Wang", "email": "m.wang@acmefund.com", "dept": "Risk Management"},
    {"name": "Brian Johnson", "email": "b.johnson@acmefund.com", "dept": "Technology"},
    {"name": "Jessica Taylor", "email": "j.taylor@acmefund.com", "dept": "Technology"},
]

LABEL_PROMPTS = {
    "CLEAN": """Generate a UNIQUE realistic hedge fund internal email that is completely clean and compliant.

CHOOSE ONE SPECIFIC SCENARIO (vary the topic, company names, and details each time):
- Meeting about [specific topic: earnings, budget, hiring, strategy, vendor, etc.]
- Asking about [specific software, system access, document location]
- Following up on [specific project, deadline, deliverable]
- Sharing [public news article, conference notes, industry report]
- Social: [birthday, lunch, happy hour, congratulations]
- HR: [time off, expense reports, performance review scheduling]
- IT: [password reset, equipment request, software issue]
- Client: [preparing materials for NAMED client meeting]

Vary the tone (formal/casual), length, and style. Use specific company names (TechCorp, MedPharm, GlobalRetail, etc.), dates, and details. Do NOT use generic subjects like 'Q4 Review' - be specific.""",

    "INSIDER_TRADING": """Generate a UNIQUE hedge fund email with an INSIDER TRADING violation.

CHOOSE ONE SPECIFIC SCENARIO with a SPECIFIC made-up company name:
- Earnings tip: [CompanyName] Q results from CFO contact
- M&A leak: [Company] acquiring/being acquired, from banker
- FDA: [BioPharmName] drug approval from inside contact
- Contract: [Company] winning [specific contract] from govt contact
- Executive: [Company] CEO change from board member
- Restructuring: [Company] layoffs from employee

Vary HOW the info was obtained (dinner, gym, golf, kids' school, old roommate, etc.) and HOW SUBTLE the language is. Be creative with company names.""",

    "CONFIDENTIALITY_BREACH": """Generate a UNIQUE hedge fund email with CONFIDENTIALITY BREACH.

CHOOSE ONE SPECIFIC SCENARIO:
- Sharing [NAMED client]'s portfolio details for competitive intel
- Forwarding [specific proprietary strategy doc] to old colleague at competitor
- Sending fee schedules for [named client] to help friend negotiate
- Sharing LP names and allocation sizes for business development
- Forwarding internal investigation notes about [specific incident]
- Sharing salary/bonus data for [specific role/team]

Vary the REASON (job interview prep, helping friend, academic research, benchmarking study, etc.) and make it feel like a natural conversation.""",

    "PERSONAL_TRADING": """Generate a UNIQUE hedge fund email with PERSONAL TRADING violation.

CHOOSE ONE SPECIFIC SCENARIO with SPECIFIC details:
- Bought [specific stock/option] without pre-clearance, up [X]%
- Trading in [IRA/Schwab/Fidelity/Robinhood] account unreported
- Had spouse/parent buy [stock] to avoid disclosure
- Investment club traded [restricted securities]
- Kept trades under $[amount] to avoid reporting
- Bought [international stock] assuming rules don't apply

Mention SPECIFIC dollar amounts, percentages, broker names. Make it conversational - often mentioned casually alongside other topics.""",

    "INFO_BARRIER_VIOLATION": """Generate a UNIQUE hedge fund email violating the Research/Trading INFO BARRIER (Chinese Wall).

Sender MUST be Research and recipient MUST be Trading (or vice versa).

CHOOSE ONE SPECIFIC SCENARIO with SPECIFIC company name:
- [CompanyName] rating change (upgrade/downgrade) coming [day]
- Price target moving from $[X] to $[Y] for [Company]
- Analyst [name] is bearish/bullish on [sector/company]
- Research report on [Company] publishing [specific day]
- "The team's been doing deep work on [Company]" - hints at conclusions

Vary how explicit vs coded the message is. Include specific company names, sectors, and timing."""
}

def random_timestamp(days_back: int = 180) -> str:
    base = datetime.now() - timedelta(days=random.randint(1, days_back))
    if random.random() < 0.8:
        hour = random.randint(8, 18)
    else:
        hour = random.choice([6, 7, 19, 20, 21, 22, 23])
    minute = random.randint(0, 59)
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0).isoformat()


def pick_sender_recipient(label: str) -> tuple[dict, dict]:
    if label == "INFO_BARRIER_VIOLATION":
        research = [e for e in EMPLOYEES if e["dept"] == "Research"]
        trading = [e for e in EMPLOYEES if e["dept"] == "Trading"]
        if random.random() < 0.5:
            sender = random.choice(research)
            recipient = random.choice(trading)
        else:
            sender = random.choice(trading)
            recipient = random.choice(research)
    else:
        sender = random.choice(EMPLOYEES)
        recipient = random.choice([e for e in EMPLOYEES if e != sender])
    return sender, recipient


def main():
    print("=" * 60)
    print("Generating Emails with Cortex LLM (Batch SQL)")
    print("=" * 60)
    
    session = Session.builder.getOrCreate()
    session.sql("USE WAREHOUSE COMPLIANCE_DEMO_WH").collect()
    session.sql("USE DATABASE COMPLIANCE_DEMO").collect()
    session.sql("USE SCHEMA ML").collect()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nStep 1: Creating prompts table...")
    
    prompts_data = []
    for label, pct in LABEL_DISTRIBUTION.items():
        count = int(NUM_EMAILS * pct)
        for i in range(count):
            sender, recipient = pick_sender_recipient(label)
            sender_name = sender["name"].split()[0]
            
            variation_seed = random.randint(1000, 9999)
            company_seeds = ["Nexus", "Vertex", "Pinnacle", "Catalyst", "Horizon", "Summit", "Atlas", "Quantum", "Nova", "Zenith", 
                           "Apex", "Vector", "Ionic", "Prism", "Flux", "Stellar", "Meridian", "Eclipse", "Vanguard", "Pioneer"]
            random_company = random.choice(company_seeds) + random.choice(["Tech", "Bio", "Med", "Corp", "Systems", "Labs", "Global", "Holdings", "Industries", "Group"])
            random_topic = random.choice(["earnings", "merger", "product launch", "restructuring", "contract", "FDA review", "IPO", "partnership", "acquisition", "dividend"])
            
            prompt = f"""{LABEL_PROMPTS[label]}

UNIQUE SEED #{variation_seed} - Use company name hint: {random_company}, topic hint: {random_topic}
Sender: {sender["name"]} ({sender["dept"]})
Recipient: {recipient["name"]} ({recipient["dept"]})
Sign off as: {sender_name}

Generate a UNIQUE subject (NOT generic like "Q4 Review") and body. JSON only."""
            
            cc = ""
            if random.random() < 0.1:
                cc_candidates = [e for e in EMPLOYEES if e not in (sender, recipient)]
                if cc_candidates:
                    cc = random.choice(cc_candidates)["email"]
            
            prompts_data.append({
                "email_id": str(uuid.uuid4()),
                "sender_email": sender["email"],
                "recipient_email": recipient["email"],
                "cc": cc,
                "sender_dept": sender["dept"],
                "recipient_dept": recipient["dept"],
                "compliance_label": label,
                "sent_at": random_timestamp(),
                "prompt": prompt
            })
    
    print(f"  Created {len(prompts_data)} prompts")
    
    session.sql("CREATE SCHEMA IF NOT EXISTS COMPLIANCE_DEMO.TEMP").collect()
    
    session.sql("""
        CREATE OR REPLACE TABLE COMPLIANCE_DEMO.TEMP.EMAIL_PROMPTS (
            email_id STRING,
            sender_email STRING,
            recipient_email STRING,
            cc STRING,
            sender_dept STRING,
            recipient_dept STRING,
            compliance_label STRING,
            sent_at STRING,
            prompt STRING
        )
    """).collect()
    
    print("\nStep 2: Uploading prompts to Snowflake...")
    prompts_df = session.create_dataframe(
        [[p["email_id"], p["sender_email"], p["recipient_email"], p["cc"], 
          p["sender_dept"], p["recipient_dept"], p["compliance_label"], 
          p["sent_at"], p["prompt"]] for p in prompts_data],
        schema=["email_id", "sender_email", "recipient_email", "cc", 
                "sender_dept", "recipient_dept", "compliance_label", "sent_at", "prompt"]
    )
    prompts_df.write.mode("overwrite").save_as_table("COMPLIANCE_DEMO.TEMP.EMAIL_PROMPTS")
    print(f"  Uploaded {len(prompts_data)} rows")
    
    print("\nStep 3: Generating emails with Cortex LLM (batch)...")
    print("  This will take a few minutes...")
    
    session.sql("""
        CREATE OR REPLACE TABLE COMPLIANCE_DEMO.TEMP.GENERATED_EMAILS AS
        SELECT 
            email_id,
            sender_email,
            recipient_email,
            cc,
            sender_dept,
            recipient_dept,
            compliance_label,
            sent_at,
            AI_COMPLETE(
                model => 'claude-haiku-4-5',
                prompt => prompt,
                response_format => {
                    'type': 'json',
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'subject': {'type': 'string'},
                            'body': {'type': 'string'}
                        },
                        'required': ['subject', 'body']
                    }
                }
            ) as llm_response
        FROM COMPLIANCE_DEMO.TEMP.EMAIL_PROMPTS
    """).collect()
    
    print("  Generation complete!")
    
    print("\nStep 4: Extracting results...")
    results = session.sql("""
        SELECT 
            email_id,
            sender_email as sender,
            recipient_email as recipient,
            cc,
            PARSE_JSON(llm_response):subject::STRING as subject,
            PARSE_JSON(llm_response):body::STRING as body,
            sent_at,
            sender_dept,
            recipient_dept,
            compliance_label
        FROM COMPLIANCE_DEMO.TEMP.GENERATED_EMAILS
    """).collect()
    
    emails = []
    for row in results:
        emails.append({
            "email_id": row["EMAIL_ID"],
            "sender": row["SENDER"],
            "recipient": row["RECIPIENT"],
            "cc": row["CC"] or "",
            "subject": row["SUBJECT"],
            "body": row["BODY"],
            "sent_at": row["SENT_AT"],
            "sender_dept": row["SENDER_DEPT"],
            "recipient_dept": row["RECIPIENT_DEPT"],
            "compliance_label": row["COMPLIANCE_LABEL"],
        })
    
    emails.sort(key=lambda x: x["sent_at"])
    
    fieldnames = [
        "email_id", "sender", "recipient", "cc", "subject", "body",
        "sent_at", "sender_dept", "recipient_dept", "compliance_label"
    ]
    
    with open(EMAILS_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(emails)
    
    print(f"\nSaved {len(emails)} emails to {EMAILS_OUTPUT}")
    
    label_counts = {}
    for email in emails:
        label = email["compliance_label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / len(emails) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    session.sql("DROP TABLE IF EXISTS COMPLIANCE_DEMO.TEMP.EMAIL_PROMPTS").collect()
    session.sql("DROP TABLE IF EXISTS COMPLIANCE_DEMO.TEMP.GENERATED_EMAILS").collect()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
