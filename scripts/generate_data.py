#!/usr/bin/env python3
"""
Generate synthetic email dataset for hedge fund compliance demo.

Creates realistic email data with compliance labels for:
- CLEAN: Normal business communications
- INSIDER_TRADING: MNPI sharing, trading tips before announcements
- CONFIDENTIALITY_BREACH: Unauthorized client/fund info sharing
- PERSONAL_TRADING: Personal investment discussions, pre-clearance violations
- INFO_BARRIER_VIOLATION: Cross-department leaks (research <-> trading)

Usage:
    python scripts/generate_data.py
"""

import csv
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

NUM_EMAILS = 150  # Total emails to generate
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "emails_synthetic.csv"

# Label distribution (realistic class imbalance)
LABEL_WEIGHTS = {
    "CLEAN": 0.70,
    "INSIDER_TRADING": 0.08,
    "CONFIDENTIALITY_BREACH": 0.08,
    "PERSONAL_TRADING": 0.07,
    "INFO_BARRIER_VIOLATION": 0.07,
}

# Departments with information barrier implications
DEPARTMENTS = [
    "Research",
    "Trading",
    "Portfolio Management",
    "Compliance",
    "Operations",
    "Legal",
    "Client Relations",
]

# Research and Trading have strict information barriers
BARRIER_DEPTS = {"Research", "Trading"}

# ============================================================================
# Employee Directory
# ============================================================================

EMPLOYEES = [
    # Research
    {"name": "Sarah Chen", "email": "s.chen@acmefund.com", "dept": "Research"},
    {"name": "Marcus Webb", "email": "m.webb@acmefund.com", "dept": "Research"},
    {"name": "Priya Sharma", "email": "p.sharma@acmefund.com", "dept": "Research"},
    # Trading
    {"name": "James Morrison", "email": "j.morrison@acmefund.com", "dept": "Trading"},
    {"name": "Elena Volkov", "email": "e.volkov@acmefund.com", "dept": "Trading"},
    {"name": "David Park", "email": "d.park@acmefund.com", "dept": "Trading"},
    # Portfolio Management
    {"name": "Michael Torres", "email": "m.torres@acmefund.com", "dept": "Portfolio Management"},
    {"name": "Amanda Foster", "email": "a.foster@acmefund.com", "dept": "Portfolio Management"},
    # Compliance
    {"name": "Robert Hayes", "email": "r.hayes@acmefund.com", "dept": "Compliance"},
    {"name": "Jennifer Liu", "email": "j.liu@acmefund.com", "dept": "Compliance"},
    # Operations
    {"name": "Kevin O'Brien", "email": "k.obrien@acmefund.com", "dept": "Operations"},
    {"name": "Lisa Martinez", "email": "l.martinez@acmefund.com", "dept": "Operations"},
    # Legal
    {"name": "Thomas Grant", "email": "t.grant@acmefund.com", "dept": "Legal"},
    # Client Relations
    {"name": "Rachel Kim", "email": "r.kim@acmefund.com", "dept": "Client Relations"},
    {"name": "Andrew Bell", "email": "a.bell@acmefund.com", "dept": "Client Relations"},
]

# ============================================================================
# Email Templates by Label
# ============================================================================

TEMPLATES = {
    "CLEAN": [
        {
            "subject": "Q4 Portfolio Review Meeting",
            "body": "Hi team,\n\nJust a reminder about our Q4 portfolio review meeting scheduled for Thursday at 2pm. Please come prepared with your sector updates.\n\nThanks,\n{sender_name}",
        },
        {
            "subject": "Updated compliance training schedule",
            "body": "All,\n\nThe annual compliance training has been rescheduled to next Monday. Please block your calendars from 10am-12pm.\n\nBest,\n{sender_name}",
        },
        {
            "subject": "Office closure reminder - Holiday",
            "body": "Team,\n\nJust a friendly reminder that the office will be closed next Friday for the holiday. Please plan accordingly.\n\nRegards,\n{sender_name}",
        },
        {
            "subject": "RE: Lunch order for client meeting",
            "body": "Sounds good. I'll order from the usual place. Can you confirm headcount by noon?\n\n{sender_name}",
        },
        {
            "subject": "Monthly risk report - draft for review",
            "body": "Hi {recipient_name},\n\nAttached is the draft monthly risk report. Please review and send any comments by EOD Wednesday.\n\nThanks,\n{sender_name}",
        },
        {
            "subject": "Conference room booking",
            "body": "Hi,\n\nI've booked the large conference room for our team meeting on Tuesday. Calendar invite to follow.\n\n{sender_name}",
        },
        {
            "subject": "System maintenance notification",
            "body": "Team,\n\nIT has scheduled system maintenance for Saturday 2am-6am. Trading systems will be unavailable during this window.\n\nPlease plan accordingly.\n{sender_name}",
        },
        {
            "subject": "Welcome aboard!",
            "body": "Hi {recipient_name},\n\nWelcome to the team! I've set up your onboarding meetings for next week. Please reach out if you have any questions.\n\nBest,\n{sender_name}",
        },
        {
            "subject": "RE: Vacation request",
            "body": "Approved. Enjoy your time off!\n\n{sender_name}",
        },
        {
            "subject": "Weekly team sync notes",
            "body": "Hi all,\n\nAttached are the notes from today's sync. Action items highlighted in yellow.\n\nThanks,\n{sender_name}",
        },
    ],
    "INSIDER_TRADING": [
        {
            "subject": "URGENT - Acme Corp news coming",
            "body": "Hey,\n\nHeads up - I heard from my contact at Acme that they're announcing a major acquisition tomorrow morning. This isn't public yet. Might want to adjust positions before the announcement.\n\nKeep this between us.\n{sender_name}",
        },
        {
            "subject": "RE: Tech sector positions",
            "body": "Trust me on this one - load up on TECH before Friday. I can't say more but the earnings will blow expectations away. My source is solid.\n\n{sender_name}",
        },
        {
            "subject": "Pharma play - time sensitive",
            "body": "{recipient_name},\n\nFDA approval coming through for BioMed's drug application. Not announced yet but I got word from inside. This could move 30%+. Act fast.\n\n{sender_name}",
        },
        {
            "subject": "Quiet tip",
            "body": "Don't share this with anyone - GlobalBank is about to announce layoffs and restructuring. Their stock will tank. You didn't hear this from me.\n\n{sender_name}",
        },
        {
            "subject": "RE: Market outlook",
            "body": "Forget the analysis - I have inside info that MegaCorp's CEO is resigning next week. Dump the position before it goes public. This is solid intel.\n\n{sender_name}",
        },
    ],
    "CONFIDENTIALITY_BREACH": [
        {
            "subject": "FW: Client portfolio details",
            "body": "Hey,\n\nForwarding you the Westbrook pension fund's full portfolio breakdown. They're way overexposed to energy. Don't tell anyone I sent this.\n\n{sender_name}",
        },
        {
            "subject": "RE: Hedge fund strategies",
            "body": "Here's our fund's full trading strategy doc. I know you're at a competitor now but this might help you. Delete after reading.\n\n{sender_name}",
        },
        {
            "subject": "Client list - confidential",
            "body": "{recipient_name},\n\nAttaching our complete client list with AUM figures. This is strictly confidential but thought it might be useful for your research.\n\n{sender_name}",
        },
        {
            "subject": "Performance data you asked for",
            "body": "As discussed, here are the detailed performance numbers for our top 5 clients' accounts. Please don't share this externally - I could get in serious trouble.\n\n{sender_name}",
        },
        {
            "subject": "FW: Investor meeting notes",
            "body": "Forwarding the notes from yesterday's LP meeting. Contains sensitive info about their allocation plans. Keep this to yourself.\n\n{sender_name}",
        },
    ],
    "PERSONAL_TRADING": [
        {
            "subject": "My stock picks",
            "body": "Hey,\n\nI bought a bunch of shares in TechStartup Inc last week in my personal account. Looking good so far! You should get in before it pops.\n\n{sender_name}",
        },
        {
            "subject": "RE: Investment advice",
            "body": "For my PA, I'm going heavy into crypto this month. Already moved $50k. Don't mention this to compliance - it's under their radar.\n\n{sender_name}",
        },
        {
            "subject": "Quick trade question",
            "body": "{recipient_name},\n\nThinking of buying some puts on a stock we're long in the fund. Just in my personal account though. Should be fine right? Don't want to bother with pre-clearance.\n\n{sender_name}",
        },
        {
            "subject": "Side investment opportunity",
            "body": "Found a great IPO coming up. Going to buy in my wife's account to avoid the disclosure requirements. Let me know if you want details.\n\n{sender_name}",
        },
        {
            "subject": "RE: After hours trading",
            "body": "Yeah I've been day trading in my Schwab account. Made $20k last month! Haven't reported it yet since it's small amounts. Compliance doesn't need to know everything.\n\n{sender_name}",
        },
    ],
    "INFO_BARRIER_VIOLATION": [
        {
            "subject": "Research update - please read",
            "body": "James,\n\nI know we're not supposed to share this directly, but my team just finished our analysis on Quantum Industries. Rating it a strong buy. Wanted you to have a heads up before it goes through official channels.\n\n- Sarah (Research)",
        },
        {
            "subject": "RE: Position sizing question",
            "body": "Between us - research is about to downgrade SolarTech to sell. I'd reduce exposure before the report comes out next week. Don't tell anyone I told you.\n\n{sender_name}",
        },
        {
            "subject": "Quick question about analysis",
            "body": "Hey {recipient_name},\n\nI know we have the Chinese wall and all, but can you give me a preview of what research is saying about the industrial sector? Need to know for position sizing today.\n\n{sender_name}",
        },
        {
            "subject": "FW: Draft research report",
            "body": "Forwarding the draft research report before it's published. Thought you might want to see the price targets before the trading desk gets them officially.\n\n{sender_name}",
        },
        {
            "subject": "Sector meeting notes - confidential",
            "body": "{recipient_name},\n\nHere are notes from our research team meeting on healthcare. We're changing several ratings next week. Keep this between us - I'm not supposed to share pre-publication.\n\n{sender_name}",
        },
    ],
}

# ============================================================================
# Generator Functions
# ============================================================================


def random_timestamp(days_back: int = 90) -> datetime:
    """Generate a random timestamp within business hours (mostly)."""
    base = datetime.now() - timedelta(days=random.randint(1, days_back))
    # 80% during business hours, 20% after hours (suspicious pattern)
    if random.random() < 0.8:
        hour = random.randint(8, 18)
    else:
        hour = random.choice([6, 7, 19, 20, 21, 22])
    minute = random.randint(0, 59)
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def pick_sender_recipient(label: str) -> tuple[dict, dict]:
    """Select sender/recipient pair appropriate for the label."""
    if label == "INFO_BARRIER_VIOLATION":
        # Must be cross-barrier communication (Research <-> Trading)
        research = [e for e in EMPLOYEES if e["dept"] == "Research"]
        trading = [e for e in EMPLOYEES if e["dept"] == "Trading"]
        if random.random() < 0.5:
            sender = random.choice(research)
            recipient = random.choice(trading)
        else:
            sender = random.choice(trading)
            recipient = random.choice(research)
    else:
        # Random selection
        sender = random.choice(EMPLOYEES)
        recipient = random.choice([e for e in EMPLOYEES if e != sender])
    return sender, recipient


def generate_email(label: str) -> dict:
    """Generate a single email record."""
    sender, recipient = pick_sender_recipient(label)
    template = random.choice(TEMPLATES[label])

    sender_name = sender["name"].split()[0]
    recipient_name = recipient["name"].split()[0]

    subject = template["subject"]
    body = template["body"].format(
        sender_name=sender_name,
        recipient_name=recipient_name,
    )

    # Occasionally add CC (10% of emails)
    cc = ""
    if random.random() < 0.1:
        cc_candidates = [e for e in EMPLOYEES if e not in (sender, recipient)]
        if cc_candidates:
            cc = random.choice(cc_candidates)["email"]

    return {
        "email_id": str(uuid.uuid4()),
        "sender": sender["email"],
        "recipient": recipient["email"],
        "cc": cc,
        "subject": subject,
        "body": body,
        "sent_at": random_timestamp().isoformat(),
        "sender_dept": sender["dept"],
        "recipient_dept": recipient["dept"],
        "compliance_label": label,
    }


def generate_dataset(num_emails: int) -> list[dict]:
    """Generate the full dataset with proper label distribution."""
    emails = []
    labels = list(LABEL_WEIGHTS.keys())
    weights = list(LABEL_WEIGHTS.values())

    for _ in range(num_emails):
        label = random.choices(labels, weights=weights, k=1)[0]
        emails.append(generate_email(label))

    # Sort by timestamp for realism
    emails.sort(key=lambda x: x["sent_at"])
    return emails


def main():
    """Generate and save the synthetic email dataset."""
    print(f"Generating {NUM_EMAILS} synthetic emails...")

    emails = generate_dataset(NUM_EMAILS)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fieldnames = [
        "email_id",
        "sender",
        "recipient",
        "cc",
        "subject",
        "body",
        "sent_at",
        "sender_dept",
        "recipient_dept",
        "compliance_label",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(emails)

    # Print summary
    label_counts = {}
    for email in emails:
        label = email["compliance_label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nSaved to: {OUTPUT_PATH}")
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / len(emails) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

