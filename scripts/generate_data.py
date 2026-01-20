#!/usr/bin/env python3
"""
Generate synthetic email dataset for hedge fund compliance demo.

Creates:
1. emails_synthetic.csv - 10,000 emails for the main demo
2. finetune_training.jsonl - 500 labeled samples for LLM fine-tuning

Compliance labels:
- CLEAN: Normal business communications
- INSIDER_TRADING: MNPI sharing, trading tips before announcements
- CONFIDENTIALITY_BREACH: Unauthorized client/fund info sharing
- PERSONAL_TRADING: Personal investment discussions, pre-clearance violations
- INFO_BARRIER_VIOLATION: Cross-department leaks (research <-> trading)

Usage:
    python scripts/generate_data.py
"""

import csv
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================

NUM_EMAILS = 10000  # Main dataset size
NUM_FINETUNE_SAMPLES = 500  # Fine-tuning training samples

OUTPUT_DIR = Path(__file__).parent.parent / "data"
EMAILS_OUTPUT = OUTPUT_DIR / "emails_synthetic.csv"
FINETUNE_OUTPUT = OUTPUT_DIR / "finetune_training.jsonl"

# Label distribution (realistic class imbalance - most emails are clean)
LABEL_WEIGHTS = {
    "CLEAN": 0.70,
    "INSIDER_TRADING": 0.08,
    "CONFIDENTIALITY_BREACH": 0.08,
    "PERSONAL_TRADING": 0.07,
    "INFO_BARRIER_VIOLATION": 0.07,
}

# Fine-tuning sample distribution (more balanced for training)
FINETUNE_DISTRIBUTION = {
    "CLEAN": 200,
    "INSIDER_TRADING": 75,
    "CONFIDENTIALITY_BREACH": 75,
    "PERSONAL_TRADING": 75,
    "INFO_BARRIER_VIOLATION": 75,
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
    "Risk Management",
    "Technology",
]

# Research and Trading have strict information barriers (Chinese walls)
BARRIER_DEPTS = {"Research", "Trading"}


# ============================================================================
# Employee Directory (Expanded)
# ============================================================================

EMPLOYEES = [
    # Research Team
    {"name": "Sarah Chen", "email": "s.chen@acmefund.com", "dept": "Research"},
    {"name": "Marcus Webb", "email": "m.webb@acmefund.com", "dept": "Research"},
    {"name": "Priya Sharma", "email": "p.sharma@acmefund.com", "dept": "Research"},
    {"name": "Daniel Kim", "email": "d.kim@acmefund.com", "dept": "Research"},
    {"name": "Laura Mitchell", "email": "l.mitchell@acmefund.com", "dept": "Research"},
    # Trading Team
    {"name": "James Morrison", "email": "j.morrison@acmefund.com", "dept": "Trading"},
    {"name": "Elena Volkov", "email": "e.volkov@acmefund.com", "dept": "Trading"},
    {"name": "David Park", "email": "d.park@acmefund.com", "dept": "Trading"},
    {"name": "Nicole Brown", "email": "n.brown@acmefund.com", "dept": "Trading"},
    {"name": "Ryan Cooper", "email": "r.cooper@acmefund.com", "dept": "Trading"},
    # Portfolio Management
    {"name": "Michael Torres", "email": "m.torres@acmefund.com", "dept": "Portfolio Management"},
    {"name": "Amanda Foster", "email": "a.foster@acmefund.com", "dept": "Portfolio Management"},
    {"name": "Steven Wright", "email": "s.wright@acmefund.com", "dept": "Portfolio Management"},
    # Compliance
    {"name": "Robert Hayes", "email": "r.hayes@acmefund.com", "dept": "Compliance"},
    {"name": "Jennifer Liu", "email": "j.liu@acmefund.com", "dept": "Compliance"},
    {"name": "Patricia Adams", "email": "p.adams@acmefund.com", "dept": "Compliance"},
    # Operations
    {"name": "Kevin O'Brien", "email": "k.obrien@acmefund.com", "dept": "Operations"},
    {"name": "Lisa Martinez", "email": "l.martinez@acmefund.com", "dept": "Operations"},
    # Legal
    {"name": "Thomas Grant", "email": "t.grant@acmefund.com", "dept": "Legal"},
    {"name": "Susan Clark", "email": "s.clark@acmefund.com", "dept": "Legal"},
    # Client Relations
    {"name": "Rachel Kim", "email": "r.kim@acmefund.com", "dept": "Client Relations"},
    {"name": "Andrew Bell", "email": "a.bell@acmefund.com", "dept": "Client Relations"},
    {"name": "Emily Davis", "email": "e.davis@acmefund.com", "dept": "Client Relations"},
    # Risk Management
    {"name": "Christopher Lee", "email": "c.lee@acmefund.com", "dept": "Risk Management"},
    {"name": "Michelle Wang", "email": "m.wang@acmefund.com", "dept": "Risk Management"},
    # Technology
    {"name": "Brian Johnson", "email": "b.johnson@acmefund.com", "dept": "Technology"},
    {"name": "Jessica Taylor", "email": "j.taylor@acmefund.com", "dept": "Technology"},
]


# ============================================================================
# Email Templates - With REALISTIC NOISE and SUBTLE VIOLATIONS
# ============================================================================

TEMPLATES = {
    "CLEAN": [
        # Standard mundane emails
        {
            "subject": "Q4 Portfolio Review Meeting",
            "body": "Hi team,\n\nJust a reminder about our Q4 portfolio review meeting scheduled for Thursday at 2pm. Please come prepared with your sector updates.\n\nThanks,\n{sender_name}",
            "reasoning": "This is a routine business communication about an internal meeting with no compliance concerns.",
        },
        {
            "subject": "Updated compliance training schedule",
            "body": "All,\n\nThe annual compliance training has been rescheduled to next Monday. Please block your calendars from 10am-12pm.\n\nBest,\n{sender_name}",
            "reasoning": "This is a standard administrative message about compliance training scheduling.",
        },
        {
            "subject": "Office closure reminder - Holiday",
            "body": "Team,\n\nJust a friendly reminder that the office will be closed next Friday for the holiday. Please plan accordingly.\n\nRegards,\n{sender_name}",
            "reasoning": "This is a routine administrative notification about office closure.",
        },
        {
            "subject": "RE: Lunch order for client meeting",
            "body": "Sounds good. I'll order from the usual place. Can you confirm headcount by noon?\n\n{sender_name}",
            "reasoning": "This is a routine logistical message about meeting arrangements.",
        },
        {
            "subject": "Monthly risk report - draft for review",
            "body": "Hi {recipient_name},\n\nAttached is the draft monthly risk report. Please review and send any comments by EOD Wednesday.\n\nThanks,\n{sender_name}",
            "reasoning": "This is a normal internal workflow message about reviewing standard reports.",
        },
        # NOISY CLEAN - Sound risky but are legitimate
        {
            "subject": "RE: Upcoming announcement discussion",
            "body": "Hi {recipient_name},\n\nRegarding the announcement - yes, I've prepared the talking points for after it goes public. Communications team has everything ready. Happy to walk through the media strategy once it's out.\n\n{sender_name}",
            "reasoning": "This discusses preparing for a public announcement but only involves post-announcement communications strategy, which is legitimate.",
        },
        {
            "subject": "Confidential project update",
            "body": "Team,\n\nJust a reminder that our Q1 strategic initiative is still confidential until the board meeting next week. Please continue to use the secure project channel for all related discussions. I've updated the timeline in SharePoint.\n\nThanks,\n{sender_name}",
            "reasoning": "This reminds team about confidentiality protocols, which is proper compliance behavior.",
        },
        {
            "subject": "Client meeting prep - sensitive topics",
            "body": "{recipient_name},\n\nFor tomorrow's meeting with the Pension Board, I've prepared the performance materials as requested. The numbers are strong this quarter. Let me know if you need any additional slides on risk metrics before I send.\n\nBest,\n{sender_name}",
            "reasoning": "This discusses client materials preparation for an official meeting, which is standard client relations work.",
        },
        {
            "subject": "Pre-trade analysis complete",
            "body": "Hi {recipient_name},\n\nI finished the pre-trade analysis you requested on the semiconductor sector. Based on publicly available data and our models, I think there's an opportunity worth exploring. Happy to discuss.\n\n{sender_name}",
            "reasoning": "This discusses trading analysis based on public data and internal models, which is legitimate research activity.",
        },
        {
            "subject": "Research report - please hold until Tuesday",
            "body": "Team,\n\nThe TechCorp research report is in final review. Publication is scheduled for Tuesday morning as planned. Please ensure all quotes have been verified with IR before we go live.\n\nThanks,\n{sender_name}",
            "reasoning": "This discusses proper research publication workflow with appropriate timing coordination.",
        },
        {
            "subject": "M&A deal update - public timeline",
            "body": "All,\n\nAs discussed in the press release yesterday, the acquisition is expected to close in Q2. I'm coordinating with Legal on the integration planning materials. Please reach out if you need anything for your workstreams.\n\n{sender_name}",
            "reasoning": "This references publicly announced M&A information and discusses standard integration planning.",
        },
        {
            "subject": "Quick question on the model",
            "body": "Hey {recipient_name},\n\nI was looking at the DCF assumptions for the pharma sector. The revenue growth looks aggressive - can we chat about the methodology? Want to make sure I understand before presenting to the IC.\n\n{sender_name}",
            "reasoning": "This is a legitimate internal discussion about valuation methodology.",
        },
        {
            "subject": "FW: Industry conference takeaways",
            "body": "Hi team,\n\nAttached are my notes from the industry conference. Great discussions with management teams about their public guidance and strategy. Key themes: supply chain improvements and margin expansion focus.\n\nLet me know if you want to discuss any of the companies.\n{sender_name}",
            "reasoning": "This shares publicly discussed information from a conference, which is standard investment research.",
        },
        {
            "subject": "Important - deadline reminder",
            "body": "{recipient_name},\n\nJust a heads up - the filing deadline is Friday. Please make sure your section is complete by Wednesday so Legal can review. This is time-sensitive but we're in good shape.\n\n{sender_name}",
            "reasoning": "This is a standard deadline reminder about regulatory filing, which is proper compliance behavior.",
        },
        {
            "subject": "RE: Position sizing discussion",
            "body": "Good points. I agree we should be thoughtful about sizing here given the volatility. Let's discuss risk parameters with the PM before making any changes. I've blocked time tomorrow at 3pm.\n\n{sender_name}",
            "reasoning": "This discusses position sizing through proper channels with portfolio management oversight.",
        },
        {
            "subject": "Analyst call transcript available",
            "body": "Team,\n\nThe earnings call transcript from yesterday is now available in the usual location. Key takeaways: management maintained guidance, buyback authorization extended. Happy to discuss implications for our thesis.\n\n{sender_name}",
            "reasoning": "This discusses publicly available earnings call information.",
        },
        {
            "subject": "SEC filing review needed",
            "body": "Hi {recipient_name},\n\nCan you review the 10-Q before we file? I flagged a few items that need your sign-off. Nothing concerning, just want to make sure we're buttoned up.\n\nThanks,\n{sender_name}",
            "reasoning": "This is standard internal review process for regulatory filings.",
        },
        {
            "subject": "RE: Investment committee prep",
            "body": "Thanks for sending the deck. I reviewed it and have a few comments on the valuation section. Otherwise looks good - I think we're ready for Thursday's IC. Let's sync tomorrow morning.\n\n{sender_name}",
            "reasoning": "This discusses internal investment committee preparation, which is standard process.",
        },
        {
            "subject": "Client portfolio rebalancing",
            "body": "{recipient_name},\n\nPer our conversation with the Henderson Foundation last week, I've prepared the rebalancing proposal. They approved the strategy shift toward fixed income. Can you review the implementation plan?\n\nBest,\n{sender_name}",
            "reasoning": "This discusses client-approved portfolio changes through proper channels.",
        },
        {
            "subject": "Market volatility update",
            "body": "Team,\n\nGiven today's market moves, I wanted to share a quick update on our positioning. We're within risk limits and the hedges are performing as expected. No action needed but happy to discuss if anyone has questions.\n\n{sender_name}",
            "reasoning": "This provides legitimate market update on portfolio status.",
        },
        # BORDERLINE CLEAN - Very similar language to violations but legitimate
        {
            "subject": "RE: Acting quickly on this",
            "body": "Hey {recipient_name},\n\nI think we need to move quickly on this opportunity - the public filing indicates the timeline is compressed. Let's get pre-clearance done ASAP and discuss with compliance before we execute anything.\n\n{sender_name}",
            "reasoning": "Uses urgency language but follows proper pre-clearance procedures.",
        },
        {
            "subject": "Between us for now",
            "body": "{recipient_name},\n\nThis is still in planning stages so keeping it internal for now, but we're considering a rebalance next quarter. Will share more broadly once we've finalized with the IC. Just wanted to give you a heads up since it affects your workstream.\n\n{sender_name}",
            "reasoning": "Uses secretive language but for legitimate internal coordination purposes.",
        },
        {
            "subject": "Tip from industry contact",
            "body": "Got a good tip from someone at the conference - they recommended we look at the company's public investor day presentation more closely. Some interesting strategic comments buried in there that we missed.\n\n{sender_name}",
            "reasoning": "References a 'tip' but explicitly refers to public information.",
        },
        {
            "subject": "Non-public info handling reminder",
            "body": "Team,\n\nReminder that any non-public information received from company meetings needs to go through compliance before we can act on it. If you get any MNPI, please wall yourself off and contact Legal immediately.\n\nThanks,\n{sender_name}",
            "reasoning": "Discusses MNPI but in context of proper compliance procedures.",
        },
        {
            "subject": "RE: Trading idea - wait for publication",
            "body": "{recipient_name},\n\nI like the thesis but let's wait for the research report to publish Tuesday before we execute. Don't want to get ahead of our own research. Happy to size it up after the official release.\n\n{sender_name}",
            "reasoning": "Discusses timing trades around research but to AVOID front-running.",
        },
        {
            "subject": "Confidential client data - proper handling",
            "body": "All,\n\nI'm compiling client portfolio data for the quarterly review. As a reminder, this information is confidential and should only be shared through approved channels. Please use the secure portal for any submissions.\n\nThanks,\n{sender_name}",
            "reasoning": "Discusses confidential data but emphasizes proper handling.",
        },
        {
            "subject": "RE: My portfolio thoughts",
            "body": "Good discussion! Here's how I'm thinking about positioning in my personal account - submitted the pre-clearance form yesterday and should hear back today. Want to make sure I'm fully compliant before doing anything.\n\n{sender_name}",
            "reasoning": "Discusses personal trading but follows proper pre-clearance.",
        },
        {
            "subject": "Sensitive information from meeting",
            "body": "Heads up - got some interesting color from the management meeting today. Going to work with compliance to determine what we can use vs. what needs to be walled off. Will circle back once we've cleared it.\n\n{sender_name}",
            "reasoning": "References sensitive information but proactively involves compliance.",
        },
        {
            "subject": "Can't share details yet",
            "body": "{recipient_name},\n\nI know you're eager for the update, but the deal isn't public yet so I can't share specifics. Should be announced next week and I can brief you fully after that. Appreciate your patience.\n\n{sender_name}",
            "reasoning": "Refuses to share non-public information, which is proper behavior.",
        },
        {
            "subject": "Acting before announcement",
            "body": "Team,\n\nWe need to finalize our communications plan before the announcement goes out. Marketing has the press release ready and IR is prepped for calls. Let's meet at 3pm to walk through the day-of timeline.\n\n{sender_name}",
            "reasoning": "Discusses acting before announcement but for communications planning, not trading.",
        },
        {
            "subject": "Research preview - internal only",
            "body": "Hi team,\n\nHere's the draft research note for internal review before publication. Please submit feedback by Thursday so we can incorporate and publish Monday. All comments should go through the usual review process.\n\n{sender_name}",
            "reasoning": "Shares unpublished research but only for legitimate editorial review.",
        },
        {
            "subject": "Stock tip follow-up",
            "body": "{recipient_name},\n\nRe: that stock you mentioned - I did my own research and the publicly available info looks interesting. Want to discuss the thesis? I'd need to run it by compliance before adding to my PA but curious about your view.\n\n{sender_name}",
            "reasoning": "Discusses a 'stock tip' but based on public research and proper procedures.",
        },
        {
            "subject": "Off the record chat",
            "body": "Had a great off-the-record conversation with the management team at the conference. They confirmed what's already in their public guidance - no surprises. Good to get the additional color on their strategy.\n\n{sender_name}",
            "reasoning": "References 'off the record' but explicitly notes only public information discussed.",
        },
        {
            "subject": "Don't share this externally",
            "body": "Team,\n\nAttaching our internal performance attribution for Q3. This is for internal use only - don't share externally, but please use it to prep for the client review meetings next week.\n\nThanks,\n{sender_name}",
            "reasoning": "Requests secrecy but for legitimate internal document handling.",
        },
        {
            "subject": "Buy recommendation - after hours research",
            "body": "{recipient_name},\n\nI stayed late reviewing the public filings and I think we should consider buying more of this name. The Q3 10-Q has some interesting details I want to walk you through. Free for a call tomorrow?\n\n{sender_name}",
            "reasoning": "Makes a buy recommendation based on published public filings.",
        },
        # More standard mundane
        {
            "subject": "Happy Birthday!",
            "body": "Happy Birthday {recipient_name}! Hope you have a great day!\n\n{sender_name}",
            "reasoning": "This is a personal greeting with no business content.",
        },
        {
            "subject": "RE: Software license renewal",
            "body": "I've approved the Bloomberg terminal renewal. Please coordinate with IT for the installation.\n\n{sender_name}",
            "reasoning": "This is a routine software procurement message.",
        },
        {
            "subject": "Building access cards",
            "body": "Hi all,\n\nNew access cards will be distributed next week. Please bring your old card to reception to swap.\n\nThanks,\n{sender_name}",
            "reasoning": "This is a standard facilities management notification.",
        },
        {
            "subject": "Performance review scheduling",
            "body": "Hi {recipient_name},\n\nI'd like to schedule your mid-year performance review. Are you available next Tuesday or Wednesday afternoon?\n\nBest,\n{sender_name}",
            "reasoning": "This is a standard HR-related scheduling message.",
        },
        {
            "subject": "Expense report reminder",
            "body": "Team,\n\nPlease submit any outstanding expense reports by end of month. Finance needs to close the books.\n\nThanks,\n{sender_name}",
            "reasoning": "This is a routine administrative reminder about expense reports.",
        },
    ],
    "INSIDER_TRADING": [
        # Obvious violations
        {
            "subject": "URGENT - Acme Corp news coming",
            "body": "Hey,\n\nHeads up - I heard from my contact at Acme that they're announcing a major acquisition tomorrow morning. This isn't public yet. Might want to adjust positions before the announcement.\n\nKeep this between us.\n{sender_name}",
            "reasoning": "This email contains material non-public information (MNPI) about an upcoming acquisition announcement and suggests trading on this information before it becomes public.",
        },
        {
            "subject": "RE: Tech sector positions",
            "body": "Trust me on this one - load up on TECH before Friday. I can't say more but the earnings will blow expectations away. My source is solid.\n\n{sender_name}",
            "reasoning": "This email suggests trading based on non-public earnings information from an inside source.",
        },
        {
            "subject": "Pharma play - time sensitive",
            "body": "{recipient_name},\n\nFDA approval coming through for BioMed's drug application. Not announced yet but I got word from inside. This could move 30%+. Act fast.\n\n{sender_name}",
            "reasoning": "This email shares non-public information about a pending FDA approval and recommends trading on it.",
        },
        # SUBTLE violations - less obvious language
        {
            "subject": "RE: Sector thoughts",
            "body": "{recipient_name},\n\nHad coffee with a friend in banking yesterday. Without getting into specifics, let's just say the industrial sector might get interesting next week. Take that for what it's worth.\n\n{sender_name}",
            "reasoning": "This subtly hints at receiving non-public M&A information and implies the recipient should act on it.",
        },
        {
            "subject": "Quick thought on RetailCo",
            "body": "I would seriously look at reducing RetailCo exposure before Thursday's earnings. Can't say why but I'd hate for us to be caught off guard. Just my two cents.\n\n{sender_name}",
            "reasoning": "This suggests trading action based on implied non-public earnings information without being explicit.",
        },
        {
            "subject": "RE: Healthcare names",
            "body": "Remember that biotech we discussed? My contact at the company mentioned they're very confident about the Phase 3 readout. Might be worth revisiting the position before next month.\n\n{sender_name}",
            "reasoning": "This shares confidential clinical trial sentiment from an inside contact and suggests trading.",
        },
        {
            "subject": "Timing consideration",
            "body": "Hey {recipient_name},\n\nIf you're still thinking about that energy trade, I'd suggest doing it before Wednesday. Heard some things at dinner last night. Nothing concrete but the timing feels right.\n\n{sender_name}",
            "reasoning": "This implies receiving non-public information and recommends timing trades around it.",
        },
        {
            "subject": "FYI - worth considering",
            "body": "{recipient_name},\n\nSaw something interesting in my inbox today from an IR contact. Can't forward it obviously but let's chat about TechStart when you have a moment. Interesting developments.\n\n{sender_name}",
            "reasoning": "This references receiving material information from investor relations before public release.",
        },
        {
            "subject": "RE: Portfolio positioning",
            "body": "On the MegaCorp position - I would not be adding here. A friend who works there mentioned some 'organizational changes' coming. Read between the lines.\n\n{sender_name}",
            "reasoning": "This shares non-public information about corporate restructuring from an employee contact.",
        },
        {
            "subject": "Good timing",
            "body": "Remember when we talked about that defense contractor? I bumped into someone from the Pentagon at the gym. Sounds like good news is coming for that contract bid. Just saying.\n\n{sender_name}",
            "reasoning": "This shares non-public government contract information from a Pentagon contact.",
        },
        {
            "subject": "Interesting data point",
            "body": "My neighbor works at ConsumerCo corporate. She mentioned they've been working crazy hours on 'something big' but wouldn't say more. Might be worth keeping an eye on.\n\n{sender_name}",
            "reasoning": "This shares information about potential corporate announcements from an inside contact.",
        },
        {
            "subject": "RE: Market thoughts",
            "body": "{recipient_name},\n\nI know we try to be careful about this stuff, but I learned something at the charity dinner last night about FinanceCo's dividend that you'll want to know. Call me.\n\n{sender_name}",
            "reasoning": "This indicates receiving material non-public dividend information from a social contact.",
        },
        {
            "subject": "Quiet heads up",
            "body": "Between us - got a text from my old colleague at the law firm. The SoftwareCo deal is happening, just finalizing terms. Market doesn't know yet. Do what you will with that.\n\n{sender_name}",
            "reasoning": "This shares confidential M&A information from a legal professional.",
        },
        {
            "subject": "RE: Semiconductor positioning",
            "body": "Was chatting with a supplier rep for ChipMaker yesterday. Their order book sounds way better than the street expects. Might be worth getting ahead of that print.\n\n{sender_name}",
            "reasoning": "This shares non-public supply chain information that could predict earnings.",
        },
        {
            "subject": "Food for thought",
            "body": "{recipient_name},\n\nMy brother-in-law is on the board at MedDevice. He let slip that the FDA meeting went really well. Obviously don't tell anyone where you heard this.\n\n{sender_name}",
            "reasoning": "This shares confidential regulatory information from a board member family connection.",
        },
    ],
    "CONFIDENTIALITY_BREACH": [
        # Obvious violations
        {
            "subject": "FW: Client portfolio details",
            "body": "Hey,\n\nForwarding you the Westbrook pension fund's full portfolio breakdown. They're way overexposed to energy. Don't tell anyone I sent this.\n\n{sender_name}",
            "reasoning": "This email shares confidential client portfolio information with an unauthorized recipient and explicitly requests secrecy.",
        },
        {
            "subject": "RE: Hedge fund strategies",
            "body": "Here's our fund's full trading strategy doc. I know you're at a competitor now but this might help you. Delete after reading.\n\n{sender_name}",
            "reasoning": "This email shares proprietary trading strategies with someone at a competitor firm.",
        },
        # SUBTLE violations
        {
            "subject": "RE: Catching up",
            "body": "Great to hear from you! Things are going well here. Between us, our biggest client just told us they're planning a major allocation shift next quarter. Could affect the market if it gets out. Anyway, let's grab lunch next week.\n\n{sender_name}",
            "reasoning": "This casually shares confidential client allocation plans in a personal email.",
        },
        {
            "subject": "FW: Interesting read",
            "body": "{recipient_name},\n\nThought you'd find this interesting - it's the risk assessment we did for the Thompson Family Office. Obviously keep it to yourself but some good insights on concentration risk.\n\n{sender_name}",
            "reasoning": "This forwards confidential client risk assessment documents.",
        },
        {
            "subject": "RE: Your question",
            "body": "You asked about how we structure fees - here's what we're charging the university endowment. It's our most favorable arrangement, so probably don't want this circulating. But figured it could help with your negotiation.\n\n{sender_name}",
            "reasoning": "This shares confidential fee arrangements with specific clients.",
        },
        {
            "subject": "Info you requested",
            "body": "{recipient_name},\n\nRemember you were curious about what other funds were doing? I pulled some data from our client meetings. Most are reducing equity exposure heading into next year. Useful context for you but please don't share the source.\n\n{sender_name}",
            "reasoning": "This shares aggregated confidential client positioning information.",
        },
        {
            "subject": "RE: Interview prep",
            "body": "For your meeting tomorrow - here's some background on your prospective employer. It's from when we pitched them last year, so it has their whole portfolio breakdown and risk tolerance. Should help you prepare.\n\n{sender_name}",
            "reasoning": "This shares confidential client due diligence materials for personal benefit.",
        },
        {
            "subject": "Quick context",
            "body": "Heads up before your call - I know you're talking to them about a potential mandate. Here's what their current allocation looks like based on our records. Should give you some ammunition.\n\n{sender_name}",
            "reasoning": "This shares confidential client portfolio information to help with competitive bidding.",
        },
        {
            "subject": "FW: Meeting notes - helpful context",
            "body": "{recipient_name},\n\nAttaching notes from our LP advisory board meeting. Some candid feedback about manager selection that might be useful for your job search. Goes without saying to keep this confidential.\n\n{sender_name}",
            "reasoning": "This forwards confidential LP meeting notes for personal use.",
        },
        {
            "subject": "Background material",
            "body": "For your due diligence - here's our performance attribution for the past 3 years at the position level. I know you're evaluating other funds too so this gives you a comparison point. Just don't let anyone know where you got it.\n\n{sender_name}",
            "reasoning": "This shares proprietary performance and positioning data with external parties.",
        },
        {
            "subject": "RE: Market intel",
            "body": "You might find this useful - compiled notes from all our client conversations this month. Good sense of where institutional money is flowing. Obviously treat this as confidential.\n\n{sender_name}",
            "reasoning": "This shares aggregated confidential client communication summaries.",
        },
        {
            "subject": "Useful reference",
            "body": "{recipient_name},\n\nRemember that compliance issue we had last year? Attaching the internal investigation report since you asked. It has some lessons learned that might be relevant for you. Please don't forward.\n\n{sender_name}",
            "reasoning": "This shares confidential internal investigation materials externally.",
        },
        {
            "subject": "RE: Benchmarking",
            "body": "Here's what I can share - our compensation data for the research team by level. I know you're putting together a market study so this should help with the analysis. Keep my name out of it.\n\n{sender_name}",
            "reasoning": "This shares confidential internal compensation information externally.",
        },
        {
            "subject": "Helpful info",
            "body": "For your business development efforts - I put together a list of our LPs with contact info and last known allocation sizes. Obviously this is sensitive so please be discreet about the source.\n\n{sender_name}",
            "reasoning": "This shares confidential investor lists and allocation data.",
        },
        {
            "subject": "RE: Side project",
            "body": "Happy to help with your research paper. Here's the data on our trading patterns over the past year - anonymized the client names but kept the flows. Should make for interesting analysis.\n\n{sender_name}",
            "reasoning": "This shares proprietary trading flow data for external research.",
        },
    ],
    "PERSONAL_TRADING": [
        # Obvious violations
        {
            "subject": "My stock picks",
            "body": "Hey,\n\nI bought a bunch of shares in TechStartup Inc last week in my personal account. Looking good so far! You should get in before it pops.\n\n{sender_name}",
            "reasoning": "This email discusses undisclosed personal trading and recommends the same trade to others.",
        },
        {
            "subject": "RE: Investment advice",
            "body": "For my PA, I'm going heavy into crypto this month. Already moved $50k. Don't mention this to compliance - it's under their radar.\n\n{sender_name}",
            "reasoning": "This email discusses unreported personal trading and explicitly mentions avoiding compliance reporting.",
        },
        # SUBTLE violations
        {
            "subject": "RE: Market discussion",
            "body": "{recipient_name},\n\nGood call on that stock we discussed. I may have picked up a few shares in my IRA - nothing major, just wanted some exposure. Didn't seem worth bothering with the pre-clearance form for such a small amount.\n\n{sender_name}",
            "reasoning": "This admits to personal trading without required pre-clearance, downplaying the violation.",
        },
        {
            "subject": "Quick question for you",
            "body": "Hey {recipient_name},\n\nMy financial advisor is recommending some changes to my portfolio. A few names overlap with what we cover. I assume that's fine since it's through an advisor and not my direct decision?\n\n{sender_name}",
            "reasoning": "This discusses trading in covered names through an advisor to avoid disclosure, which is still a violation.",
        },
        {
            "subject": "RE: Weekend plans",
            "body": "Nothing exciting - just been spending some time on my Robinhood account. Made a few trades on Friday. I should probably update my disclosure but it's just small stuff in tech names.\n\n{sender_name}",
            "reasoning": "This casually mentions undisclosed personal trading activity.",
        },
        {
            "subject": "Friendly advice",
            "body": "{recipient_name},\n\nMy sister asked for investment ideas - I pointed her toward that energy company we've been researching. Figured since it's her account, no disclosure needed on my end. She's pretty excited about it.\n\n{sender_name}",
            "reasoning": "This discusses directing family trades to avoid personal disclosure requirements.",
        },
        {
            "subject": "RE: Personal finance chat",
            "body": "Yeah, I've been building a position in some of the small caps we follow. It's in my 401k so I assumed that's exempt from reporting? Either way, it's been working out well.\n\n{sender_name}",
            "reasoning": "This discusses undisclosed trading in covered securities with a false assumption about exemptions.",
        },
        {
            "subject": "Following up",
            "body": "Remember that SPAC we talked about? I bought some warrants in my TD account. I know options need pre-clearance but warrants are different, right? Anyway, up 40% already.\n\n{sender_name}",
            "reasoning": "This discusses trading in derivatives without proper clearance based on a mistaken interpretation.",
        },
        {
            "subject": "RE: Coffee tomorrow?",
            "body": "Sure thing. By the way, I finally pulled the trigger on that biotech position we've been watching. Used my joint account with my wife - figured that simplifies the paperwork situation.\n\n{sender_name}",
            "reasoning": "This discusses using a joint account to complicate disclosure requirements.",
        },
        {
            "subject": "Quick thought",
            "body": "{recipient_name},\n\nSaw a great opportunity in Asian markets over the weekend. Made a few trades in my Interactive Brokers account. Since it's international and after hours, I'm assuming it doesn't fall under our policy?\n\n{sender_name}",
            "reasoning": "This discusses undisclosed international trading with false assumptions about policy coverage.",
        },
        {
            "subject": "RE: Investment club",
            "body": "The club I'm in made some trades last month - including a few names that might be on our restricted list. But since it's a club decision and not just me, I think we're fine from a disclosure standpoint.\n\n{sender_name}",
            "reasoning": "This discusses trading restricted securities through an investment club to avoid direct attribution.",
        },
        {
            "subject": "Update",
            "body": "Finally moved some money into that ETF we discussed. I know it has some individual holdings that might need disclosure, but since it's an ETF and not direct stock ownership, I figured it's cleaner.\n\n{sender_name}",
            "reasoning": "This discusses avoiding single-stock disclosure through ETF purchases of the same names.",
        },
        {
            "subject": "RE: Market thoughts",
            "body": "{recipient_name},\n\nI've been testing a new trading strategy in my personal account - mostly short-term stuff. It's been time-consuming to pre-clear every trade so I've just been keeping positions under a week.\n\n{sender_name}",
            "reasoning": "This admits to avoiding pre-clearance by keeping holding periods short.",
        },
        {
            "subject": "Good news",
            "body": "That trade we discussed worked out great - made a nice profit in my Schwab account. I used limit orders over a few days to keep it under the reporting threshold. Smart, right?\n\n{sender_name}",
            "reasoning": "This discusses structuring trades to avoid reporting thresholds.",
        },
        {
            "subject": "RE: Catching up",
            "body": "Things are good! Been doing some option spreads on the side - nothing crazy, just generating some income. I'm not reporting them since they're hedged and have limited risk.\n\n{sender_name}",
            "reasoning": "This discusses unreported options trading with a false rationalization about hedged positions.",
        },
    ],
    "INFO_BARRIER_VIOLATION": [
        # Obvious violations
        {
            "subject": "Research update - please read",
            "body": "James,\n\nI know we're not supposed to share this directly, but my team just finished our analysis on Quantum Industries. Rating it a strong buy. Wanted you to have a heads up before it goes through official channels.\n\n- Sarah (Research)",
            "reasoning": "This email shares unpublished research ratings directly with trading personnel, violating the information barrier between research and trading.",
        },
        {
            "subject": "RE: Position sizing question",
            "body": "Between us - research is about to downgrade SolarTech to sell. I'd reduce exposure before the report comes out next week. Don't tell anyone I told you.\n\n{sender_name}",
            "reasoning": "This email shares upcoming research rating changes with trading before publication, violating the Chinese wall.",
        },
        # SUBTLE violations
        {
            "subject": "RE: Sector discussion",
            "body": "{recipient_name},\n\nGood chat earlier. Just to follow up on the healthcare point - without getting into specifics, I'd say our research team's view on the sector has evolved recently. Worth factoring into your positioning decisions.\n\n{sender_name}",
            "reasoning": "This subtly communicates research team sentiment changes to trading without explicit details.",
        },
        {
            "subject": "Timing heads up",
            "body": "Hey {recipient_name},\n\nI noticed you're working on that energy trade. Just wanted to mention that some research is being finalized this week that might be relevant. Can't say more but thought you'd want to know the timing.\n\n{sender_name}",
            "reasoning": "This alerts trading to pending research publication timing without sharing content.",
        },
        {
            "subject": "RE: Quick question",
            "body": "On the semiconductor question - let's just say the research team has been doing a lot of work on ChipCo and the conclusions might surprise people. Publication is next Monday. Draw your own conclusions.\n\n{sender_name}",
            "reasoning": "This implies research conclusions and timing without explicit sharing.",
        },
        {
            "subject": "FYI - interesting development",
            "body": "{recipient_name},\n\nI was in a meeting where some interesting analysis came up about IndustrialCo. Can't share the specifics but the thesis is changing. Might want to think about that.\n\n{sender_name}",
            "reasoning": "This communicates research view changes through vague hints.",
        },
        {
            "subject": "RE: Market views",
            "body": "You asked about my thoughts on the consumer sector. Honestly, I've seen some of the work being done on retail names internally. Let's just say I'd be cautious here until things get published.\n\n{sender_name}",
            "reasoning": "This references unpublished internal research to guide trading decisions.",
        },
        {
            "subject": "Lunch follow-up",
            "body": "Great lunch yesterday. Regarding what we discussed about TechStart - I mentioned it to one of our analysts afterward. His reaction was... let's say interesting. Might be worth waiting a bit before doing anything.\n\n{sender_name}",
            "reasoning": "This communicates analyst sentiment on a specific name to influence trading timing.",
        },
        {
            "subject": "RE: Model question",
            "body": "{recipient_name},\n\nI can't share the actual research output, but I'll say the team's work on price targets in the banking sector would probably inform your view. Things have changed from where we were last quarter.\n\n{sender_name}",
            "reasoning": "This hints at upcoming price target changes without sharing explicit numbers.",
        },
        {
            "subject": "Helpful context",
            "body": "Before you finalize that trade - I sat in on the research meeting this morning. Not at liberty to share details but the discussion on that name was... spirited. Food for thought.\n\n{sender_name}",
            "reasoning": "This communicates research meeting discussions to trading personnel.",
        },
        {
            "subject": "RE: Trading idea",
            "body": "Interesting idea. I happened to chat with our analyst who covers that space. Without betraying any confidences, let's just say your timing might be either really good or really bad depending on the next week's publications.\n\n{sender_name}",
            "reasoning": "This implies analyst views and publication timing to influence trade timing.",
        },
        {
            "subject": "Coffee chat follow-up",
            "body": "{recipient_name},\n\nGood to catch up yesterday. On the biotech discussion - I mentioned I'd try to find out more. Turns out our healthcare analyst has some strong opinions on that FDA decision. Can't elaborate but thought you'd want to know.\n\n{sender_name}",
            "reasoning": "This actively seeks and shares research analyst views across the barrier.",
        },
        {
            "subject": "RE: Portfolio thoughts",
            "body": "You mentioned being bullish on energy. I don't want to influence your view, but I'd just say the research team's sentiment has shifted meaningfully. They're finalizing something now.\n\n{sender_name}",
            "reasoning": "This communicates research team sentiment changes to inform portfolio decisions.",
        },
        {
            "subject": "Interesting observation",
            "body": "I noticed you've been building a position in MedTech. Not to be cryptic, but I have visibility into some analysis that might be relevant. Let's grab coffee and I can at least point you in the right direction.\n\n{sender_name}",
            "reasoning": "This offers to share research insights on a specific position across the barrier.",
        },
        {
            "subject": "RE: Quick sync",
            "body": "{recipient_name},\n\nPer our call - yes, there's research activity on the names you mentioned. I can't share specifics but let's say the market might be surprised when it comes out. Take that how you will.\n\n{sender_name}",
            "reasoning": "This confirms research activity and implies market-moving conclusions.",
        },
    ],
}


# ============================================================================
# Reasoning templates for fine-tuning
# ============================================================================

REASONING_BY_LABEL = {
    "CLEAN": [
        "This is a routine business communication with no compliance concerns.",
        "This is a standard administrative message with no sensitive information.",
        "This is normal internal workflow with no indication of policy violations.",
        "This is a personal/social message with no compliance implications.",
        "This discusses publicly available information only.",
    ],
    "INSIDER_TRADING": [
        "This email contains material non-public information (MNPI) and suggests trading on it.",
        "This email shares non-public corporate information from an inside source.",
        "This email recommends trading based on information not yet publicly disclosed.",
        "This email contains confidential information about upcoming announcements and implies trading action.",
        "This email shares insider knowledge and explicitly suggests positioning before public disclosure.",
    ],
    "CONFIDENTIALITY_BREACH": [
        "This email shares confidential client information with unauthorized recipients.",
        "This email discloses proprietary fund information that should remain confidential.",
        "This email forwards sensitive internal documents without authorization.",
        "This email shares confidential performance or fee data inappropriately.",
        "This email reveals private client details or communications.",
    ],
    "PERSONAL_TRADING": [
        "This email discusses undisclosed personal trading activity.",
        "This email reveals attempts to circumvent trading compliance requirements.",
        "This email discusses personal trades that should require pre-clearance.",
        "This email shows intent to avoid or bypass disclosure requirements.",
        "This email discusses using alternative accounts to hide trading activity.",
    ],
    "INFO_BARRIER_VIOLATION": [
        "This email shares unpublished research information across the information barrier.",
        "This email violates the Chinese wall between research and trading.",
        "This email provides advance notice of research publications to trading personnel.",
        "This email shares research opinions or ratings before official publication.",
        "This email crosses information barriers by sharing non-public research insights.",
    ],
}


# ============================================================================
# Generator Functions
# ============================================================================


def random_timestamp(days_back: int = 180) -> datetime:
    """Generate a random timestamp within business hours (mostly)."""
    base = datetime.now() - timedelta(days=random.randint(1, days_back))
    # 80% during business hours, 20% after hours (suspicious pattern)
    if random.random() < 0.8:
        hour = random.randint(8, 18)
    else:
        hour = random.choice([6, 7, 19, 20, 21, 22, 23])
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


def generate_finetune_sample(label: str) -> dict:
    """Generate a single fine-tuning training sample."""
    template = random.choice(TEMPLATES[label])
    
    sender_name = random.choice(["Alex", "Jordan", "Sam", "Chris", "Morgan", "Taylor"])
    recipient_name = random.choice(["Pat", "Casey", "Riley", "Quinn", "Drew", "Blake"])
    
    body = template["body"].format(
        sender_name=sender_name,
        recipient_name=recipient_name,
    )
    
    # Get the reasoning from template or generate one
    if "reasoning" in template:
        reasoning = template["reasoning"]
    else:
        reasoning = random.choice(REASONING_BY_LABEL[label])
    
    prompt = f"Classify this hedge fund email for compliance violations.\n\nEmail: {body}\n\nClassification:"
    completion = f"{label} - {reasoning}"
    
    return {
        "prompt": prompt,
        "completion": completion,
    }


def generate_email_dataset(num_emails: int, label_noise_rate: float = 0.08) -> list[dict]:
    """Generate the full email dataset with proper label distribution.
    
    Args:
        num_emails: Number of emails to generate
        label_noise_rate: Fraction of labels to randomly flip (simulates real-world labeling errors)
    """
    emails = []
    labels = list(LABEL_WEIGHTS.keys())
    weights = list(LABEL_WEIGHTS.values())
    violation_labels = [l for l in labels if l != "CLEAN"]

    for _ in range(num_emails):
        label = random.choices(labels, weights=weights, k=1)[0]
        email = generate_email(label)
        
        # Add label noise - flip some labels to simulate real-world labeling errors
        if random.random() < label_noise_rate:
            if label == "CLEAN":
                # Some clean emails might get mislabeled as violations
                email["compliance_label"] = random.choice(violation_labels)
            else:
                # Some violations might get mislabeled as clean
                email["compliance_label"] = "CLEAN"
        
        emails.append(email)

    # Sort by timestamp for realism
    emails.sort(key=lambda x: x["sent_at"])
    return emails


def generate_finetune_dataset() -> list[dict]:
    """Generate the fine-tuning dataset with balanced distribution."""
    samples = []
    
    for label, count in FINETUNE_DISTRIBUTION.items():
        for _ in range(count):
            samples.append(generate_finetune_sample(label))
    
    # Shuffle the samples
    random.shuffle(samples)
    return samples


def main():
    """Generate and save both datasets."""
    print("=" * 60)
    print("Generating Synthetic Email Datasets")
    print("=" * 60)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate email dataset
    print(f"\n1. Generating {NUM_EMAILS:,} synthetic emails...")
    emails = generate_email_dataset(NUM_EMAILS)

    # Write emails CSV
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

    with open(EMAILS_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(emails)

    # Print email summary
    label_counts = {}
    for email in emails:
        label = email["compliance_label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"   Saved to: {EMAILS_OUTPUT}")
    print("\n   Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / len(emails) * 100
        print(f"     {label}: {count:,} ({pct:.1f}%)")

    # Generate fine-tuning dataset
    print(f"\n2. Generating {NUM_FINETUNE_SAMPLES} fine-tuning samples...")
    finetune_samples = generate_finetune_dataset()
    
    # Write fine-tuning JSONL
    with open(FINETUNE_OUTPUT, "w", encoding="utf-8") as f:
        for sample in finetune_samples:
            f.write(json.dumps(sample) + "\n")
    
    # Print fine-tuning summary
    ft_label_counts = {}
    for sample in finetune_samples:
        label = sample["completion"].split(" - ")[0]
        ft_label_counts[label] = ft_label_counts.get(label, 0) + 1
    
    print(f"   Saved to: {FINETUNE_OUTPUT}")
    print("\n   Label distribution:")
    for label, count in sorted(ft_label_counts.items(), key=lambda x: -x[1]):
        print(f"     {label}: {count}")
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
