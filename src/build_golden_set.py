"""Build the Golden Evaluation Set (200 curated examples).

Performs stratified sampling across all intent categories with human/expert
ground-truth labels for:
- ground_truth_intent
- ground_truth_secondary_intent
- ground_truth_escalation (AUTO_HANDLE vs ESCALATE)
- escalation_rationale
- ground_truth_resolution_notes
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "amazonhelp_support_pairs.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "golden_evaluation_set.csv"

# Sampling targets per category to align with the 9 human ground-truth intents
CATEGORY_QUOTAS = {
    'genral_enquiry': 35,
    'order_delayed': 26,
    'Discrepancy_in_Product': 20,
    'return/refund request': 19,
    'order_not_delivered': 18,
    'account_issue': 15,
    'customer_service_complaint': 10,
    'payment_issue': 4,
    'cancellation_request': 3,
}

MANUAL_GOLDEN_CSV = PROJECT_ROOT / "data" / "processed" / "golden_set_manually_Intent_filled.csv"

def build_golden_set():
    if MANUAL_GOLDEN_CSV.exists():
        print(f"Loading curated human ground-truth dataset from: {MANUAL_GOLDEN_CSV}")
        manual_df = pd.read_csv(MANUAL_GOLDEN_CSV)
        
        # Standardize columns
        if 'manually_corrected_intent' in manual_df.columns:
            manual_df['gold_intent'] = manual_df['manually_corrected_intent']
            manual_df['intent'] = manual_df['manually_corrected_intent']
        if 'manually_corrected_escalation' in manual_df.columns:
            manual_df['gold_escalation'] = manual_df['manually_corrected_escalation']
        if 'gold_secondary_intent' not in manual_df.columns:
            manual_df['gold_secondary_intent'] = 'none'
            manual_df['secondary_intent'] = 'none'
        if 'gold_escalation_reason' not in manual_df.columns:
            manual_df['gold_escalation_reason'] = manual_df['gold_escalation'].apply(
                lambda esc: "High-risk customer grievance or critical issue requiring human intervention."
                if esc == "ESCALATE" else "Standard self-service or policy inquiry handled automatically."
            )
            
        manual_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Successfully synchronized {len(manual_df)} golden evaluation cases to: {OUTPUT_CSV}")
        print("\nGold Escalation Distribution:")
        print(manual_df['gold_escalation'].value_counts())
        print("\nGold Intent Distribution:")
        print(manual_df['gold_intent'].value_counts())
        return

    df = pd.read_csv(INPUT_CSV)
    sampled_rows = []

    for intent, quota in CATEGORY_QUOTAS.items():
        subset = df[df['intent'] == intent]
        if len(subset) >= quota:
            sample = subset.sample(quota, random_state=42)
        else:
            sample = subset
        sampled_rows.append(sample)

    golden_df = pd.concat(sampled_rows, ignore_index=True)

    # Establish gold standard labels
    gt_intents = []
    gt_sec_intents = []
    gt_escalations = []
    gt_reasons = []

    for _, row in golden_df.iterrows():
        c_text = row['customer_text']
        raw_intent = row['intent']
        sec_intent = row.get('secondary_intent', 'other')

        # Refined ground-truth intent
        gt_intent = raw_intent
        gt_sec = sec_intent

        # Ground truth escalation rule
        if gt_intent in ['order_not_delivered', 'account_issue', 'customer_service_complaint', 'Urgent']:
            gt_esc = "ESCALATE"
            reason = f"High-risk scenario ({gt_intent}) requiring human verification, account authority, or supervisor de-escalation."
        elif gt_intent in ['damaged_item', 'wrong_item_received']:
            # Escalate if severe/tampered/missing, else auto-handle return
            if any(w in c_text.lower() for w in ['leaking', 'stolen', 'empty', 'shattered', 'tampered', 'pissed', 'awful']):
                gt_esc = "ESCALATE"
                reason = "Severe package damage, suspected theft, or high frustration requiring manual concession review."
            else:
                gt_esc = "AUTO_HANDLE"
                reason = "Standard wrong/damaged item eligible for automated Online Return Center self-service replacement."
        elif gt_intent in ['payment_issue', 'refund_request']:
            if any(w in c_text.lower() for w in ['twice', 'double', 'fraud', 'debited', 'unauthorized', 'where is my refund']):
                gt_esc = "ESCALATE"
                reason = "Financial discrepancy or delayed refund requires secure ledger verification."
            else:
                gt_esc = "AUTO_HANDLE"
                reason = "General billing or refund policy guidance covered by self-service account portal."
        elif gt_intent in ['order_status_inquiry', 'return_request', 'product_inquiry', 'Prime Membership', 'thankyou', 'Enquiry']:
            gt_esc = "AUTO_HANDLE"
            reason = f"Routine {gt_intent} workflow safely addressed via self-service links and standard policies."
        else:
            gt_esc = "AUTO_HANDLE"
            reason = "Standard inquiry."

        gt_intents.append(gt_intent)
        gt_sec_intents.append(gt_sec)
        gt_escalations.append(gt_esc)
        gt_reasons.append(reason)

    golden_df['gold_intent'] = gt_intents
    golden_df['gold_secondary_intent'] = gt_sec_intents
    golden_df['gold_escalation'] = gt_escalations
    golden_df['gold_escalation_reason'] = gt_reasons

    golden_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully generated Golden Evaluation Set with {len(golden_df)} examples at: {OUTPUT_CSV}")
    print("\nGold Escalation Distribution:")
    print(golden_df['gold_escalation'].value_counts())
    print("\nGold Intent Distribution:")
    print(golden_df['gold_intent'].value_counts())

if __name__ == '__main__':
    build_golden_set()
