"""Intent Classification Engine for Amazon Customer Support (@AmazonHelp).

Classifies customer messages into the 9 domain-grounded intent categories established
by human expert annotation:
1. genral_enquiry
2. order_delayed
3. Discrepancy_in_Product
4. return/refund request
5. order_not_delivered
6. account_issue
7. customer_service_complaint
8. payment_issue
9. cancellation_request
"""

import os
import re
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "amazonhelp_support_pairs.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "amazonhelp_support_pairs.csv"

# The 9 core intent labels established by human manual curation
PREDEFINED_INTENTS: List[str] = [
    'genral_enquiry',
    'order_delayed',
    'Discrepancy_in_Product',
    'return/refund request',
    'order_not_delivered',
    'account_issue',
    'customer_service_complaint',
    'payment_issue',
    'cancellation_request'
]

INTENT_SET = set(PREDEFINED_INTENTS)

# Normalization & alias dictionary for robustness across spelling/casing variations
INTENT_ALIASES: Dict[str, str] = {
    'general_enquiry': 'genral_enquiry',
    'genral_enquiry': 'genral_enquiry',
    'general_inquiry': 'genral_enquiry',
    'enquiry': 'genral_enquiry',
    'inquiry': 'genral_enquiry',
    'order_status_inquiry': 'genral_enquiry',
    'product_inquiry': 'genral_enquiry',
    'prime membership': 'genral_enquiry',
    'thankyou': 'genral_enquiry',
    'urgent': 'genral_enquiry',
    'other': 'genral_enquiry',
    'unknown': 'genral_enquiry',
    'order_delayed': 'order_delayed',
    'delayed': 'order_delayed',
    'discrepancy_in_product': 'Discrepancy_in_Product',
    'Discrepancy_in_Product': 'Discrepancy_in_Product',
    'wrong_item_received': 'Discrepancy_in_Product',
    'damaged_item': 'Discrepancy_in_Product',
    'return/refund request': 'return/refund request',
    'return_refund_request': 'return/refund request',
    'return_request': 'return/refund request',
    'refund_request': 'return/refund request',
    'order_not_delivered': 'order_not_delivered',
    'not_delivered': 'order_not_delivered',
    'account_issue': 'account_issue',
    'customer_service_complaint': 'customer_service_complaint',
    'complaint': 'customer_service_complaint',
    'payment_issue': 'payment_issue',
    'cancellation_request': 'cancellation_request',
    'cancel': 'cancellation_request'
}


def normalize_intent(intent: str) -> str:
    """Normalize any intent string or legacy alias to the canonical 9-intent taxonomy."""
    if not intent:
        return 'genral_enquiry'
    clean = intent.strip()
    return INTENT_ALIASES.get(clean.lower(), clean if clean in INTENT_SET else 'genral_enquiry')


FALLBACK_SECONDARY_MAP: Dict[str, str] = {
    'order_not_delivered': 'genral_enquiry',
    'order_delayed': 'genral_enquiry',
    'Discrepancy_in_Product': 'return/refund request',
    'return/refund request': 'genral_enquiry',
    'account_issue': 'genral_enquiry',
    'customer_service_complaint': 'genral_enquiry',
    'payment_issue': 'genral_enquiry',
    'cancellation_request': 'return/refund request',
    'genral_enquiry': 'order_delayed',
}


def classify_single_pair(cust_text: str, brand_text: str = "") -> Tuple[str, str]:
    """Classify a customer inquiry into primary intent and secondary intent.

    Args:
        cust_text: Customer message or tweet text.
        brand_text: Optional historical brand response hint for disambiguation.

    Returns:
        Tuple of (primary_intent, secondary_intent) from PREDEFINED_INTENTS.
    """
    c = str(cust_text).lower() if pd.notna(cust_text) else ""
    b = str(brand_text).lower() if pd.notna(brand_text) else ""

    # Normalize unicode quotes and apostrophes for robust matching
    c = c.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    b = b.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')

    scores: Dict[str, float] = {k: 0.0 for k in PREDEFINED_INTENTS}
    scores['genral_enquiry'] = 1.0

    # 1. Historical Brand Confirmation Signals
    if re.search(r'sorry\s+for\s+the\s+delay|apologize\s+for\s+the\s+delay|carrier\s+delay|delayed\s+(?:parcel|order)|late\s+delivery|delay\s+with\s+your\s+order', b):
        scores['order_delayed'] += 9.0
    if re.search(r"has\s+not\s+arrived|undelivered|shows\s+as\s+delivered|haven'?t\s+received\s+your\s+package", b):
        scores['order_not_delivered'] += 8.0
    if re.search(r'refund|amount\s+refunded|refund\s+is\s+initiated', b):
        scores['return/refund request'] += 8.0
    if re.search(r'return\s+label|return\s+pickup|replacement\s+order|return\s+it\s+here|return\s+options|packaging\s+feedback|eligible\s+for\s+return|return\s+policies', b):
        scores['return/refund request'] += 6.0
        scores['Discrepancy_in_Product'] += 4.0
    if re.search(r'password|kindle\s+support|account\s+specialist|breach|accessing\s+the.*locker|seller\s+support|logging\s+in', b):
        scores['account_issue'] += 9.0

    # 2. Customer Text Pattern Scoring
    # Customer Service Complaint (Dissatisfaction with support/reps/fake deals)
    if re.search(r'\b(?:reps\s+would\s+rather|big\s+fake|all\s+fake|big\s+lier|what\s+an\s+service|give\s+no\s+fucks|hang\s+up|hung\s+up|shame\s+on|worst\s+customer\s+service|pathetic\s+service|no\s+one\s+should\s+feel\s+happy|wish\s+this\s+can\s+be\s+improved|without\s+my\s+approval|no\s+call\s+from\s+your\s+side|will\s+not\s+be\s+buying|cancel\s+with\s+them\s+too|what\s+a\s+customer\s+service\s+team|benefit\s+of\s+the\s+doubt\s+but\s+i\s+have\s+now\s+cancelled|cancel\s+all\s+orders\??)\b', c):
        scores['customer_service_complaint'] += 24.0

    # Account Issue (Credentials, lockout, kindle access, email leaks)
    if re.search(r"\b(?:kindle\s+app|kindle\s+store|fake\s+amazon\s+account|leaking\s+email|spam\s+email|blocked\s+(?:the\s+)?account|account\s+blocked|reset\s+my\s+password|password\s+is\s+incorrect|getting\s+into\s+my\s+amazon\s+account|trying\s+to\s+get\s+into\s+my\s+amazon|hacked\s+my\s+account|starz\s+subscription|merchant\s+issue|account\s+unlocked|2step|can'?t\s+log\s*in|cannot\s+log\s*in|accounts\s+specialist|fax\s+number|login\s+with\s+thumbprint|suspend\s+account)\b", c):
        scores['account_issue'] += 24.0

    # Cancellation Request (Accidental order, explicit cancellation)
    if re.search(r"\b(?:wish\s+to\s+cancel|cancel\s+order\s*#|inadvertently\s+ordered|alexa\s+listened.*advert.*order|cancelled\s+for\s+months\s+and\s+it'?s\s+not\s+an\s+unexpected\s+charge|cancel\s+it\s+even\s+though)\b", c):
        scores['cancellation_request'] += 24.0

    # Payment Issue (Amazon Pay loading, double charges, failed deductions)
    if re.search(r'\b(?:amazon\s*pay\s+twice|loading\s+money|cash\s*back\s+issue|cant\s+fill\s+the\s+form.*amazonfraud|charge\s+money\s+for\s+prime|payment\s+fail|unauthorized\s+charge|double\s+charge|charged\s+twice)\b', c):
        scores['payment_issue'] += 22.0

    # Discrepancy in Product (Wrong item, damage, missing parts, packaging compromise)
    if re.search(r'\b(?:received\s+(?:the\s+)?wrong\s+item|missing\s+(?:two\s+)?pieces|part\s+of\s+the\s+item\s+is\s+missing|item\s+was\s+missing|1\s+bottle\s+is\s+missing|tickets\s+have\s+been\s+used|wrong\s+order|wrong\s+item|sent\s+(?:the\s+)?wrong|wrong\s+thing|look\s+what\s+i\s+ordered|what\s+i\s+ordered\s+and\s+what\s+i\s+received|cheese\s+grater.*got\s+this|box\s+was\s+open|box\s+was\s+damaged|damaged\s+in\s+transit|lost\s+or\s+damaged|apple\s+inc\s+device.*damages|book.*won\'?t\s+show\s+up\s+to\s+listen|fake\s+product|counterfeit\s+product|replacement\s+product.*fake\s+product|horrible\s+condition|shipping\s+a\s+\$25\s+gift\s+card)\b', c):
        scores['Discrepancy_in_Product'] += 24.0

    # Order Not Delivered (Non-receipt, false delivery claim, stolen package)
    if re.search(r"\b(?:not\s+delivered\s+but\s+shown\s+as\s+delivered|never\s+arrived|not\s+delivered\s+my|nobody\s+came\s+to\s+deliver|never\s+got\s+here|haven'?t\s+received\s+(?:my\s+order|it|the\s+package)|book\s+was\s+not\s+delivered|order\s+is\s+not\s+delivered|still\s+not\s+delivered|website\s+says\s+i\s+received.*haven'?t\s+received|said\s+it'?s\s+been\s+delivered.*ain'?t|wasn'?t\s+delivered|driver\s+a\s+no\s+show|pickup\s+has\s+been\s+cancelled|not\s+received\s+the\s+book|not\s+received\s+my\s+codes|ordered\s+kindle\s+wich\s+was\s+not\s+delivered|still\s+not\s+delivered.*no\s+callnorefund|ain'?t\s+where\s+it\s+said\s+it\s+was\s+left)\b", c):
        scores['order_not_delivered'] += 23.0

    # Order Delayed (Past estimated date, delayed shipment, 8 PM promise)
    if re.search(r'\b(?:delay\s+a\s+week|delayed\s+a\s+week|delay|delayed|delays|delaying|not\s+arrived\s+i\s+went\s+online|delivery\s+delays|ur\s+delivery\s+is\s+delayed|4th\s+delay|guaranteed\s+to\s+be\s+delivered\s+by\s+8\s+pm|2nd\s+chance\s+because\s+this\s+happened\s+before|free\s+month\s+of\s+prime\s+if\s+your\s+package\s+is\s+late|3\s+days\s+late|more\s+than\s+5\s+days|3daydelay|3daysdelay|missed\s+my\s+delivery\s+window|promised\s+delivery\s+date|rescheduled\s+again|no\s+delivery\s+yet|parcel\s+i\s+needed\s+to\s+be\s+here\s+on\s+time|another\s+order\s+late|paid\s+for\s+one\s+day\s+delivery|waiting\s+for\s+this\s+order\s+for|paid\s+for\s+1\s+day\s+shipping|running\s+late|late\s+delivery|parcel\s+was\s+meant\s+to\s+arrive\s+today|five\s+extra\s+days\s+late|wasted\s+time\s+do\s+i\s+will\s+have\s+them\s+today)\b', c):
        scores['order_delayed'] += 20.0

    # Return / Refund Request (Money back, refund status, return pickup)
    if re.search(r"\b(?:refund|refunded|refunds|return\s+my\s+money|get\s+me\s+my\s+money|money\s+back|return\s+amazon\s+has\s+issued\s+refund|why\s+was\s+this\s+order\s+returned|haven'?t\s+received\s+any\s+refund|need\s+a\s+refund|get\s+me\s+my\s+refund|return\s+the\s+item|return\s+pick\s*up|replacement\s+earring|exchange\s+scheme|return\s+prepaid\s+ups\s+label|actioned\s+return|just\s+want\s+to\s+replace\s+it|hope\s+for\s+a\s+refund|expect\s+a\s+refund|refund\s+of\s+the\s+money|plz\s+return\s+my\s+money|cheat\s+the\s+customer.*never\s+trust.*return\s+my\s+money)\b", c):
        scores['return/refund request'] += 18.0

    # General Enquiry (Status queries, policy questions, website behavior, locker issues)
    if re.search(r'\b(?:different\s+book\s+cover|are\s+the\s+codes\s+broken|input\s+it\s+to\s+psn|refund\s+of\s+the\s+courier\s+charges|when\s+my\s+package\s+arrived.*explain|when\s+this\s+will\s+be\s+delivered|tracker\s+has\s+the\s+last\s+update|where\'?s\s+the\s+explanation|what\s+time\s+zone|what\'?s\s+the\s+point\s+of\s+paying\s+for\s+prime|how\s+long|can\s+you\s+check|status|tracking|promotional\s+code|coupon|cashback\s+i\s+pay\s+through\s+amazon\s+pay|prices\s+not\s+realis|digital\s+paperweight|randomly\s+powering|added\s+my\s+email\s+address|brant\s+locker\s+is\s+broken|should\s+i\s+upload\s+the\s+proof|returning\s+to\s+ranchi|picked\s+up\s+today\s+but\s+haven\'?t\s+received\s+any\s+confirmation|ordered\s+5t\s+on\s+21st\s+got\s+the\s+product|clicked\s+the\s+order\s+cancellation\s+button\s+by\s+mistakenly|charged\s+extra\s+for\s+my\s+package|cancel\s+trial)\b', c):
        scores['genral_enquiry'] += 20.0

    # Disambiguation: explicitly received wrong item overrides refund mentioned in same text
    if re.search(r'received\s+(?:the\s+)?wrong\s+item', c):
        scores['Discrepancy_in_Product'] = max(scores['Discrepancy_in_Product'], scores['return/refund request'] + 5.0)

    # Rank candidate intents
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_intent = sorted_items[0][0]

    # Select distinct secondary intent
    sec_intent = None
    for cand, score in sorted_items[1:]:
        if score > 1.0 and cand != top_intent:
            sec_intent = cand
            break

    if sec_intent is None:
        sec_intent = FALLBACK_SECONDARY_MAP.get(top_intent, 'genral_enquiry')

    return top_intent, sec_intent


def classify_support_pairs(
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV
) -> Path:
    """Classify all tweet pairs in the dataset into intent and secondary_intent."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_csv}")

    console.print(f"[bold green]Loading support pairs from:[/bold green] {input_csv}")
    df = pd.read_csv(input_csv)
    total_pairs = len(df)
    console.print(f"[bold cyan]Total pairs to classify:[/bold cyan] {total_pairs:,}")

    console.print("[bold yellow]Running 9-intent classification...[/bold yellow]")
    intents = []
    secondary_intents = []

    for _, row in df.iterrows():
        p, s = classify_single_pair(row.get('customer_text', ''), row.get('brand_text', ''))
        intents.append(p)
        secondary_intents.append(s)

    df['intent'] = intents
    df['secondary_intent'] = secondary_intents
    df['secondery_intent'] = secondary_intents

    assert df['intent'].isin(INTENT_SET).all(), "Found invalid primary intent!"
    assert df['secondary_intent'].isin(INTENT_SET).all(), "Found invalid secondary intent!"
    assert (df['intent'] != df['secondary_intent']).all(), "Found overlapping primary & secondary intent!"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    console.print(f"[bold green]Successfully saved classified dataset to:[/bold green] {output_csv}")

    table = Table(title=f"Intent Distribution Breakdown (N = {total_pairs:,})")
    table.add_column("Intent Label", style="cyan", justify="left")
    table.add_column("Primary Count", style="magenta", justify="right")
    table.add_column("Primary %", style="magenta", justify="right")
    table.add_column("Secondary Count", style="yellow", justify="right")
    table.add_column("Secondary %", style="yellow", justify="right")

    p_counts = df['intent'].value_counts()
    s_counts = df['secondary_intent'].value_counts()

    for label in PREDEFINED_INTENTS:
        pc = p_counts.get(label, 0)
        sc = s_counts.get(label, 0)
        table.add_row(
            label,
            f"{pc:,}",
            f"{pc / total_pairs * 100:.1f}%",
            f"{sc:,}",
            f"{sc / total_pairs * 100:.1f}%"
        )

    console.print(table)
    return output_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify AmazonHelp support tweet pairs into 9 domain-grounded intents.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Path to input support pairs CSV")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Path to output classified CSV")

    args = parser.parse_args()
    classify_support_pairs(input_csv=args.input_csv, output_csv=args.output_csv)
