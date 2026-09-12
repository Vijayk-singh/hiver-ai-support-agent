import os
import re
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "amazonhelp_support_pairs.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "amazonhelp_support_pairs.csv"

# The 18 predefined intent labels
PREDEFINED_INTENTS: List[str] = [
    'order_not_delivered',
    'order_delayed',
    'order_status_inquiry',
    'refund_request',
    'return_request',
    'wrong_item_received',
    'damaged_item',
    'cancellation_request',
    'payment_issue',
    'account_issue',
    'customer_service_complaint',
    'product_inquiry',
    'other',
    'Prime Membership',
    'thankyou',
    'Enquiry',
    'unknown',
    'Urgent'
]

INTENT_SET = set(PREDEFINED_INTENTS)

# Pattern weights for customer text
INTENT_RULES: Dict[str, List[Tuple[str, float]]] = {
    'thankyou': [
        (r'\b(?:thank\s*you|thanks|thx|thnx|grateful|kudos|shoutout|much\s+appreciated|cheers|awesome\s+service|great\s+service|stellar\s+service|appreciate\s+(?:the|your)\s+help|sorted\s+now|resolved\s+now)\b', 14.0),
        (r'\b(?:love\s+@?amazon|love\s+@?user|you\s+guys\s+are\s+(?:the\s+best|awesome|great|amazing))\b', 14.0),
        (r'\b(?:appreciate\s+it|many\s+thanks|big\s+thanks|thank\s+u)\b', 10.0),
        (r'^(?:thanks?|thank\s+you|thx|cheers)[!\.\s]*$', 20.0),
    ],
    'Urgent': [
        (r'\b(?:urgent|urgently|emergency|asap|immediately|right\s+now|critical|at\s+the\s+earliest|matter\s+of\s+urgency|time\s+sensitive)\b', 14.0),
        (r'\bneed\s+(?:this|it)\s+(?:today|tomorrow|asap|urgently|now|immediately)\b', 12.0),
        (r'\b(?:before\s+my\s+(?:flight|wedding|birthday|trip|travel))\b', 10.0),
        (r'\b(?:cannot|can\'t)\s+wait\b', 8.0),
        (r'\bquick(?:ly)?\s+please\b', 8.0),
        (r'\bhigh\s+priority\b', 10.0),
    ],
    'wrong_item_received': [
        (r'\b(?:wrong\s+(?:item|product|order|book|size|color|package|parcel|device|goods|model|variant|thing|edition)|incorrect\s+(?:item|product|order|package))\b', 16.0),
        (r'\b(?:received|sent|got|delivered)\s+(?:the\s+)?wrong\b', 16.0),
        (r'\bdifferent\s+(?:item|product|thing|article|package)\b', 12.0),
        (r'\bordered\s+.*\s+(?:received|got|sent|delivered)\b', 14.0),
        (r'\binstead\s+of\s+(?:the\s+)?(?:item|product|one\s+i\s+ordered|what\s+i\s+ordered)\b', 12.0),
        (r'\bnot\s+what\s+i\s+ordered\b', 14.0),
        (r'\bsent\s+(?:me\s+)?someone\s+else\'s\b', 14.0),
    ],
    'damaged_item': [
        (r'\b(?:damaged|broken|cracked|smashed|shattered|dented|punctured|torn|ripped|leaking|spilled|tampered|unsealed|ruined|busted)\b', 14.0),
        (r'\b(?:defective|faulty|malfunctioning|not\s+working|stopped\s+working|dead\s+on\s+arrival|poor\s+quality|counterfeit|fake)\b', 12.0),
        (r'\b(?:empty\s+box|seal\s+broken|seal\s+(?:was\s+)?opened|box\s+(?:was\s+)?damaged|package\s+(?:was\s+)?damaged)\b', 14.0),
        (r'\b(?:accidentally\s+opened|items?\s+missing\s+worth|missing\s+from\s+(?:the\s+)?(?:box|package)|missing\s+parts?)\b', 14.0),
        (r'\bcondition\s+of\s+(?:the\s+)?(?:item|product|package|parcel)\b', 10.0),
        (r'\bpackaging\s+was\s+(?:terrible|poor|bad|awful|ruined)\b', 11.0),
        (r'\bpackage\s+was\s+(?:torn|open|smashed|crushed)\b', 14.0),
    ],
    'cancellation_request': [
        (r'\b(?:cancel|cancelling|cancelled|cancellation)\b', 8.0),
        (r'\bcancel\s+(?:my\s+)?(?:order|item|subscription|delivery|purchase|booking)\b', 16.0),
        (r'\b(?:can\'t|cannot|unable\s+to)\s+cancel\b', 14.0),
        (r'\bwhy\s+was\s+my\s+order\s+cancelled\b', 14.0),
        (r'\bstop\s+(?:my\s+)?(?:order|shipment|delivery)\b', 12.0),
        (r'\bcancellation\s+request\b', 14.0),
        (r'\bwant\s+to\s+cancel\b', 14.0),
    ],
    'refund_request': [
        (r'\b(?:refund|refunds|refunded|refunding)\b', 12.0),
        (r'\b(?:money\s+back|return\s+my\s+money|give\s+(?:me\s+)?my\s+money|get\s+my\s+money\s+back)\b', 16.0),
        (r'\b(?:reimburse|reimbursement)\b', 12.0),
        (r'\bwhere\s+is\s+my\s+refund\b', 16.0),
        (r'\bcredit\s+back\s+(?:to\s+my\s+account|my\s+card|my\s+bank)\b', 12.0),
        (r'\brefund\s+(?:status|pending|amount|not\s+received|delay|processed|process)\b', 14.0),
        (r'\bstill\s+haven\'t\s+received\s+(?:the|my)\s+refund\b', 16.0),
        (r'\bwaiting\s+for\s+(?:my\s+)?refund\b', 14.0),
    ],
    'return_request': [
        (r'\b(?:return|returns|returning|returned)\b', 9.0),
        (r'\b(?:replacement|replace|replacing|replaced|exchange|exchanging)\b', 11.0),
        (r'\b(?:pick\s*up|pickup|pick-up)\b', 9.0),
        (r'\b(?:send\s+back|sent\s+back|ship\s+back)\b', 11.0),
        (r'\breturn\s+(?:window|policy|label|process|request|item|product|order|pickup|courier)\b', 14.0),
        (r'\bhow\s+do\s+i\s+return\b', 14.0),
        (r'\bwant\s+to\s+return\b', 14.0),
        (r'\bsend\s+me\s+(?:a\s+)?new\s+(?:product|item|one)\b', 12.0),
        (r'\breturn\s+(?:the\s+)?parcel\b', 12.0),
        (r'\bprocessed\s+a\s+return\b', 14.0),
    ],
    'Prime Membership': [
        (r'\bprime\s*(?:video|now|music|membership|member|trial|student|reading|pantry)\b', 16.0),
        (r'\bcharged\s+for\s+prime\b', 16.0),
        (r'\bcancel\s+prime\b', 16.0),
        (r'\bprime\s+delivery\b', 12.0),
        (r'\bprime\s+account\b', 12.0),
        (r'\b(?:amazon\s+)?prime\b', 8.0),
        (r'\bstream(?:ing)?\s+(?:on\s+prime|video)\b', 12.0),
        (r'\bprime\s+benefits\b', 14.0),
        (r'\bprime\s+now\s+app\b', 14.0),
    ],
    'payment_issue': [
        (r'\b(?:payment|paid|charged|charging|charges|charge|debited|debit|deducted|deduction)\b', 8.0),
        (r'\b(?:double\s+charge[d]?|charged\s+twice|charged\s+again|two\s+times|multiple\s+times)\b', 16.0),
        (r'\b(?:credit\s+card|debit\s+card|bank\s+account|bank\s+statement|transaction)\b', 9.0),
        (r'\b(?:gift\s+card|gift\s+voucher|amazon\s*pay|pay\s+balance|wallet)\b', 12.0),
        (r'\b(?:promo\s*code|discount\s+code|coupon|cashback|promo\s+offer)\b', 10.0),
        (r'\b(?:payment\s+fail(?:ed|ure)?|transaction\s+failed|amount\s+deducted|money\s+deducted)\b', 15.0),
        (r'\bunauthorized\s+charge\b', 16.0),
        (r'\bmoney\s+deducted\s+(?:but|without)\b', 16.0),
        (r'\bbalance\s+not\s+(?:showing|updated|credited)\b', 13.0),
        (r'\badd\s+balance\b', 12.0),
        (r'\bovercharged\b', 14.0),
        (r'\bextra\s+(?:charge|money|fee)\b', 10.0),
    ],
    'account_issue': [
        (r'\b(?:account|login|logging\s*in|log\s*in|sign\s*in|signing\s*in|password|passcode|otp|verification\s*code)\b', 8.0),
        (r'\b(?:locked\s*out|account\s+locked|account\s+suspended|account\s+blocked|account\s+closed|account\s+disabled|hacked)\b', 16.0),
        (r'\b(?:close|delete|deactivate)\s+(?:my\s+)?account\b', 15.0),
        (r'\b(?:reset\s+password|change\s+password|forgot\s+password)\b', 14.0),
        (r'\b(?:two\s*factor|2fa|otp\s+not\s+received|verification\s+code)\b', 14.0),
        (r'\b(?:seller\s+account|merchant\s+account|seller\s+central)\b', 13.0),
        (r'\bunauthorized\s+access\b', 15.0),
        (r'\bcan(?:not|\'t)\s+(?:log\s*in|sign\s*in|access\s+my\s+account)\b', 15.0),
        (r'\b(?:email\s+address|phone\s+number)\s+on\s+account\b', 10.0),
        (r'\bhousehold\s+account\b', 10.0),
    ],
    'order_not_delivered': [
        (r'\b(?:not\s+delivered|never\s+delivered|never\s+arrived|didn\'t\s+arrive|not\s+arrived|hasn\'t\s+arrived|did\s+not\s+arrive)\b', 16.0),
        (r'\b(?:haven\'t|hasn\'t|didn\'t|never)\s+(?:received|gotten|got|receive)\s+(?:my\s+)?(?:order|package|parcel|item|delivery)\b', 16.0),
        (r'\b(?:missing|lost)\s+(?:package|parcel|order|item|shipment)\b', 14.0),
        (r'\b(?:says|marked|shows|stated|claimed)\s+(?:as\s+)?(?:delivered|dlvd)\s+(?:but|and\s+not|yet|nowhere|haven\'t|nobody)\b', 16.0),
        (r'\bdelivered\s+to\s+(?:wrong|someone\s+else|different\s+address|another\s+house|wrong\s+country|bushes)\b', 15.0),
        (r'\bwhere\s+is\s+my\s+(?:order|package|parcel)\b.*(?:says?\s+delivered|marked|never\s+came|nowhere)', 16.0),
        (r'\bstill\s+(?:have\s+not|haven\'t|have\s+never)\s+received\b', 15.0),
        (r'\b(?:couldn\'t|can\'t|didn\'t)\s+deliver\b', 12.0),
        (r'\bno\s+show\b', 11.0),
        (r'\bnon\s*delivery\b', 14.0),
        (r'\bnot\s+received\s+anything\b', 15.0),
        (r'\bnot\s+at\s+my\s+house\b', 14.0),
        (r'\bnever\s+showed\s+up\b', 14.0),
        (r'\bpackage\s+was\s+(?:left|dumped)\s+(?:outside|in\s+the\s+rain|bushes)\b', 12.0),
    ],
    'order_delayed': [
        (r'\b(?:delayed|delay|delays|running\s+late|late\s+delivery|order\s+is\s+late|package\s+is\s+late|delivery\s+is\s+late)\b', 15.0),
        (r'\b(?:supposed|scheduled|expected|promised|due|meant)\s+to\s+(?:arrive|be\s+delivered|come|be\s+here)\s+(?:by|today|yesterday|on|\d+)\b', 13.0),
        (r'\b(?:past|missed)\s+(?:the\s+)?(?:delivery\s+date|estimated\s+date|guaranteed\s+date)\b', 15.0),
        (r'\bwaiting\s+(?:for\s+\d+\s+days|since\s+\w+|over\s+\d+\s+days|past\s+due|even\s+after\s+a\s+month)\b', 13.0),
        (r'\bguaranteed\s+delivery\s+(?:missed|failed|late)\b', 15.0),
        (r'\bwhy\s+(?:is\s+it|is\s+my\s+order)\s+(?:taking\s+so\s+long|so\s+late|delayed)\b', 14.0),
        (r'\b(?:hours?|days?|weeks?|month)\s+late\b', 14.0),
        (r'\bstill\s+(?:waiting|not\s+dispatched|no\s+sign|no\s+delivery)\b', 10.0),
        (r'\bpostponed\b', 12.0),
        (r'\bheld\s+up\b', 10.0),
        (r'\bno\s+sign\s+of\s+it\b', 12.0),
    ],
    'order_status_inquiry': [
        (r'\b(?:tracking|track|awb|dispatch|dispatched|shipped|shipping)\b', 7.0),
        (r'\bwhere\s+is\s+my\s+(?:order|package|parcel|item|delivery)\b', 14.0),
        (r'\bwhen\s+will\s+(?:my\s+order|it|the\s+item|package)\s+(?:arrive|ship|be\s+delivered|come|dispatch|be\s+dispatched)\b', 14.0),
        (r'\bhas\s+(?:my\s+order|it)\s+(?:shipped|been\s+dispatched|been\s+shipped|left)\b', 13.0),
        (r'\b(?:status\s+of\s+(?:my\s+)?order|order\s+status|delivery\s+status|shipping\s+status)\b', 15.0),
        (r'\btracking\s+(?:number|id|link|not\s+updating|not\s+working|details|info)\b', 14.0),
        (r'\b(?:estimated\s+delivery|expected\s+delivery|delivery\s+date|dispatch\s+date)\b', 11.0),
        (r'\bany\s+update\s+on\s+(?:my\s+)?(?:order|delivery|package|shipment)\b', 14.0),
        (r'\bout\s+for\s+delivery\b', 10.0),
        (r'\bshipping\s+information\b', 12.0),
        (r'\bcheck\s+on\s+my\s+order\b', 13.0),
    ],
    'customer_service_complaint': [
        (r'\b(?:customer\s+service|customer\s+support|call\s+center|helpline|executive|representative|agent|rep|advisor|associate|delivery\s+staff|delivery\s+driver|delivery\s+guy|delivery\s+person|courier\s+guy)\b', 7.0),
        (r'\b(?:terrible|horrible|worst|pathetic|poor|awful|vile|disgraceful|disgrace|horrendous|unacceptable|disgusting|appalling|shameful|shocking|crap|shitty|useless|incompetent|clueless|unhelpful|ridiculous)\s+(?:service|support|help|experience|handling|staff|driver|behavior|behaviour|people)?\b', 16.0),
        (r'\b(?:worst\s+(?:customer\s+)?service|worst\s+experience|pathetic\s+service|terrible\s+service|horrible\s+service)\b', 16.0),
        (r'\b(?:hung\s+up|disconnected|hanging\s+up|cut\s+(?:off|the\s+call)|dropped\s+the\s+call)\b', 15.0),
        (r'\b(?:rude|impolite|arrogant|misbehaved|insulting|lies|lying|liars|cheat|fraud|cheated|scam|fraudsters?|criminals?)\b', 14.0),
        (r'\b(?:no\s+response|no\s+reply|no\s+one\s+(?:is\s+)?helping|ignoring\s+me|ignored|wasting\s+my\s+time|waste\s+of\s+time)\b', 13.0),
        (r'\b(?:complaint|grievance|escalat(?:e|ion)|supervisor|manager|ombudsman)\b', 12.0),
        (r'\b(?:never\s+buying|never\s+ordering|never\s+using|so\s+done\s+with)\s+(?:from\s+)?(?:amazon|you)\b', 14.0),
        (r'\b(?:fed\s+up|pissed\s+off|extremely\s+frustrated|losing\s+patience|disappointed|drop\s+the\s+ball|upset\s+me)\b', 12.0),
        (r'\bcontacted\s+\d+\s+times\b', 12.0),
        (r'\bstill\s+await\s+a\s+reply\b', 11.0),
        (r'\bshame\s+on\s+you\b', 13.0),
    ],
    'product_inquiry': [
        (r'\b(?:specifications|specs|compatible|compatibility|dimensions|model|variant|genuine|authentic|original)\b', 13.0),
        (r'\b(?:in\s+stock|out\s+of\s+stock|restock|back\s+in\s+stock|availability|when\s+will\s+it\s+be\s+available)\b', 14.0),
        (r'\bproduct\s+(?:details|description|code|page|information)\b', 11.0),
        (r'\b(?:price\s+difference|price\s+drop|deal|lightning\s+deal|offer\s+price|cost|mrp)\b', 12.0),
        (r'\b(?:does\s+this|will\s+this|is\s+this)\s+(?:work|fit|support|have|come\s+with|supported)\b', 12.0),
        (r'\b(?:warranty\s+(?:period|policy|covered|claim|details)|have\s+warranty|under\s+warranty)\b', 12.0),
        (r'\b(?:echo\s+show|kindle|fire\s+stick|fire\s+tv|echo\s+dot|paperwhite)\b', 8.0),
        (r'\bdifference\s+between\s+(?:these|the\s+two)\b', 12.0),
    ],
    'Enquiry': [
        (r'\b(?:enquiry|inquiry|inquire|curious|wondering)\b', 10.0),
        (r'\bhow\s+(?:do|can)\s+i\b', 9.0),
        (r'\bis\s+it\s+possible\s+(?:to|that)\b', 10.0),
        (r'\bcan\s+(?:i|you)\s+(?:tell\s+me|confirm|clarify|help\s+me\s+with)\b', 9.0),
        (r'\bdo\s+you\s+(?:ship|deliver|provide|accept|have)\b', 10.0),
        (r'\bwhat\s+is\s+the\s+(?:process|procedure|policy|meaning)\b', 11.0),
        (r'\bwhere\s+can\s+i\s+(?:find|check|see|buy)\b', 9.0),
        (r'\bquestion\s+(?:about|regarding)\b', 10.0),
        (r'\bcan\s+i\s+change\s+(?:my\s+)?(?:address|delivery|order)\b', 11.0),
        (r'\bdo\s+you\s+(?:not\s+)?give\s+a\s+time\s+slot\b', 11.0),
    ],
    'unknown': [
        (r'^\W*$', 18.0),
        (r'^[?!\.\s]+$', 18.0),
        (r'^[a-z]{1,3}$', 12.0),
        (r'^\s*$', 18.0),
    ],
}

COMPILED_RULES: Dict[str, List[Tuple[re.Pattern, float]]] = {
    intent: [(re.compile(p, re.I), score) for p, score in rules]
    for intent, rules in INTENT_RULES.items()
}

BRAND_HINTS: Dict[str, Tuple[re.Pattern, float]] = {
    'order_not_delivered': (re.compile(r'has\s+not\s+arrived|hasn\'t\s+arrived|did\s+not\s+arrive|marked\s+as\s+delivered|undelivered|shows\s+as\s+delivered|haven\'t\s+received\s+your\s+package|order\s+is\s+undeliverable|trouble\s+with\s+the\s+delivery', re.I), 8.0),
    'order_delayed': (re.compile(r'sorry\s+for\s+the\s+delay|apologize\s+for\s+the\s+delay|carrier\s+delay|delay\s+in\s+delivery|delayed\s+parcel|late\s+delivery|delay\s+with\s+your\s+order', re.I), 8.0),
    'order_status_inquiry': (re.compile(r'track\s+your\s+(?:package|parcel|order)|estimated\s+delivery\s+date|has\s+not\s+yet\s+dispatched|tracking\s+information|track\s+it\s+here|carrier\s+responsible', re.I), 8.0),
    'refund_request': (re.compile(r'refund|refunds\s+take|issued\s+a\s+refund|refund\s+has\s+been\s+issued|amount\s+refunded', re.I), 8.0),
    'return_request': (re.compile(r'Online\s+Return\s+Center|return\s+label|return\s+pickup|replacement\s+order|returns?\s+policy', re.I), 8.0),
    'damaged_item': (re.compile(r'arrived\s+damaged|damaged\s+item|defective|condition.*arrived|packing\s+feedback|contents\s+still\s+okay', re.I), 8.0),
    'wrong_item_received': (re.compile(r'incorrect\s+item|wrong\s+item|different\s+item', re.I), 9.0),
    'cancellation_request': (re.compile(r'cancel.*order|cancellation|cannot\s+be\s+cancelled|cancel\s+this\s+for\s+you', re.I), 8.0),
    'payment_issue': (re.compile(r'payment\s+method|billing|Amazon\s+Pay|gift\s+card\s+balance|charges\s+pending|authorized\s+charge|pricing', re.I), 8.0),
    'account_issue': (re.compile(r'password\s+assistance|account\s+specialist|security\s+team|Seller\s+Support|sign\s+in|account\s+being\s+locked', re.I), 8.0),
    'Prime Membership': (re.compile(r'Prime\s+membership|Prime\s+Video|Prime\s+benefits|manage\s+Prime|Prime\s+Now', re.I), 8.0),
    'customer_service_complaint': (re.compile(r'unpleasant\s+experience|experience\s+with\s+our\s+(?:representative|associate|executive)|not\s+the\s+experience\s+we\s+expect|how\s+this\s+was\s+handled|internal\s+investigation', re.I), 8.0),
    'thankyou': (re.compile(r'you\'re\s+welcome|you\s+are\s+welcome|glad\s+we\s+could\s+help|happy\s+to\s+help|anytime|glad\s+to\s+hear|our\s+pleasure', re.I), 8.0),
    'product_inquiry': (re.compile(r'product\s+details|check\s+the\s+listing|manufacturer|compatibility|troubleshooting', re.I), 8.0),
}


def classify_single_pair(cust_text: str, brand_text: str) -> Tuple[str, str]:
    """Classify a single tweet pair into primary intent and secondary intent."""
    cust_str = str(cust_text) if pd.notna(cust_text) else ""
    brand_str = str(brand_text) if pd.notna(brand_text) else ""

    scores: Dict[str, float] = {k: 0.0 for k in PREDEFINED_INTENTS}

    # 1. Customer text rules
    for intent, rules in COMPILED_RULES.items():
        for pat, weight in rules:
            if pat.search(cust_str):
                scores[intent] += weight

    # 2. Brand confirmation hints
    for intent, (pat, weight) in BRAND_HINTS.items():
        if pat.search(brand_str):
            scores[intent] += weight

    # 3. Disambiguation heuristics
    # Delivered vs not delivered conflict
    if re.search(r'(?:says?|shows?|marked)\s+(?:as\s+)?delivered', cust_str, re.I) and re.search(r'\b(?:not|never|haven\'t|hasn\'t|didn\'t|nowhere|no\s+sign)\b', cust_str, re.I):
        scores['order_not_delivered'] += 15.0

    # Damaged package/contents
    if re.search(r'\b(?:broken|damaged|smashed|opened|torn|ripped|cracked)\b', cust_str, re.I):
        scores['damaged_item'] += 10.0

    # Wrong item received
    if re.search(r'\b(?:wrong|incorrect|different)\s+(?:item|product|thing|order|book)\b', cust_str, re.I):
        scores['wrong_item_received'] += 15.0

    # Explicit customer service complaint
    if re.search(r'\b(?:worst\s+customer\s+service|useless\s+customer\s+service|hung\s+up|rude|liars?|pathetic\s+service)\b', cust_str, re.I):
        scores['customer_service_complaint'] += 12.0

    # Pure thank you check
    if scores['thankyou'] >= 10.0 and not any(scores[k] >= 8.0 for k in ['order_not_delivered', 'order_delayed', 'damaged_item', 'wrong_item_received', 'payment_issue', 'customer_service_complaint']):
        scores['thankyou'] += 15.0

    # Urgent check
    if re.search(r'\b(?:urgent|urgently|asap|emergency)\b', cust_str, re.I):
        scores['Urgent'] += 10.0

    # Prime mention
    if re.search(r'\bprime\b', cust_str, re.I) and not scores['Prime Membership']:
        scores['Prime Membership'] += 7.0

    # Rank candidate intents
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_intent, top_score = sorted_items[0]

    if top_score <= 1.0:
        if re.search(r'\?|how|what|can|why|where|when', cust_str, re.I):
            top_intent = 'Enquiry'
        else:
            top_intent = 'other'

    # Secondary intent selection (must be distinct from top_intent and in predefined list)
    sec_intent = None
    for cand, score in sorted_items[1:]:
        if score > 1.0 and cand != top_intent:
            sec_intent = cand
            break

    if sec_intent is None:
        # Fallback secondary intent mapping
        fallback_map = {
            'order_not_delivered': 'order_status_inquiry',
            'order_delayed': 'order_status_inquiry',
            'order_status_inquiry': 'Enquiry',
            'refund_request': 'return_request',
            'return_request': 'refund_request',
            'wrong_item_received': 'return_request',
            'damaged_item': 'return_request',
            'cancellation_request': 'refund_request',
            'payment_issue': 'Enquiry',
            'account_issue': 'Enquiry',
            'customer_service_complaint': 'other',
            'product_inquiry': 'Enquiry',
            'Prime Membership': 'Enquiry',
            'thankyou': 'other',
            'Enquiry': 'other',
            'unknown': 'other',
            'Urgent': 'order_status_inquiry',
            'other': 'Enquiry'
        }
        sec_intent = fallback_map.get(top_intent, 'other')

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

    console.print("[bold yellow]Running intent and secondary intent classification...[/bold yellow]")
    intents = []
    secondary_intents = []

    for idx, row in df.iterrows():
        p, s = classify_single_pair(row.get('customer_text', ''), row.get('brand_text', ''))
        intents.append(p)
        secondary_intents.append(s)

    # Assign columns
    df['intent'] = intents
    df['secondary_intent'] = secondary_intents
    # Add secondery_intent alias to be completely robust to spelling variations
    df['secondery_intent'] = secondary_intents

    # Verification checks
    assert df['intent'].isin(INTENT_SET).all(), "Found invalid primary intent!"
    assert df['secondary_intent'].isin(INTENT_SET).all(), "Found invalid secondary intent!"
    assert df['secondery_intent'].isin(INTENT_SET).all(), "Found invalid secondery intent alias!"
    assert (df['intent'] != df['secondary_intent']).all(), "Found overlapping primary & secondary intent!"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    console.print(f"[bold green]Successfully saved classified dataset to:[/bold green] {output_csv}")

    # Summary table
    table = Table(title="Intent Distribution Breakdown (N = 15,000)")
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

    # Sample rows
    console.print("\n[bold green]Sample Classified Tweet Pairs:[/bold green]")
    sample_df = df.sample(5, random_state=42)
    for i, (_, row) in enumerate(sample_df.iterrows(), 1):
        console.print(f"\n[bold]Example {i}[/bold]")
        console.print(f"[cyan]Customer:[/cyan] {row['customer_text']}")
        console.print(f"[blue]Brand:[/blue]    {row['brand_text'][:100]}...")
        console.print(f"[magenta]Primary Intent:[/magenta]   {row['intent']}")
        console.print(f"[yellow]Secondary Intent:[/yellow] {row['secondary_intent']}")

    return output_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify AmazonHelp support tweet pairs into predefined intents.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Path to input support pairs CSV")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Path to output classified CSV")

    args = parser.parse_args()
    classify_support_pairs(
        input_csv=args.input_csv,
        output_csv=args.output_csv
    )
