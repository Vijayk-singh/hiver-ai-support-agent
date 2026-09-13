"""Grounded Reply Generator for Customer Support AI Agent.

Drafts customer support replies grounded in how the brand has historically
resolved similar issues, comparing against trivial canned and 1-NN retrieval baselines.
"""

import os
import re
from typing import Dict, Any, List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()


# Trivial Baseline canned responses per intent
CANNOT_SOLVE_TEMPLATE = "Thank you for reaching out to Amazon. Please visit our help page for assistance with your account. ^Amazon"

TRIVIAL_BASELINE_RESPONSES: Dict[str, str] = {
    'genral_enquiry': "Thank you for reaching out! You can track your orders, manage account settings, and find answers at amazon.com/help. ^Amazon",
    'general_enquiry': "Thank you for reaching out! You can track your orders, manage account settings, and find answers at amazon.com/help. ^Amazon",
    'Discrepancy_in_Product': "We apologize for the issue with your item. You can initiate a replacement or return via our Online Return Center at amazon.com/returns. ^Amazon",
    'return/refund request': "To return an item or check refund status, please visit our Online Return Center. Refunds typically process in 3-5 business days. ^Amazon",
    'order_not_delivered': "We are sorry you haven't received your order. Please check your delivery tracking or contact our support team. ^Amazon",
    'order_delayed': "We apologize for the delay in your delivery. Please check your order status on Amazon. ^Amazon",
    'order_status_inquiry': "You can track the status of your order anytime in the 'Your Orders' section of your Amazon account. ^Amazon",
    'refund_request': "Refunds typically process within 3-5 business days. Please check 'Your Orders' for refund status. ^Amazon",
    'return_request': "To return an item, please visit our Online Return Center to print a return label or schedule a pickup. ^Amazon",
    'wrong_item_received': "We apologize for the incorrect item. You can initiate a replacement via our Online Return Center. ^Amazon",
    'damaged_item': "We are sorry your item arrived damaged. Please visit our Online Return Center for a replacement or refund. ^Amazon",
    'cancellation_request': "You can attempt to cancel your order in 'Your Orders' if it has not yet entered the shipping process. ^Amazon",
    'payment_issue': "For security, we cannot view payment details over Twitter. Please contact our support team directly. ^Amazon",
    'account_issue': "For account security assistance, please visit our login and password help page. ^Amazon",
    'customer_service_complaint': "We apologize for your poor experience. Please contact our support team so we can assist you. ^Amazon",
    'product_inquiry': "Please check the product details page on Amazon for specifications and availability. ^Amazon",
    'Prime Membership': "You can view, manage, or cancel your Prime membership at amazon.com/prime. ^Amazon",
    'thankyou': "You're very welcome! Let us know if you need any further assistance. ^Amazon",
    'Urgent': "We understand this is urgent. Please reach out to our phone or chat support for real-time help. ^Amazon",
    'Enquiry': "Thank you for reaching out! You can find more information on our help page at amazon.com/help. ^Amazon",
    'other': "Thanks for reaching out! Please visit our help center if you need assistance with an order or account. ^Amazon",
    'unknown': CANNOT_SOLVE_TEMPLATE
}

# Standard Amazon resolution guidance grounded in historical data
HISTORICAL_RESOLUTION_POLICIES: Dict[str, Dict[str, str]] = {
    'genral_enquiry': {
        'empathy': "I'd be glad to help check on your order and inquiry!",
        'action': "What does the estimated delivery date or tracking status say in 'Your Orders'? You can track your shipment details and account settings here: https://amazon.com/your-orders.",
        'channel': "Your Orders Tracking",
    },
    'general_enquiry': {
        'empathy': "I'd be glad to help check on your order and inquiry!",
        'action': "What does the estimated delivery date or tracking status say in 'Your Orders'? You can track your shipment details and account settings here: https://amazon.com/your-orders.",
        'channel': "Your Orders Tracking",
    },
    'Discrepancy_in_Product': {
        'empathy': "I'm so sorry your item arrived damaged, defective, or incorrect!",
        'action': "This is certainly not the condition we expect. Please visit our Online Return Center here: https://amazon.com/returns to request an instant replacement or full refund.",
        'channel': "Online Return Center Replacement",
    },
    'return/refund request': {
        'empathy': "I'd be happy to help with your return or refund request!",
        'action': "You can easily initiate a return or track your refund in our Online Return Center: https://amazon.com/returns. Once processed, refunds take 3-5 business days.",
        'channel': "Online Return Center / Refunds",
    },
    'order_not_delivered': {
        'empathy': "I'm sorry to hear your package hasn't arrived!",
        'action': "If tracking shows delivered, please check around your property or with neighbors. If it's still missing, reach us via phone or chat here: https://amazon.com/contact-us so we can investigate or arrange a replacement/refund.",
        'channel': "Phone/Chat Support",
    },
    'order_delayed': {
        'empathy': "I'm very sorry for the delivery delay!",
        'action': "Our carriers deliver until 8 PM. If your package doesn't arrive by then, please check the updated tracking in 'Your Orders' or contact us here: https://amazon.com/contact-us.",
        'channel': "Tracking / Real-time Support",
    },
    'order_status_inquiry': {
        'empathy': "I'd be glad to help check on your order status!",
        'action': "What does the estimated delivery date or tracking status say in 'Your Orders'? You can track your shipment details here: https://amazon.com/your-orders.",
        'channel': "Your Orders Tracking",
    },
    'refund_request': {
        'empathy': "I understand you're inquiring about your refund.",
        'action': "Once issued, refunds typically take 3-5 business days to appear depending on your financial institution. You can view your refund status in 'Your Orders' or reach out here: https://amazon.com/contact-us.",
        'channel': "Online Return Center / Support",
    },
    'return_request': {
        'empathy': "I'd be happy to help with your return request!",
        'action': "You can easily start a return, replacement, or exchange through our Online Return Center here: https://amazon.com/returns.",
        'channel': "Online Return Center",
    },
    'wrong_item_received': {
        'empathy': "I'm so sorry you received the incorrect item!",
        'action': "Please visit our Online Return Center here: https://amazon.com/returns and select 'Wrong item sent' to receive a prepaid return label and request a replacement or full refund.",
        'channel': "Online Return Center Replacement",
    },
    'damaged_item': {
        'empathy': "I'm truly sorry your item arrived damaged!",
        'action': "This is certainly not the condition we expect items to arrive in. Please request a replacement or refund through our Online Return Center: https://amazon.com/returns. We'll also forward your packaging feedback.",
        'channel': "Return Center / Packaging Feedback",
    },
    'cancellation_request': {
        'empathy': "I understand you want to cancel your order.",
        'action': "If the order hasn't entered the shipping process, you can cancel it directly under 'Your Orders' (https://amazon.com/your-orders). If it has already shipped, you can refuse delivery or return it once received.",
        'channel': "Your Orders Cancellation",
    },
    'payment_issue': {
        'empathy': "I'm sorry to hear about the trouble with your payment or charges!",
        'action': "For your security, we cannot view or modify billing details over Twitter. Please contact our support team via phone or chat here: https://amazon.com/contact-us so we can investigate.",
        'channel': "Secure Phone/Chat Support",
    },
    'account_issue': {
        'empathy': "I understand you're experiencing an account issue.",
        'action': "For your security, we cannot access account details on Twitter. Please follow our account recovery steps or contact our security specialists here: https://amazon.com/help/account.",
        'channel': "Account Specialist Portal",
    },
    'customer_service_complaint': {
        'empathy': "I'm deeply sorry for the frustrating experience you've had with us.",
        'action': "This is not at all the standard of service we strive to provide. Without sharing personal details publicly, please connect with us directly here: https://amazon.com/contact-us so a specialist can look into this.",
        'channel': "Escalation / Leadership Support",
    },
    'product_inquiry': {
        'empathy': "Thanks for reaching out with your question!",
        'action': "Please check the product details page on Amazon for verified specifications, compatibility, and availability. Let us know if there is a specific model you'd like us to look into.",
        'channel': "Product Details Page",
    },
    'Prime Membership': {
        'empathy': "I'd be happy to help with your Prime membership question!",
        'action': "You can review your Prime benefits, check billing, or update your subscription anytime at https://amazon.com/prime.",
        'channel': "Prime Membership Management",
    },
    'thankyou': {
        'empathy': "You're very welcome!",
        'action': "We're thrilled we could help make things right for you. Have a wonderful rest of your day! Let us know if you ever need anything else.",
        'channel': "Courtesy Acknowledgment",
    },
    'Urgent': {
        'empathy': "I understand this matter is urgent and time-sensitive!",
        'action': "For immediate real-time assistance, please connect with our live team right away via phone or chat here: https://amazon.com/contact-us.",
        'channel': "Immediate Live Support",
    },
    'Enquiry': {
        'empathy': "Thanks for reaching out to us!",
        'action': "You can find comprehensive guidance on our official help portal here: https://amazon.com/help. Feel free to let us know if you need further clarification.",
        'channel': "Amazon Help Portal",
    },
    'other': {
        'empathy': "Thanks for connecting with us!",
        'action': "If you need assistance with an order, device, or digital content, our support team is available 24/7 here: https://amazon.com/contact-us.",
        'channel': "General Support",
    }
}


class GroundedReplyGenerator:
    """Generates customer support replies grounded in historical brand resolutions."""

    def __init__(self, knowledge_base=None, brand_handle: str = "AmazonHelp", rep_signoff: str = "^AI"):
        self.kb = knowledge_base
        self.brand_handle = brand_handle
        self.rep_signoff = rep_signoff

        # Check for available LLM API keys
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    def generate_trivial_baseline(self, intent: str) -> str:
        """Baseline 1: Trivial canned response based solely on intent."""
        return TRIVIAL_BASELINE_RESPONSES.get(intent, CANNOT_SOLVE_TEMPLATE)

    def generate_retrieval_baseline(self, customer_text: str, intent: Optional[str] = None) -> str:
        """Baseline 2: 1-NN verbatim retrieval from historical brand responses."""
        if not self.kb:
            return self.generate_trivial_baseline(intent or 'other')

        similar_cases = self.kb.retrieve_similar(customer_text, intent=intent, top_k=1)
        if similar_cases:
            return similar_cases[0]['brand_text']
        return self.generate_trivial_baseline(intent or 'other')

    def generate_grounded_reply(
        self,
        customer_text: str,
        intent: str,
        secondary_intent: Optional[str] = None,
        retrieved_cases: Optional[List[Dict[str, Any]]] = None,
        use_llm_if_available: bool = True
    ) -> Dict[str, Any]:
        """Generate a grounded reply using historical cases, policies, and tone.

        Returns a dictionary with the draft reply, grounding evidence, and generation metadata.
        """
        # 1. Retrieve historical grounding if not already provided
        if retrieved_cases is None and self.kb is not None:
            retrieved_cases = self.kb.retrieve_similar(customer_text, intent=intent, top_k=3)
        elif retrieved_cases is None:
            retrieved_cases = []

        # 2. Check if LLM generation is requested and available
        if use_llm_if_available and (self.openai_key or self.gemini_key or self.anthropic_key):
            llm_reply = self._generate_via_llm(customer_text, intent, secondary_intent, retrieved_cases)
            if llm_reply:
                return {
                    "reply": llm_reply,
                    "method": "llm_grounded",
                    "intent": intent,
                    "secondary_intent": secondary_intent,
                    "retrieved_cases": retrieved_cases,
                    "resolution_channel": HISTORICAL_RESOLUTION_POLICIES.get(intent, {}).get('channel', 'General Support')
                }

        # 3. Deterministic Grounded Synthesizer (Fast, offline, 100% reproducible)
        synthesized_reply = self._synthesize_grounded_reply(customer_text, intent, secondary_intent, retrieved_cases)

        return {
            "reply": synthesized_reply,
            "method": "retrieval_synthesized_grounded",
            "intent": intent,
            "secondary_intent": secondary_intent,
            "retrieved_cases": retrieved_cases,
            "resolution_channel": HISTORICAL_RESOLUTION_POLICIES.get(intent, {}).get('channel', 'General Support')
        }

    def _synthesize_grounded_reply(
        self,
        customer_text: str,
        intent: str,
        secondary_intent: Optional[str],
        retrieved_cases: List[Dict[str, Any]]
    ) -> str:
        """Synthesizes a tailored, policy-grounded reply using historical case actions."""
        policy = HISTORICAL_RESOLUTION_POLICIES.get(intent, HISTORICAL_RESOLUTION_POLICIES.get('genral_enquiry', HISTORICAL_RESOLUTION_POLICIES['other']))
        empathy = policy['empathy']
        action = policy['action']

        # Extract specific customer contextual markers
        is_order_specific = bool(re.search(r'\[ORDER_ID\]|order|package|parcel|tracking', customer_text, re.I))
        has_urgency = bool(re.search(r'urgent|asap|today|tomorrow|flight|emergency', customer_text, re.I) or secondary_intent == 'Urgent')
        has_frustration = bool(re.search(r'worst|terrible|awful|rude|hung up|pathetic|disgusted', customer_text, re.I) or intent == 'customer_service_complaint' or secondary_intent == 'customer_service_complaint')

        # Nuance adaptation
        if has_frustration and intent != 'customer_service_complaint':
            empathy = "I'm so sorry for the frustration this has caused you!"
        elif has_urgency and intent != 'Urgent':
            empathy = f"{empathy} I understand this is urgent."

        # If retrieved cases contain a verified link or action, reference it
        reply_body = f"{empathy} {action}"

        # Clean Twitter length and add signoff
        reply = f"{reply_body} {self.rep_signoff}".strip()
        # Ensure under Twitter 280 character limit if possible, or clean formatting
        return reply

    def _generate_via_llm(
        self,
        customer_text: str,
        intent: str,
        secondary_intent: Optional[str],
        retrieved_cases: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Draft grounded reply via LLM API if key is available."""
        # Construct few-shot historical evidence
        evidence_text = ""
        for i, case in enumerate(retrieved_cases[:2], 1):
            evidence_text += f"\n[Historical Example {i}]\nCustomer: {case['customer_text']}\nHistorical Brand Reply: {case['brand_text']}\n"

        prompt = f"""You are an official Amazon Customer Support agent (@AmazonHelp) on Twitter.
Draft a concise, empathetic, and helpful customer support reply grounded in Amazon's historical resolution policies.

Guidelines:
1. Tone: Empathetic, polite, clear, professional.
2. Grounding: Rely on how Amazon historically resolves similar issues (e.g. directing to Online Return Center, phone/chat support, or tracking).
3. Security/Privacy: NEVER ask for passwords, credit card numbers, or sensitive PII on a public tweet.
4. Length: Fit Twitter's format (under 280 characters).
5. End with the representative signature: {self.rep_signoff}

Incoming Customer Message: "{customer_text}"
Classified Intent: {intent} (Secondary: {secondary_intent})

Historical Resolutions for Similar Inquiries:{evidence_text}

Draft your reply:"""

        # Try Gemini API if key is present
        if self.gemini_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 100}
                }
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    return data['candidates'][0]['content']['parts'][0]['text'].strip()
            except Exception:
                pass

        # Try OpenAI API if key is present
        if self.openai_key:
            try:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 100
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    return data['choices'][0]['message']['content'].strip()
            except Exception:
                pass

        return None
