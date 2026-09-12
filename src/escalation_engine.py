"""Escalation Decision Engine for Customer Support AI Agent.

Decides whether an incoming customer message should be auto-handled or escalated
to a human agent — with a stated reason, confidence, risk level, priority,
and target operational team.
"""

import re
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class EscalationDecision(BaseModel):
    """Structured decision output for customer support triage."""
    decision: str = Field(description="Either 'AUTO_HANDLE' or 'ESCALATE'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    reason: str = Field(description="Detailed human-readable stated reason for the decision")
    risk_level: str = Field(description="Risk level: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'")
    priority: str = Field(description="Queue priority: 'LOW', 'MEDIUM', 'HIGH', or 'P0_URGENT'")
    suggested_team: str = Field(description="Target resolution team or workflow")


class EscalationEngine:
    """Multi-factor decision engine for customer support automation."""

    def __init__(self):
        pass

    def evaluate(
        self,
        customer_text: str,
        intent: str,
        secondary_intent: Optional[str] = None,
        retrieved_cases: Optional[List[Dict[str, Any]]] = None
    ) -> EscalationDecision:
        """Evaluates whether to auto-handle or escalate to a human agent."""
        text = str(customer_text) if customer_text else ""
        sec_intent = secondary_intent or 'other'

        # -------------------------------------------------------------
        # 1. P0_URGENT / Critical Escalation Triggers
        # -------------------------------------------------------------
        if (intent == 'Urgent' or sec_intent == 'Urgent' or
                re.search(r'\b(?:urgent|urgently|emergency|critical|time\s+sensitive|matter\s+of\s+urgency)\b', text, re.I) or
                re.search(r'\b(?:flight|wedding|birthday|funeral|hospital|travel)\s+(?:today|tomorrow|morning|night)\b', text, re.I)):
            return EscalationDecision(
                decision="ESCALATE",
                confidence=0.96,
                reason="Time-critical deadline or explicit emergency detected. Requires immediate human prioritization to prevent irreversible delivery failure.",
                risk_level="CRITICAL",
                priority="P0_URGENT",
                suggested_team="Priority Expedited Support"
            )

        # -------------------------------------------------------------
        # 2. Account Security, Credential & Access Compromise
        # -------------------------------------------------------------
        if (intent == 'account_issue' or
                re.search(r'\b(?:hacked|locked\s*out|unauthorized\s+access|suspended|stolen\s+account|2fa|otp\s+not\s+received|reset\s+password)\b', text, re.I)):
            return EscalationDecision(
                decision="ESCALATE",
                confidence=0.94,
                reason="Account security and authentication credentials cannot be manipulated over public channels; requires verified identity check by an Account Specialist.",
                risk_level="HIGH",
                priority="HIGH",
                suggested_team="Account Security & Specialist Team"
            )

        # -------------------------------------------------------------
        # 3. Severe Customer Frustration / Conduct Complaints / Churn Threat
        # -------------------------------------------------------------
        if (intent == 'customer_service_complaint' or
                re.search(r'\b(?:hung\s+up|disconnected\s+(?:the\s+)?call|rude|lying|liars|cheat|fraud|illegal|lawyer|sue|court|consumer\s+forum)\b', text, re.I) or
                re.search(r'\b(?:never\s+using\s+amazon|contacted\s+\d+\s+times|waiting\s+\d+\s+(?:days|weeks)|unacceptable\s+behavior)\b', text, re.I)):
            return EscalationDecision(
                decision="ESCALATE",
                confidence=0.92,
                reason="High-risk customer dissatisfaction, repeated failed contact attempts, or agent misconduct complaint requiring supervisor de-escalation.",
                risk_level="HIGH",
                priority="HIGH",
                suggested_team="Customer Relations / Executive Escalations"
            )

        # -------------------------------------------------------------
        # 4. Physical Delivery Loss / Theft / False Delivered Status
        # -------------------------------------------------------------
        if (intent == 'order_not_delivered' or
                re.search(r'(?:says?|shows?|marked)\s+(?:as\s+)?delivered.*(?:not|never|haven\'t|nowhere|stolen|missing|empty)', text, re.I)):
            return EscalationDecision(
                decision="ESCALATE",
                confidence=0.89,
                reason="Delivery dispute (tracking marked delivered but customer reports non-receipt); requires carrier geo-tracking trace or human refund/reshipment authorization.",
                risk_level="HIGH",
                priority="HIGH",
                suggested_team="Shipping & Delivery Operations"
            )

        # -------------------------------------------------------------
        # 5. Financial Discrepancies & Disputed Deductions
        # -------------------------------------------------------------
        if (intent in ['payment_issue', 'refund_request'] or
                re.search(r'\b(?:double\s+charge|charged.*twice|unauthorized|money\s+deducted|overcharged|dispute|refund\s+not\s+received|charged\s+again|debited)\b', text, re.I)):
            return EscalationDecision(
                decision="ESCALATE",
                confidence=0.91,
                reason="Direct financial discrepancy, disputed charge, or refund calculation requires secure payment ledger verification.",
                risk_level="HIGH",
                priority="MEDIUM",
                suggested_team="Billing & Payment Services"
            )

        # -------------------------------------------------------------
        # 6. Physical Damage, Contamination & Product Safety
        # -------------------------------------------------------------
        if (intent == 'damaged_item' and
                re.search(r'\b(?:leaking|shattered|broken|empty\s+box|tampered|seal\s+broken|spilled|damaged\s+goods)\b', text, re.I)):
            return EscalationDecision(
                decision="ESCALATE",
                confidence=0.86,
                reason="Physical damage or packaging compromise detected; requires safety inspection or damaged goods concession review.",
                risk_level="MEDIUM",
                priority="MEDIUM",
                suggested_team="Returns & Replacements Support"
            )

        # -------------------------------------------------------------
        # 7. Safe Auto-Handled Workflows (Self-Service Routine Queries)
        # -------------------------------------------------------------
        if intent == 'thankyou':
            return EscalationDecision(
                decision="AUTO_HANDLE",
                confidence=0.99,
                reason="Polite courtesy expression or praise; safely auto-handled with warm brand acknowledgment.",
                risk_level="LOW",
                priority="LOW",
                suggested_team="Self-Service Automated Bot"
            )

        if intent == 'order_status_inquiry':
            return EscalationDecision(
                decision="AUTO_HANDLE",
                confidence=0.91,
                reason="Routine shipment tracking inquiry; safely auto-handled by guiding customer to real-time tracking in 'Your Orders'.",
                risk_level="LOW",
                priority="LOW",
                suggested_team="Self-Service Automated Bot"
            )

        if intent == 'return_request':
            return EscalationDecision(
                decision="AUTO_HANDLE",
                confidence=0.88,
                reason="Standard product return process; safely auto-handled by directing customer to the Online Return Center self-service workflow.",
                risk_level="LOW",
                priority="LOW",
                suggested_team="Self-Service Automated Bot"
            )

        if intent in ['product_inquiry', 'Enquiry']:
            return EscalationDecision(
                decision="AUTO_HANDLE",
                confidence=0.86,
                reason="Informational query regarding product specifications or general store policies; safely auto-handled with official documentation links.",
                risk_level="LOW",
                priority="LOW",
                suggested_team="Self-Service Automated Bot"
            )

        if intent == 'Prime Membership' and not re.search(r'\b(?:cancel|charged\s+without|refund|stolen)\b', text, re.I):
            return EscalationDecision(
                decision="AUTO_HANDLE",
                confidence=0.87,
                reason="General Prime membership feature or benefit inquiry; safely auto-handled with direct link to Prime management.",
                risk_level="LOW",
                priority="LOW",
                suggested_team="Self-Service Automated Bot"
            )

        # -------------------------------------------------------------
        # 8. Fallback for Remaining Categories
        # -------------------------------------------------------------
        if intent in ['other', 'order_delayed']:
            return EscalationDecision(
                decision="AUTO_HANDLE",
                confidence=0.78,
                reason=f"Standard workflow for intent '{intent}'; customer provided with self-service tracking and carrier delivery window guidelines.",
                risk_level="LOW",
                priority="LOW",
                suggested_team="Self-Service Automated Bot"
            )

        return EscalationDecision(
            decision="ESCALATE",
            confidence=0.80,
            reason=f"Category '{intent}' involves multi-step resolution or policy judgment requiring human agent verification.",
            risk_level="MEDIUM",
            priority="MEDIUM",
            suggested_team="General Customer Support"
        )
