"""Escalation Decision Engine for Customer Support AI Agent.

Decides whether an incoming customer message should be auto-handled or escalated
to a human agent — with a stated reason, confidence, risk level, priority,
and target operational team. Fully aligned with the 9-intent human curated taxonomy.
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
        norm_intent = (intent or '').strip()
        sec_intent = (secondary_intent or '').strip()

        # Map legacy / alias intent strings to canonical forms
        alias_map = {
            'general_enquiry': 'genral_enquiry',
            'order_status_inquiry': 'genral_enquiry',
            'product_inquiry': 'genral_enquiry',
            'Enquiry': 'genral_enquiry',
            'other': 'genral_enquiry',
            'thankyou': 'genral_enquiry',
            'wrong_item_received': 'Discrepancy_in_Product',
            'damaged_item': 'Discrepancy_in_Product',
            'return_request': 'return/refund request',
            'refund_request': 'return/refund request',
            'return_refund_request': 'return/refund request',
        }
        norm_intent = alias_map.get(norm_intent, norm_intent)
        sec_intent = alias_map.get(sec_intent, sec_intent)

        # -------------------------------------------------------------
        # 1. P0_URGENT / Critical Emergency Triggers
        # -------------------------------------------------------------
        if (norm_intent == 'Urgent' or sec_intent == 'Urgent' or
                re.search(r'\b(?:urgent|urgently|emergency|critical|time\s+sensitive|matter\s+of\s+urgency)\b', text, re.I) or
                re.search(r'\b(?:flight|wedding|birthday|funeral|hospital|travel)\s+(?:today|tomorrow|morning|night)\b', text, re.I)):
            return EscalationDecision(
                decision="ESCALATE",
                confidence=0.98,
                reason="Time-critical deadline or explicit emergency detected. Requires immediate human prioritization to prevent irreversible failure.",
                risk_level="CRITICAL",
                priority="P0_URGENT",
                suggested_team="Priority Expedited Support"
            )

        # -------------------------------------------------------------
        # 2. Physical Delivery Non-Receipt / Theft / False Delivery (100% Escalate)
        # -------------------------------------------------------------
        if norm_intent == 'order_not_delivered':
            return EscalationDecision(
                decision="ESCALATE",
                confidence=0.96,
                reason="Delivery dispute: customer reports non-receipt, lost/stolen parcel, or tracking marked delivered without delivery. Requires courier geo-trace and refund/reshipment authorization.",
                risk_level="HIGH",
                priority="HIGH",
                suggested_team="Shipping & Delivery Operations"
            )

        # -------------------------------------------------------------
        # 3. Account Security & Credential Compromise (100% Escalate)
        # -------------------------------------------------------------
        if norm_intent == 'account_issue':
            return EscalationDecision(
                decision="ESCALATE",
                confidence=0.95,
                reason="Account security, authentication credentials, or account lockout. Cannot be modified over public channels; requires verified identity audit by an Account Specialist.",
                risk_level="HIGH",
                priority="HIGH",
                suggested_team="Account Security & Specialist Team"
            )

        # -------------------------------------------------------------
        # 4. Payment & Billing Discrepancies
        # -------------------------------------------------------------
        if norm_intent == 'payment_issue':
            # Check for financial loss, double deductions, failed loading, or fraud claims
            if re.search(r'\b(?:twice|double|fraud|cant\s+fill\s+the\s+form|money\s+doesn\'?t\s+get\s+loaded|unauthorized|deducted|overcharged|dispute)\b', text, re.I):
                return EscalationDecision(
                    decision="ESCALATE",
                    confidence=0.93,
                    reason="Financial discrepancy, disputed payment transaction, or failed Amazon Pay balance loading requires secure payment ledger verification.",
                    risk_level="HIGH",
                    priority="HIGH",
                    suggested_team="Billing & Payment Services"
                )
            return EscalationDecision(
                decision="AUTO_HANDLE",
                confidence=0.88,
                reason="Routine payment or cashback policy inquiry; safely auto-handled with secure self-service billing documentation.",
                risk_level="LOW",
                priority="LOW",
                suggested_team="Self-Service Automated Bot"
            )

        # -------------------------------------------------------------
        # 5. Product Discrepancies (Damage, Wrong Item, Missing Parts) -> AUTO_HANDLE
        # -------------------------------------------------------------
        if norm_intent == 'Discrepancy_in_Product':
            return EscalationDecision(
                decision="AUTO_HANDLE",
                confidence=0.92,
                reason="Product discrepancy (damaged item, wrong order sent, or missing parts); customer directed to Online Return Center for self-service prepaid return label and instant replacement.",
                risk_level="LOW",
                priority="LOW",
                suggested_team="Self-Service Automated Bot"
            )

        # -------------------------------------------------------------
        # 6. Returns & Refunds -> AUTO_HANDLE
        # -------------------------------------------------------------
        if norm_intent == 'return/refund request':
            return EscalationDecision(
                decision="AUTO_HANDLE",
                confidence=0.92,
                reason="Standard return or refund status query; customer guided to Online Return Center and provided standard 3-5 business day financial institution processing windows.",
                risk_level="LOW",
                priority="LOW",
                suggested_team="Self-Service Automated Bot"
            )

        # -------------------------------------------------------------
        # 7. Order Delays -> AUTO_HANDLE
        # -------------------------------------------------------------
        if norm_intent == 'order_delayed':
            return EscalationDecision(
                decision="AUTO_HANDLE",
                confidence=0.90,
                reason="Delivery delay guidance; carrier delivery window (up to 8 PM) and self-service 'Your Orders' tracking links provided.",
                risk_level="LOW",
                priority="LOW",
                suggested_team="Self-Service Automated Bot"
            )

        # -------------------------------------------------------------
        # 8. Customer Service Feedback / Venting -> AUTO_HANDLE
        # -------------------------------------------------------------
        if norm_intent == 'customer_service_complaint':
            return EscalationDecision(
                decision="AUTO_HANDLE",
                confidence=0.89,
                reason="Customer feedback or dissatisfaction expressed; acknowledged with empathetic brand response, internal feedback escalation, and direct contact options.",
                risk_level="LOW",
                priority="LOW",
                suggested_team="Self-Service Automated Bot"
            )

        # -------------------------------------------------------------
        # 9. Cancellation Requests -> AUTO_HANDLE
        # -------------------------------------------------------------
        if norm_intent == 'cancellation_request':
            return EscalationDecision(
                decision="AUTO_HANDLE",
                confidence=0.94,
                reason="Order cancellation workflow; customer guided to cancel directly via 'Your Orders' before dispatch or initiate return upon arrival.",
                risk_level="LOW",
                priority="LOW",
                suggested_team="Self-Service Automated Bot"
            )

        # -------------------------------------------------------------
        # 10. General Enquiries & Routine FAQs -> AUTO_HANDLE
        # -------------------------------------------------------------
        if norm_intent in ['genral_enquiry', 'general_enquiry']:
            return EscalationDecision(
                decision="AUTO_HANDLE",
                confidence=0.93,
                reason="General shipment tracking, store policy, or information inquiry; safely auto-handled with self-service 'Your Orders' tracking and help documentation.",
                risk_level="LOW",
                priority="LOW",
                suggested_team="Self-Service Automated Bot"
            )

        # Fallback safe auto-handle
        return EscalationDecision(
            decision="AUTO_HANDLE",
            confidence=0.85,
            reason=f"Standard inquiry for intent '{norm_intent}'; safely guided via self-service portal.",
            risk_level="LOW",
            priority="LOW",
            suggested_team="Self-Service Automated Bot"
        )
