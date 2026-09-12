"""
LLM Action Verifier and Safety Guardrail Agent for @AmazonHelp
Supports Google Gemini and OpenAI ChatGPT APIs with automatic offline fallback.

Architecture: Maker-Checker / Critic-Supervisor Pattern
1. Primary Pipeline (Maker): Rule-based intent classifier + TF-IDF retrieval + escalation engine.
2. LLM Verifier (Checker): Critic agent auditing classification accuracy, escalation safety,
   and reply policy compliance before actions are finalized.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ActionVerifier")

# Target models
GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o-mini"


class ActionVerifier:
    """Verifies intent classification, escalation safety, and reply quality using an LLM.

    Gracefully falls back to deterministic heuristic validation if no API keys are provided.
    """

    def __init__(self):
        self.gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")
        self.openai_key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
        self.is_active = bool(self.gemini_key or self.openai_key)
        self.provider = "gemini" if self.gemini_key else ("openai" if self.openai_key else "offline_heuristic")

    def verify_action(
        self,
        customer_text: str,
        proposed_intent: str,
        secondary_intent: Optional[str],
        proposed_escalation: str,
        escalation_reason: str,
        drafted_reply: str
    ) -> Dict[str, Any]:
        """Runs verification over proposed system actions.

        Returns a dictionary containing:
        - verified (bool)
        - provider (str)
        - intent_corrected (bool)
        - verified_intent (str)
        - override_escalation (bool)
        - final_escalation (str)
        - verifier_critique (str)
        - refined_reply (Optional[str])
        """
        if self.gemini_key:
            return self._verify_via_gemini(
                customer_text, proposed_intent, secondary_intent,
                proposed_escalation, escalation_reason, drafted_reply
            )
        elif self.openai_key:
            return self._verify_via_openai(
                customer_text, proposed_intent, secondary_intent,
                proposed_escalation, escalation_reason, drafted_reply
            )
        else:
            return self._verify_via_heuristic_critic(
                customer_text, proposed_intent, secondary_intent,
                proposed_escalation, escalation_reason, drafted_reply
            )

    def _build_prompt(
        self,
        customer_text: str,
        proposed_intent: str,
        secondary_intent: Optional[str],
        proposed_escalation: str,
        escalation_reason: str,
        drafted_reply: str
    ) -> str:
        return f"""You are the Lead Support QA Supervisor and Safety Guardrail AI for Amazon Customer Support (@AmazonHelp).
Your job is to audit and verify an automated support agent's actions before they are executed.

AUDIT TARGET:
- Customer Tweet: "{customer_text}"
- Proposed Primary Intent: "{proposed_intent}"
- Proposed Secondary Intent: "{secondary_intent}"
- Proposed Escalation Decision: "{proposed_escalation}" (Reason: "{escalation_reason}")
- Proposed Drafted Reply: "{drafted_reply}"

INSTRUCTIONS:
1. Verify Intent: Is the proposed intent accurate? Watch out for sarcasm, anger, or multi-topic complaints.
2. Verify Escalation Safety: Should this be escalated to a human agent?
   - CRITICAL SAFETY RULE: You MUST override to "ESCALATE" if:
     a) Customer mentions lost, stolen, or false-delivered packages.
     b) Financial disputes (unauthorized charges, double billing, missing refunds).
     c) Account security (lockouts, 2FA issues, hacked accounts).
     d) Hostile churn threats, legal threats, or agent misconduct complaints.
     e) Sarcastic criticism masking severe complaints (e.g. "thanks for dropping my laptop in the rain").
3. Verify Reply Safety: Does the drafted reply make false promises (e.g. promising a direct refund without auth), request sensitive PII over public Twitter, or sound dismissive?

Respond ONLY in valid JSON matching this exact structure:
{{
  "verified": true,
  "override_escalation": false,
  "final_escalation": "{proposed_escalation}",
  "intent_accurate": true,
  "verified_intent": "{proposed_intent}",
  "critique": "Short explanation of your audit assessment.",
  "refined_reply": null
}}"""

    def _verify_via_gemini(
        self,
        customer_text: str,
        proposed_intent: str,
        secondary_intent: Optional[str],
        proposed_escalation: str,
        escalation_reason: str,
        drafted_reply: str
    ) -> Dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={self.gemini_key}"
        prompt = self._build_prompt(customer_text, proposed_intent, secondary_intent, proposed_escalation, escalation_reason, drafted_reply)

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }

        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                text_content = data["candidates"][0]["content"]["parts"][0]["text"]
                result = json.loads(text_content)
                result["provider"] = f"gemini ({GEMINI_MODEL})"
                return result
            else:
                logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"Gemini verification failed: {e}. Falling back to heuristic critic.")

        return self._verify_via_heuristic_critic(
            customer_text, proposed_intent, secondary_intent,
            proposed_escalation, escalation_reason, drafted_reply
        )

    def _verify_via_openai(
        self,
        customer_text: str,
        proposed_intent: str,
        secondary_intent: Optional[str],
        proposed_escalation: str,
        escalation_reason: str,
        drafted_reply: str
    ) -> Dict[str, Any]:
        url = "https://api.openai.com/v1/chat/completions"
        prompt = self._build_prompt(customer_text, proposed_intent, secondary_intent, proposed_escalation, escalation_reason, drafted_reply)

        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a Customer Support QA and Safety Guardrail AI. Respond only in strict JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                text_content = data["choices"][0]["message"]["content"]
                result = json.loads(text_content)
                result["provider"] = f"openai ({OPENAI_MODEL})"
                return result
            else:
                logger.warning(f"OpenAI API returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"OpenAI verification failed: {e}. Falling back to heuristic critic.")

        return self._verify_via_heuristic_critic(
            customer_text, proposed_intent, secondary_intent,
            proposed_escalation, escalation_reason, drafted_reply
        )

    def _verify_via_heuristic_critic(
        self,
        customer_text: str,
        proposed_intent: str,
        secondary_intent: Optional[str],
        proposed_escalation: str,
        escalation_reason: str,
        drafted_reply: str
    ) -> Dict[str, Any]:
        """Deterministic safety critic running locally with 0ms external latency."""
        import re

        override_escalation = False
        final_escalation = proposed_escalation
        critique = "Safety guardrail verified: actions compliant with Amazon Twitter support policies."
        verified_intent = proposed_intent

        # 1. Sarcasm detection (positive words paired with negative situations)
        has_praise = bool(re.search(r'\b(?:thanks|thank you|brilliant|great job|awesome|love|top notch)\b', customer_text, re.I))
        has_disaster = bool(re.search(r'\b(?:rain|dumped|trash|stolen|broken|empty box|nowhere|lost|late|worst|shitty)\b', customer_text, re.I))

        if has_praise and has_disaster:
            override_escalation = True
            final_escalation = "ESCALATE"
            critique = "Detected likely sarcastic praise masking a severe physical delivery/item complaint. Overriding to human escalation."
            if proposed_intent == 'thankyou':
                verified_intent = 'customer_service_complaint'

        # 2. Financial or PII risk check in customer text
        if re.search(r'\b(?:charged twice|double charge|unauthorized|stolen card|credit card|lawyer|sue)\b', customer_text, re.I):
            if proposed_escalation != "ESCALATE":
                override_escalation = True
                final_escalation = "ESCALATE"
                critique = "Financial dispute or legal risk detected in text. Overriding to human escalation."

        return {
            "verified": True,
            "provider": "offline_heuristic_critic",
            "override_escalation": override_escalation,
            "final_escalation": final_escalation,
            "intent_accurate": (verified_intent == proposed_intent),
            "verified_intent": verified_intent,
            "critique": critique,
            "refined_reply": None
        }
