"""
AI Verifier & Safety Guardrail Agent for @AmazonHelp
Audits intent accuracy, detects sarcasm, and verifies escalation safety.
Supports Google Gemini (gemini-2.5-flash) and OpenAI ChatGPT (gpt-4o-mini).
If the verifier is offline or fails, the pipeline safely falls back to the Primary Engine.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ActionVerifier")

GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o-mini"


class ActionVerifier:
    """Independent AI Supervisor that audits and critiques the Primary Engine's actions."""

    def __init__(self):
        self.gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")
        self.openai_key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
        self.is_configured = bool(self.gemini_key or self.openai_key)
        self.preferred_provider = "gemini" if self.gemini_key else ("openai" if self.openai_key else "none")

    def verify_action(
        self,
        customer_text: str,
        proposed_intent: str,
        secondary_intent: Optional[str],
        proposed_escalation: str,
        escalation_reason: str,
        drafted_reply: str
    ) -> Dict[str, Any]:
        """Audits the proposed actions using an LLM.

        If the verifier is offline or fails, returns verifier_online=False
        so the main system can proceed independently without halting.
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
            return {
                "verifier_online": False,
                "provider": "offline",
                "override_escalation": False,
                "final_escalation": proposed_escalation,
                "intent_accurate": True,
                "verified_intent": proposed_intent,
                "critique": "AI Verifier agent is inactive (no API key configured). Result is determined solely on the basis of the Primary System.",
                "refined_reply": None
            }

    def _build_prompt(
        self,
        customer_text: str,
        proposed_intent: str,
        secondary_intent: Optional[str],
        proposed_escalation: str,
        escalation_reason: str,
        drafted_reply: str
    ) -> str:
        return f"""You are the Lead Customer Support QA Supervisor and Safety Guardrail AI for Amazon Customer Support (@AmazonHelp).
Your job is to audit and verify an automated support agent's actions before they are executed.

AUDIT TARGET:
- Customer Tweet: "{customer_text}"
- Proposed Primary Intent: "{proposed_intent}"
- Proposed Secondary Intent: "{secondary_intent}"
- Proposed Escalation Decision: "{proposed_escalation}" (Reason: "{escalation_reason}")
- Proposed Drafted Reply: "{drafted_reply}"

INSTRUCTIONS:
1. Check Intent Accuracy: Watch out for sarcasm, anger, or multi-topic complaints.
2. Check Critical Escalation Safety: Should this query be escalated to a human agent?
   - CRITICAL ESCALATION TRIGGERS (You MUST vote "ESCALATE" if any apply):
     a) Missing, lost, stolen, or false-delivered packages.
     b) Financial disputes (unauthorized deductions, double charges, refund delays).
     c) Account security (lockouts, 2FA errors, suspicious access).
     d) Sarcasm masking severe dissatisfaction (e.g. "thanks Amazon for leaving my parcel in the pouring rain").
     e) Hostile churn threats, legal threats, or representative misconduct complaints.
3. Check Reply Policy: Ensure the draft doesn't make unauthorized commitments or solicit PII over public channels.

Respond ONLY in valid JSON matching this exact structure:
{{
  "verified": true,
  "override_escalation": false,
  "final_escalation": "{proposed_escalation}",
  "intent_accurate": true,
  "verified_intent": "{proposed_intent}",
  "critique": "Clear, concise 1-2 sentence supervisory assessment.",
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
            resp = requests.post(url, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                text_content = data["candidates"][0]["content"]["parts"][0]["text"]
                result = json.loads(text_content)
                result["verifier_online"] = True
                result["provider"] = f"Gemini 2.5 Flash"
                return result
            else:
                logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text}")
                return {
                    "verifier_online": False,
                    "provider": "Gemini (Error)",
                    "override_escalation": False,
                    "final_escalation": proposed_escalation,
                    "intent_accurate": True,
                    "verified_intent": proposed_intent,
                    "critique": f"AI Verifier agent is temporarily unreachable (HTTP {resp.status_code}). Result is determined solely on the basis of the Primary System.",
                    "refined_reply": None
                }
        except Exception as e:
            logger.warning(f"Gemini verification call failed: {e}")
            return {
                "verifier_online": False,
                "provider": "Gemini (Unavailable)",
                "override_escalation": False,
                "final_escalation": proposed_escalation,
                "intent_accurate": True,
                "verified_intent": proposed_intent,
                "critique": "AI Verifier agent is currently offline/timed out. Result is determined solely on the basis of the Primary System.",
                "refined_reply": None
            }

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
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                text_content = data["choices"][0]["message"]["content"]
                result = json.loads(text_content)
                result["verifier_online"] = True
                result["provider"] = f"OpenAI ({OPENAI_MODEL})"
                return result
            else:
                logger.warning(f"OpenAI API returned status {resp.status_code}: {resp.text}")
                return {
                    "verifier_online": False,
                    "provider": "OpenAI (Error)",
                    "override_escalation": False,
                    "final_escalation": proposed_escalation,
                    "intent_accurate": True,
                    "verified_intent": proposed_intent,
                    "critique": f"AI Verifier agent is temporarily unreachable (HTTP {resp.status_code}). Result is determined solely on the basis of the Primary System.",
                    "refined_reply": None
                }
        except Exception as e:
            logger.warning(f"OpenAI verification call failed: {e}")
            return {
                "verifier_online": False,
                "provider": "OpenAI (Unavailable)",
                "override_escalation": False,
                "final_escalation": proposed_escalation,
                "intent_accurate": True,
                "verified_intent": proposed_intent,
                "critique": "AI Verifier agent is currently offline/timed out. Result is determined solely on the basis of the Primary System.",
                "refined_reply": None
            }
