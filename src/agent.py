"""Unified AI Support Agent for Amazon Customer Support.

Integrates:
1. Intent Classification (Primary & Secondary Intent)
2. Historical Resolution Grounding (Knowledge Base Retrieval)
3. Grounded Reply Drafting (with Baseline Comparisons)
4. Escalation Decision Making (AUTO_HANDLE vs ESCALATE with stated reason)
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classify_intents import classify_single_pair, PREDEFINED_INTENTS
from src.knowledge_base import SupportKnowledgeBase
from src.reply_generator import GroundedReplyGenerator
from src.escalation_engine import EscalationEngine, EscalationDecision
from src.verifier import ActionVerifier

console = Console()
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "amazonhelp_support_pairs.csv"


class SupportAgentResult(BaseModel):
    """Complete output representation for a customer inquiry."""
    customer_text: str
    intent: str
    secondary_intent: str
    escalation: EscalationDecision
    drafted_reply: str
    reply_method: str
    target_channel: str
    trivial_baseline_reply: str
    retrieval_baseline_reply: str
    historical_cases: List[Dict[str, Any]] = []
    verifier_audit: Optional[Dict[str, Any]] = None


class AmazonSupportAgent:
    """Production AI support agent for @AmazonHelp."""

    def __init__(self, data_path: Path = DEFAULT_DATA_PATH):
        self.data_path = Path(data_path)
        self.kb = SupportKnowledgeBase(data_path=self.data_path)
        self.reply_generator = GroundedReplyGenerator(knowledge_base=self.kb)
        self.escalation_engine = EscalationEngine()
        self.verifier = ActionVerifier()

    def process_message(
        self,
        customer_text: str,
        brand_text_hint: str = ""
    ) -> SupportAgentResult:
        """Process an incoming customer message through all stages."""
        # Step 1: Classify intent
        intent, sec_intent = classify_single_pair(customer_text, brand_text_hint)

        # Step 2: Retrieve historical grounding cases
        retrieved_cases = self.kb.retrieve_similar(customer_text, intent=intent, top_k=2)

        # Step 3: Decide auto-handle vs escalate with stated reason
        escalation_decision = self.escalation_engine.evaluate(
            customer_text=customer_text,
            intent=intent,
            secondary_intent=sec_intent,
            retrieved_cases=retrieved_cases
        )

        # Step 4: Draft grounded reply and baselines
        b1_reply = self.reply_generator.generate_trivial_baseline(intent)
        b2_reply = self.reply_generator.generate_retrieval_baseline(customer_text, intent=intent)
        agent_reply_info = self.reply_generator.generate_grounded_reply(
            customer_text=customer_text,
            intent=intent,
            secondary_intent=sec_intent,
            retrieved_cases=retrieved_cases
        )

        # Step 5: Verification & Safety Guardrail (LLM Critic / Heuristic Guardrail)
        final_reply = agent_reply_info['reply']
        audit = self.verifier.verify_action(
            customer_text=customer_text,
            proposed_intent=intent,
            secondary_intent=sec_intent,
            proposed_escalation=escalation_decision.decision,
            escalation_reason=escalation_decision.reason,
            drafted_reply=final_reply
        )

        # Apply guardrail overrides if critic detected safety violation or sarcasm
        if audit.get("override_escalation"):
            escalation_decision.decision = audit.get("final_escalation", "ESCALATE")
            escalation_decision.reason = f"[Critic Override]: {audit.get('critique', '')}"
            escalation_decision.priority = "HIGH"
            escalation_decision.risk_level = "HIGH"

        if audit.get("refined_reply"):
            final_reply = audit["refined_reply"]
        elif escalation_decision.decision == "ESCALATE":
            if "specialist" not in final_reply.lower() and "phone or chat" not in final_reply.lower():
                final_reply = f"{final_reply} A support specialist has also been notified to assist. ^AI"

        return SupportAgentResult(
            customer_text=customer_text,
            intent=intent,
            secondary_intent=sec_intent,
            escalation=escalation_decision,
            drafted_reply=final_reply,
            reply_method=agent_reply_info['method'],
            target_channel=agent_reply_info['resolution_channel'],
            trivial_baseline_reply=b1_reply,
            retrieval_baseline_reply=b2_reply,
            historical_cases=retrieved_cases,
            verifier_audit=audit
        )


def display_agent_result(result: SupportAgentResult):
    """Pretty prints the full agent triage and reply report."""
    console.print(Panel(f"[bold cyan]{result.customer_text}[/bold cyan]", title="Incoming Customer Inquiry", border_style="cyan"))

    # Intent & Routing Table
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold white", justify="left")
    grid.add_column(justify="left")

    grid.add_row("Primary Intent:", f"[magenta]{result.intent}[/magenta]")
    grid.add_row("Secondary Intent:", f"[yellow]{result.secondary_intent}[/yellow]")
    grid.add_row("Target Channel:", f"[blue]{result.target_channel}[/blue]")
    console.print(grid)

    # Escalation Decision Panel
    esc = result.escalation
    dec_color = "red" if esc.decision == "ESCALATE" else "green"
    esc_content = (
        f"[bold {dec_color}]Decision:[/bold {dec_color}] [{dec_color}]{esc.decision}[/{dec_color}]  "
        f"[bold]Confidence:[/bold] {esc.confidence:.0%}  "
        f"[bold]Risk Level:[/bold] {esc.risk_level}  "
        f"[bold]Priority:[/bold] {esc.priority}\n"
        f"[bold]Assigned Team:[/bold] {esc.suggested_team}\n"
        f"[bold]Stated Reason:[/bold] {esc.reason}"
    )
    console.print(Panel(esc_content, title=f"Escalation Decision: {esc.decision}", border_style=dec_color))

    # Historical Grounding Table
    if result.historical_cases:
        hist_table = Table(title="Historical Grounding Cases (Retrieved from Corpus)", title_style="bold blue")
        hist_table.add_column("Similarity", style="cyan", width=10)
        hist_table.add_column("Historical Customer Tweet", style="white", ratio=1)
        hist_table.add_column("Historical Amazon Resolution", style="green", ratio=1)

        for c in result.historical_cases:
            hist_table.add_row(
                f"{c['similarity']:.3f}",
                c['customer_text'][:100] + ("..." if len(c['customer_text']) > 100 else ""),
                c['brand_text'][:110] + ("..." if len(c['brand_text']) > 110 else "")
            )
        console.print(hist_table)

    # Drafted Replies Comparison Table
    rep_table = Table(title="Drafted Reply & Baselines", title_style="bold green")
    rep_table.add_column("Approach", style="bold yellow", width=22)
    rep_table.add_column("Reply Content", style="white", ratio=1)

    rep_table.add_row("[Trivial Canned]", result.trivial_baseline_reply)
    rep_table.add_row("[Simple 1-NN]", result.retrieval_baseline_reply)
    rep_table.add_row("[Proposed Agent]", f"[bold green]{result.drafted_reply}[/bold green]")
    console.print(rep_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Amazon AI Support Agent Pipeline.")
    parser.add_argument("--text", type=str, default=None, help="Customer inquiry text to process.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to support dataset.")

    args = parser.parse_args()

    console.print("[dim]Starting Amazon AI Support Agent...[/dim]")
    agent = AmazonSupportAgent(data_path=args.data_path)

    if args.text:
        res = agent.process_message(args.text)
        display_agent_result(res)
    else:
        # Run demo cases
        demo_queries = [
            "Where is my package? The tracking says out for delivery today.",
            "I bought shoes and received an empty box! The seal was completely torn open!",
            "Your representative was super rude and hung up on me when I asked why my refund was delayed!",
            "Thank you so much Amazon, you guys resolved my issue in 5 minutes!",
            "URGENT: I need my medicine order before 4pm because I am traveling tonight!"
        ]
        for q in demo_queries:
            console.print("\n" + "=" * 80)
            res = agent.process_message(q)
            display_agent_result(res)
