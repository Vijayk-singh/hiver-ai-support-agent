"""CLI and Batch Pipeline to Draft Grounded Replies for Amazon Support Inquiries.

Supports:
1. Interactive / single query reply drafting with grounding evidence.
2. Batch evaluation mode comparing Baseline 1 (Canned), Baseline 2 (1-NN), and Grounded Agent.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge_base import SupportKnowledgeBase
from src.reply_generator import GroundedReplyGenerator
from src.classify_intents import classify_single_pair

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "amazonhelp_support_pairs.csv"


def draft_for_message(customer_text: str, kb: SupportKnowledgeBase, generator: GroundedReplyGenerator):
    """Classify intent, retrieve historical cases, and draft grounded reply."""
    intent, sec_intent = classify_single_pair(customer_text, "")
    retrieved = kb.retrieve_similar(customer_text, intent=intent, top_k=3)
    grounded_res = generator.generate_grounded_reply(
        customer_text=customer_text,
        intent=intent,
        secondary_intent=sec_intent,
        retrieved_cases=retrieved
    )
    b1 = generator.generate_trivial_baseline(intent)
    b2 = generator.generate_retrieval_baseline(customer_text, intent=intent)

    console.print(Panel(f"[bold cyan]Customer Message:[/bold cyan] {customer_text}", title="Incoming Inquiry"))

    # Intent and routing
    meta_table = Table(show_header=False, box=None)
    meta_table.add_row("[bold]Primary Intent:[/bold]", f"[magenta]{intent}[/magenta]")
    meta_table.add_row("[bold]Secondary Intent:[/bold]", f"[yellow]{sec_intent}[/yellow]")
    meta_table.add_row("[bold]Target Resolution Channel:[/bold]", f"[green]{grounded_res['resolution_channel']}[/green]")
    console.print(meta_table)

    # Historical Grounding Cases
    hist_table = Table(title="Top Retrieved Historical Brand Resolutions", title_style="bold blue")
    hist_table.add_column("#", style="dim", width=3)
    hist_table.add_column("Similarity", style="cyan", width=10)
    hist_table.add_column("Historical Customer Tweet", style="white", ratio=1)
    hist_table.add_column("Historical Brand Resolution", style="green", ratio=1)

    for i, c in enumerate(retrieved, 1):
        hist_table.add_row(
            str(i),
            f"{c['similarity']:.3f}",
            c['customer_text'][:110] + ("..." if len(c['customer_text']) > 110 else ""),
            c['brand_text'][:120] + ("..." if len(c['brand_text']) > 120 else "")
        )
    console.print(hist_table)

    # Generated Responses Comparison
    comp_table = Table(title="Generated Replies Comparison Across Approaches", title_style="bold green")
    comp_table.add_column("Approach", style="bold yellow", width=22)
    comp_table.add_column("Drafted Reply", style="white", ratio=1)

    comp_table.add_row("[Trivial Canned]", b1)
    comp_table.add_row("[Simple 1-NN Retrieval]", b2)
    comp_table.add_row("[Grounded Agent]", f"[bold green]{grounded_res['reply']}[/bold green]")
    console.print(comp_table)


def run_batch_drafting(
    input_csv: Path,
    output_csv: Optional[Path],
    sample_size: int,
    kb: SupportKnowledgeBase,
    generator: GroundedReplyGenerator
):
    """Run reply drafting on a sample of support pairs and save results."""
    console.print(f"[bold green]Running batch drafting on {sample_size} support inquiries...[/bold green]")
    df = pd.read_csv(input_csv)
    sample_df = df.sample(min(sample_size, len(df)), random_state=42).copy()

    b1_list = []
    b2_list = []
    grounded_list = []
    retrieved_refs = []

    for _, row in sample_df.iterrows():
        c_text = row['customer_text']
        intent = row.get('intent', 'other')
        sec_intent = row.get('secondary_intent', 'other')

        retrieved = kb.retrieve_similar(c_text, intent=intent, top_k=2)
        b1 = generator.generate_trivial_baseline(intent)
        b2 = generator.generate_retrieval_baseline(c_text, intent=intent)
        grounded = generator.generate_grounded_reply(c_text, intent=intent, secondary_intent=sec_intent, retrieved_cases=retrieved)

        b1_list.append(b1)
        b2_list.append(b2)
        grounded_list.append(grounded['reply'])
        retrieved_refs.append(retrieved[0]['brand_text'] if retrieved else "")

    sample_df['reply_baseline_canned'] = b1_list
    sample_df['reply_baseline_1nn'] = b2_list
    sample_df['reply_grounded_agent'] = grounded_list
    sample_df['top_retrieved_historical_reply'] = retrieved_refs

    if output_csv:
        sample_df.to_csv(output_csv, index=False)
        console.print(f"[bold green]Saved batch draft comparisons to:[/bold green] {output_csv}")

    # Show first 3 samples
    for i, (_, row) in enumerate(sample_df.head(3).iterrows(), 1):
        console.print(f"\n[bold]Batch Sample {i}[/bold] (Intent: [magenta]{row['intent']}[/magenta])")
        console.print(f"[cyan]Customer:[/cyan] {row['customer_text']}")
        console.print(f"[dim]Ground Truth Brand:[/dim] {row['brand_text']}")
        console.print(f"[bold green]Grounded Agent Reply:[/bold green] {row['reply_grounded_agent']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Draft replies grounded in historical brand resolutions.")
    parser.add_argument("--text", type=str, default=None, help="Single customer tweet message to draft reply for.")
    parser.add_argument("--batch", type=int, default=None, help="Run batch drafting on N samples.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_DATA_PATH, help="Path to input CSV.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Path to save batch comparisons CSV.")

    args = parser.parse_args()

    console.print("[dim]Initializing Knowledge Base and Retrieval Engine...[/dim]")
    kb = SupportKnowledgeBase(data_path=args.input_csv)
    generator = GroundedReplyGenerator(knowledge_base=kb)

    if args.text:
        draft_for_message(args.text, kb, generator)
    elif args.batch:
        run_batch_drafting(args.input_csv, args.output_csv, args.batch, kb, generator)
    else:
        # Default test query
        demo_query = "3 different people have given 3 different answers and I still don't have my order. Says delivered Saturday, was not, I was home all day"
        draft_for_message(demo_query, kb, generator)
