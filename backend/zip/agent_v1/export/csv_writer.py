import csv
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
OUTPUT_DIR = "outputs"


def export_to_csv(samples: list[dict], batch_id: str) -> str:
    """Write final validated samples to CSV. Returns file path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{OUTPUT_DIR}/{batch_id}_{ts}.csv"

    fieldnames = ["id", "question", "answer", "confidence_score", "topic", "source_url"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "id": s.get("id", ""),
                "question": s.get("question", ""),
                "answer": s.get("answer", ""),
                "confidence_score": round(float(s.get("confidence_score", 0)), 3),
                "topic": s.get("topic", ""),
                "source_url": s.get("source_url", "")
            })

    console.print(f"\n[bold green]✓ CSV exported → {filename}[/bold green]")
    return filename


def print_metrics(state: dict, csv_path: str) -> None:
    """Print a rich summary of the rag_pipeline run."""
    validation = state.get("validation_report") or {}
    final = state.get("final_samples") or []
    approved = state.get("hitl_approved_samples") or []
    rejected = state.get("hitl_rejected_ids") or []
    sample_batch = state.get("sample_batch") or {}
    topic_analysis = state.get("topic_analysis") or {}
    link_batch = state.get("link_batch") or {}
    question_set = state.get("question_set") or {}

    console.rule("[bold cyan]AutoTune Pipeline — Run Summary[/bold cyan]")

    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="cyan", width=35)
    table.add_column("Value", style="bold white", width=20)

    table.add_row("Batch ID", state.get("batch_id", "—"))
    table.add_row("Conversations processed", str(len(state.get("conversations", []))))
    table.add_row("Topics identified", str(len(topic_analysis.get("topics", []))))
    table.add_row("Links fetched", str(len(link_batch.get("links", []))))
    table.add_row("Questions generated", str(len(question_set.get("questions", []))))
    table.add_row("Samples generated", str(len(sample_batch.get("samples", []))))
    table.add_row("HITL approved", str(len(approved)))
    table.add_row("HITL rejected", str(len(rejected)))
    table.add_row("Validator passed", str(validation.get("passed", "—")))
    table.add_row("Validator failed", str(validation.get("failed", "—")))
    table.add_row("Final samples in CSV", str(len(final)))
    table.add_row("Output file", csv_path)

    console.print(table)
    console.rule("[dim]Pipeline Logs[/dim]")
    for log in state.get("logs", []):
        console.print(f"  [dim]{log}[/dim]")