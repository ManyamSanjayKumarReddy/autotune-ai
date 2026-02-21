# AutoTune AI — LangGraph Multi-Agent Pipeline
## Complete Implementation Guide (Cut 1: Core Graph Only)

> **Scope of this guide:** Build and run the full 5-agent LangGraph pipeline locally.  
> No FastAPI, no Celery, no frontend wiring — just the graph working end-to-end.  
> Integration comes in Cut 2.

---

## Table of Contents

1. [Folder Structure](#1-folder-structure)
2. [Dependencies & Setup](#2-dependencies--setup)
3. [State Design](#3-state-design)
4. [Tools](#4-tools)
5. [Agent Implementations](#5-agent-implementations)
   - [Agent 1 — Conversation Analyzer](#agent-1--conversation-analyzer)
   - [Agent 2 — Web Link Fetcher](#agent-2--web-link-fetcher)
   - [Agent 3 — Content Analyzer & Question Generator](#agent-3--content-analyzer--question-generator)
   - [Agent 4 — Dataset Sampler](#agent-4--dataset-sampler)
   - [Human-in-the-Loop Node](#human-in-the-loop-node)
   - [Agent 5 — Validator](#agent-5--validator)
6. [Graph Assembly](#6-graph-assembly)
7. [CSV Export & Metrics](#7-csv-export--metrics)
8. [Running Locally](#8-running-locally)
9. [Token Budget Strategy](#9-token-budget-strategy)
10. [Prompt Reference](#10-prompt-reference)
11. [Common Errors & Fixes](#11-common-errors--fixes)

---

## 1. Folder Structure

Extend your existing `agent_v1/` to this layout:

```
autotune_graph/
├── pyproject.toml
└── agent_v1/
    ├── graph/
    │   ├── __init__.py
    │   ├── states.py          ← REWRITE (new shared state)
    │   ├── graph.py           ← REWRITE (new 5-agent graph)
    │   └── app.py             ← entry point / CLI runner
    ├── agents/
    │   ├── __init__.py
    │   ├── analyzer.py        ← Agent 1
    │   ├── link_fetcher.py    ← Agent 2
    │   ├── question_gen.py    ← Agent 3
    │   ├── sampler.py         ← Agent 4
    │   └── validator.py       ← Agent 5
    ├── tools/
    │   ├── __init__.py
    │   ├── filesystem.py      ← keep existing
    │   └── web_search.py      ← NEW
    ├── prompts/
    │   ├── __init__.py
    │   └── prompts.py         ← REWRITE (5 new prompts)
    ├── hitl/
    │   ├── __init__.py
    │   └── review.py          ← Human-in-the-loop logic
    └── export/
        ├── __init__.py
        └── csv_writer.py      ← CSV + metrics
```

---

## 2. Dependencies & Setup

### `pyproject.toml` — replace your existing one

```toml
[project]
name = "autotune-graph"
version = "0.2.0"
description = "AutoTune AI — LangGraph multi-agent fine-tuning pipeline"
requires-python = ">=3.12"

dependencies = [
    # LangGraph + LangChain
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-community>=0.3.0",   # for DuckDuckGoSearchRun
    # Scraping / HTTP
    "duckduckgo-search>=6.0.0",
    "requests>=2.32.0",
    "beautifulsoup4>=4.12.0",
    # Utilities
    "python-dotenv>=1.1.0",
    "pydantic>=2.7.0",
    "rich>=13.7.0",                 # pretty console output
    "pandas>=2.2.0",                # CSV export
]
```

```bash
# Install
uv pip install -e .          # if you use uv
# or
pip install -e .

# .env file at project root
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=...        # optional but recommended for tracing
LANGSMITH_TRACING=true
```

---

## 3. State Design

**File:** `agent_v1/graph/states.py`

The entire graph shares one `AutoTuneState` TypedDict. Each agent reads from it and returns a partial dict to update it.

```python
from typing import TypedDict, List, Optional, Annotated
from pydantic import BaseModel, Field
import operator


# ── Pydantic models for structured LLM outputs ────────────────────────────

class TopicAnalysis(BaseModel):
    """Output of Agent 1"""
    is_real_issue: bool = Field(
        description="True if the conversation reveals a genuine model knowledge gap"
    )
    reasoning: str = Field(
        description="Why this is or isn't a real issue (1-2 sentences)"
    )
    topics: List[str] = Field(
        description="List of training topics to address (empty if not a real issue)",
        max_length=5
    )


class WebLink(BaseModel):
    url: str
    title: str
    relevance_reason: str


class LinkBatch(BaseModel):
    """Output of Agent 2"""
    links: List[WebLink] = Field(max_length=5)


class QuestionSet(BaseModel):
    """Output of Agent 3"""
    questions: List[str] = Field(
        description="10 diverse questions derived from the content and topics",
        min_length=5,
        max_length=10
    )


class DataSample(BaseModel):
    """Single training sample"""
    id: str
    question: str
    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    topic: str
    source_url: str


class SampleBatch(BaseModel):
    """Output of Agent 4"""
    samples: List[DataSample]


class ValidationResult(BaseModel):
    """Output of Agent 5 per sample"""
    sample_id: str
    is_valid: bool
    issues: List[str] = Field(default_factory=list)
    revised_answer: Optional[str] = None


class ValidationReport(BaseModel):
    """Output of Agent 5"""
    results: List[ValidationResult]
    total: int
    passed: int
    failed: int


# ── Shared Graph State ────────────────────────────────────────────────────

class AutoTuneState(TypedDict):
    # Input
    conversations: List[dict]           # raw disliked conversations [{role, content}]
    batch_id: str                        # unique ID for this run

    # Agent 1 output
    topic_analysis: Optional[TopicAnalysis]

    # Agent 2 output
    link_batch: Optional[LinkBatch]
    scraped_content: Optional[str]       # cleaned text from all links

    # Agent 3 output
    question_set: Optional[QuestionSet]

    # Agent 4 output
    sample_batch: Optional[SampleBatch]

    # HITL
    hitl_approved_samples: Optional[List[DataSample]]   # after human review
    hitl_rejected_ids: Optional[List[str]]

    # Agent 5 output
    validation_report: Optional[ValidationReport]
    final_samples: Optional[List[DataSample]]

    # Metrics (accumulated with operator.add)
    logs: Annotated[List[str], operator.add]

    # Control flow
    skip: bool                           # True if Agent 1 says not a real issue
```

---

## 4. Tools

### `agent_v1/tools/web_search.py`

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
import requests
from bs4 import BeautifulSoup

# Shared DDG instance (rate-limit-safe: reuse one object)
_ddg = DuckDuckGoSearchRun()


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Returns list of {title, url, snippet} dicts.
    Uses DuckDuckGo — no API key needed.
    """
    raw = _ddg.run(query)
    # DDG returns a single blob string; we use it as context
    # For structured results, use duckduckgo_search directly:
    from duckduckgo_search import DDGS
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")
            })
    return results


def scrape_url(url: str, max_chars: int = 3000) -> str:
    """
    Fetches a URL and returns cleaned plain text (max_chars).
    Returns empty string on failure — never raises.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (AutoTune research bot)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts, styles, nav
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars]
    except Exception as e:
        return f"[SCRAPE FAILED: {e}]"
```

---

## 5. Agent Implementations

### Agent 1 — Conversation Analyzer

**File:** `agent_v1/agents/analyzer.py`

**Job:** Receives 2–3 disliked conversations. Decides if there's a real knowledge gap and returns topics.

```python
from langchain_openai import ChatOpenAI
from agent_v1.graph.states import AutoTuneState, TopicAnalysis
from agent_v1.prompts.prompts import analyzer_prompt
import json


def analyzer_agent(state: AutoTuneState) -> dict:
    """
    Agent 1: Analyze disliked conversations → topics
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    conversations = state["conversations"]

    # Format conversations as readable string (token-budget-aware)
    conv_text = _format_conversations(conversations, max_tokens=800)

    prompt = analyzer_prompt(conv_text)

    result: TopicAnalysis = llm.with_structured_output(TopicAnalysis).invoke(prompt)

    log = (
        f"[Agent1/Analyzer] is_real_issue={result.is_real_issue} | "
        f"topics={result.topics}"
    )

    return {
        "topic_analysis": result,
        "skip": not result.is_real_issue,
        "logs": [log]
    }


def _format_conversations(conversations: list[dict], max_tokens: int = 800) -> str:
    """
    Flatten list of conversations into a string.
    Truncates to stay within token budget (~4 chars per token).
    """
    max_chars = max_tokens * 4
    lines = []
    for i, conv in enumerate(conversations):
        lines.append(f"--- Conversation {i+1} ---")
        if isinstance(conv, list):
            # conv is a list of {role, content} messages
            for msg in conv:
                role = msg.get("role", "?").upper()
                content = msg.get("content", "")
                lines.append(f"{role}: {content}")
        elif isinstance(conv, dict):
            lines.append(str(conv))
    full = "\n".join(lines)
    return full[:max_chars]
```

---

### Agent 2 — Web Link Fetcher

**File:** `agent_v1/agents/link_fetcher.py`

**Job:** Gets topics from Agent 1, searches the web, returns 3–5 relevant links, then scrapes their content.

```python
from langchain_openai import ChatOpenAI
from agent_v1.graph.states import AutoTuneState, LinkBatch
from agent_v1.tools.web_search import search_web, scrape_url
from agent_v1.prompts.prompts import link_fetcher_prompt


def link_fetcher_agent(state: AutoTuneState) -> dict:
    """
    Agent 2: Topics → web links → scraped content
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    topics = state["topic_analysis"].topics

    # Build a search query from topics
    query = " ".join(topics[:3])  # keep query focused

    # Search
    raw_results = search_web(query, max_results=5)

    # Ask LLM to pick the most relevant links
    results_text = "\n".join(
        f"{i+1}. [{r['title']}] {r['url']}\n   {r['snippet']}"
        for i, r in enumerate(raw_results)
    )

    prompt = link_fetcher_prompt(topics, results_text)
    link_batch: LinkBatch = llm.with_structured_output(LinkBatch).invoke(prompt)

    # Scrape each chosen link
    scraped_parts = []
    for link in link_batch.links:
        content = scrape_url(link.url, max_chars=2000)
        if content and not content.startswith("[SCRAPE FAILED"):
            scraped_parts.append(f"[SOURCE: {link.title}]\n{content}")

    scraped_content = "\n\n".join(scraped_parts)

    log = (
        f"[Agent2/LinkFetcher] query='{query}' | "
        f"links_found={len(link_batch.links)} | "
        f"scraped_chars={len(scraped_content)}"
    )

    return {
        "link_batch": link_batch,
        "scraped_content": scraped_content,
        "logs": [log]
    }
```

---

### Agent 3 — Content Analyzer & Question Generator

**File:** `agent_v1/agents/question_gen.py`

**Job:** Reads scraped content + topics → generates 10 training questions.

```python
from langchain_openai import ChatOpenAI
from agent_v1.graph.states import AutoTuneState, QuestionSet
from agent_v1.prompts.prompts import question_gen_prompt


def question_gen_agent(state: AutoTuneState) -> dict:
    """
    Agent 3: Scraped content + topics → 10 questions
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    topics = state["topic_analysis"].topics
    scraped = state["scraped_content"] or ""

    # Trim scraped content to stay within token budget
    # ~6000 chars ≈ 1500 tokens — safe for gpt-4o-mini's 128k context
    scraped_trimmed = scraped[:6000]

    prompt = question_gen_prompt(topics, scraped_trimmed)
    question_set: QuestionSet = llm.with_structured_output(QuestionSet).invoke(prompt)

    log = (
        f"[Agent3/QuestionGen] topics={topics} | "
        f"questions_generated={len(question_set.questions)}"
    )

    return {
        "question_set": question_set,
        "logs": [log]
    }
```

---

### Agent 4 — Dataset Sampler

**File:** `agent_v1/agents/sampler.py`

**Job:** For each of the 10 questions, generate a full training sample: `{id, question, answer, confidence_score, topic, source_url}`.

> **Token strategy:** Process questions in sub-batches of 3 to avoid long-context drift and reduce hallucination risk.

```python
import uuid
from langchain_openai import ChatOpenAI
from agent_v1.graph.states import AutoTuneState, SampleBatch, DataSample
from agent_v1.prompts.prompts import sampler_prompt


def sampler_agent(state: AutoTuneState) -> dict:
    """
    Agent 4: 10 questions → SampleBatch with answers + confidence scores
    Processes in sub-batches of 3 to stay token-efficient.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    questions = state["question_set"].questions
    topics = state["topic_analysis"].topics
    scraped = state["scraped_content"] or ""
    links = state["link_batch"].links if state["link_batch"] else []

    # Sub-batch size: 3 questions at a time
    SUB_BATCH = 3
    all_samples: list[DataSample] = []

    for i in range(0, len(questions), SUB_BATCH):
        q_batch = questions[i:i + SUB_BATCH]

        # Pick a representative source URL for this batch
        source_url = links[0].url if links else "no-source"

        prompt = sampler_prompt(
            questions=q_batch,
            topics=topics,
            context=scraped[:3000],     # fresh context per sub-batch
            source_url=source_url
        )

        batch_result: SampleBatch = llm.with_structured_output(SampleBatch).invoke(prompt)

        # Assign stable IDs
        for sample in batch_result.samples:
            sample.id = f"{state['batch_id']}-{uuid.uuid4().hex[:8]}"
            all_samples.append(sample)

    final_batch = SampleBatch(samples=all_samples)

    log = (
        f"[Agent4/Sampler] questions_in={len(questions)} | "
        f"samples_generated={len(all_samples)}"
    )

    return {
        "sample_batch": final_batch,
        "logs": [log]
    }
```

---

### Human-in-the-Loop Node

**File:** `agent_v1/hitl/review.py`

This is a **blocking CLI review** for Cut 1. In Cut 2 this becomes a REST endpoint + UI.

```python
from rich.console import Console
from rich.table import Table
from rich import box
from agent_v1.graph.states import AutoTuneState, DataSample

console = Console()


def hitl_review_node(state: AutoTuneState) -> dict:
    """
    Human-in-the-Loop: Review samples row by row in the terminal.
    Returns approved and rejected sample IDs.
    """
    samples = state["sample_batch"].samples

    console.print("\n[bold cyan]═══ HUMAN REVIEW ═══[/bold cyan]")
    console.print(f"[yellow]{len(samples)} samples to review.[/yellow]\n")

    # Show all samples in a rich table first
    _print_samples_table(samples)

    # Ask: review individually or approve all?
    mode = console.input(
        "\n[bold]Review mode:[/bold] (A)pprove all / (R)eview row by row / (S)kip all → "
    ).strip().upper()

    approved: list[DataSample] = []
    rejected_ids: list[str] = []

    if mode == "A":
        approved = samples
        console.print(f"[green]✓ Approved all {len(samples)} samples.[/green]")

    elif mode == "S":
        rejected_ids = [s.id for s in samples]
        console.print(f"[red]✗ Rejected all {len(samples)} samples.[/red]")

    else:  # Row-by-row review
        for sample in samples:
            console.rule(f"[bold]Sample {sample.id}[/bold]")
            console.print(f"[cyan]Question:[/cyan] {sample.question}")
            console.print(f"[green]Answer:[/green]   {sample.answer}")
            console.print(f"[yellow]Confidence:[/yellow] {sample.confidence_score:.2f}")
            console.print(f"[dim]Topic: {sample.topic} | Source: {sample.source_url}[/dim]")

            choice = console.input(
                "\n[bold](A)pprove / (R)eject / (E)dit answer → [/bold]"
            ).strip().upper()

            if choice == "A":
                approved.append(sample)
                console.print("[green]✓ Approved[/green]")
            elif choice == "E":
                new_answer = console.input("New answer: ").strip()
                sample.answer = new_answer
                sample.confidence_score = 1.0   # human-edited = max confidence
                approved.append(sample)
                console.print("[green]✓ Approved (edited)[/green]")
            else:
                rejected_ids.append(sample.id)
                console.print("[red]✗ Rejected[/red]")

    log = (
        f"[HITL] approved={len(approved)} | rejected={len(rejected_ids)}"
    )

    return {
        "hitl_approved_samples": approved,
        "hitl_rejected_ids": rejected_ids,
        "logs": [log]
    }


def _print_samples_table(samples: list[DataSample]) -> None:
    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("ID", style="dim", width=20)
    table.add_column("Question", style="cyan", width=40)
    table.add_column("Answer", style="green", width=40)
    table.add_column("Score", style="yellow", width=6)

    for s in samples:
        table.add_row(
            s.id,
            s.question[:80],
            s.answer[:80],
            f"{s.confidence_score:.2f}"
        )
    console.print(table)
```

---

### Agent 5 — Validator

**File:** `agent_v1/agents/validator.py`

**Job:** Final pass over human-approved samples. Checks factual consistency, answer quality, and flags issues.

```python
from langchain_openai import ChatOpenAI
from agent_v1.graph.states import AutoTuneState, ValidationReport, ValidationResult, DataSample
from agent_v1.prompts.prompts import validator_prompt


def validator_agent(state: AutoTuneState) -> dict:
    """
    Agent 5: Validate approved samples → final clean dataset
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    approved = state["hitl_approved_samples"] or []

    if not approved:
        report = ValidationReport(results=[], total=0, passed=0, failed=0)
        return {
            "validation_report": report,
            "final_samples": [],
            "logs": ["[Agent5/Validator] No approved samples to validate."]
        }

    # Validate in batches of 3
    SUB_BATCH = 3
    all_results: list[ValidationResult] = []

    for i in range(0, len(approved), SUB_BATCH):
        batch = approved[i:i + SUB_BATCH]
        prompt = validator_prompt(batch)
        report_chunk: ValidationReport = llm.with_structured_output(ValidationReport).invoke(prompt)
        all_results.extend(report_chunk.results)

    # Build final passing samples
    passed_ids = {r.sample_id for r in all_results if r.is_valid}
    failed_ids = {r.sample_id for r in all_results if not r.is_valid}

    # Apply any revised answers
    revisions = {
        r.sample_id: r.revised_answer
        for r in all_results
        if r.revised_answer
    }

    final_samples: list[DataSample] = []
    for sample in approved:
        if sample.id in passed_ids:
            if sample.id in revisions:
                sample.answer = revisions[sample.id]
            final_samples.append(sample)

    full_report = ValidationReport(
        results=all_results,
        total=len(approved),
        passed=len(passed_ids),
        failed=len(failed_ids)
    )

    log = (
        f"[Agent5/Validator] total={full_report.total} | "
        f"passed={full_report.passed} | failed={full_report.failed}"
    )

    return {
        "validation_report": full_report,
        "final_samples": final_samples,
        "logs": [log]
    }
```

---

## 6. Graph Assembly

**File:** `agent_v1/graph/graph.py` — full replacement

```python
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent_v1.graph.states import AutoTuneState
from agent_v1.agents.analyzer import analyzer_agent
from agent_v1.agents.link_fetcher import link_fetcher_agent
from agent_v1.agents.question_gen import question_gen_agent
from agent_v1.agents.sampler import sampler_agent
from agent_v1.agents.validator import validator_agent
from agent_v1.hitl.review import hitl_review_node
from agent_v1.export.csv_writer import export_to_csv, print_metrics


def init_environment():
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    load_dotenv()


# ── Routing Functions ──────────────────────────────────────────────────────

def route_after_analyzer(state: AutoTuneState) -> str:
    """Skip the rag_pipeline if Agent 1 says it's not a real issue."""
    if state.get("skip"):
        return "skip"
    return "link_fetcher"


def route_after_hitl(state: AutoTuneState) -> str:
    """Skip validation if all samples were rejected."""
    approved = state.get("hitl_approved_samples") or []
    if not approved:
        return "skip"
    return "validator"


# ── Graph Builder ──────────────────────────────────────────────────────────

def build_graph():
    """
    Builds and compiles the AutoTune LangGraph.

    Flow:
        analyzer
            ↓ (real issue?)
        link_fetcher → question_gen → sampler → hitl_review
                                                    ↓ (approved?)
                                                validator → END
    """
    graph = StateGraph(AutoTuneState)

    # Register nodes
    graph.add_node("analyzer", analyzer_agent)
    graph.add_node("link_fetcher", link_fetcher_agent)
    graph.add_node("question_gen", question_gen_agent)
    graph.add_node("sampler", sampler_agent)
    graph.add_node("hitl_review", hitl_review_node)
    graph.add_node("validator", validator_agent)

    # Entry point
    graph.set_entry_point("analyzer")

    # Conditional routing after analyzer
    graph.add_conditional_edges(
        "analyzer",
        route_after_analyzer,
        {
            "link_fetcher": "link_fetcher",
            "skip": END          # not a real issue — stop here
        }
    )

    # Linear edges
    graph.add_edge("link_fetcher", "question_gen")
    graph.add_edge("question_gen", "sampler")
    graph.add_edge("sampler", "hitl_review")

    # Conditional routing after HITL
    graph.add_conditional_edges(
        "hitl_review",
        route_after_hitl,
        {
            "validator": "validator",
            "skip": END          # all rejected — stop
        }
    )

    graph.add_edge("validator", END)

    # Compile with in-memory checkpointer (enables HITL interrupts in future)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── Public Entry Point ─────────────────────────────────────────────────────

def run_pipeline(
    conversations: list[dict],
    batch_id: str = "batch-001"
) -> AutoTuneState:
    """
    Run the full AutoTune rag_pipeline.

    Args:
        conversations: List of disliked conversation dicts.
                       Each item: list of {role, content} messages.
        batch_id: Unique identifier for this rag_pipeline run.

    Returns:
        Final graph state (AutoTuneState dict).
    """
    init_environment()
    graph = build_graph()

    initial_state: AutoTuneState = {
        "conversations": conversations,
        "batch_id": batch_id,
        "topic_analysis": None,
        "link_batch": None,
        "scraped_content": None,
        "question_set": None,
        "sample_batch": None,
        "hitl_approved_samples": None,
        "hitl_rejected_ids": None,
        "validation_report": None,
        "final_samples": None,
        "logs": [],
        "skip": False,
    }

    config = {"configurable": {"thread_id": batch_id}}
    final_state = graph.invoke(initial_state, config=config)

    # Export CSV and print metrics
    if final_state.get("final_samples"):
        csv_path = export_to_csv(final_state["final_samples"], batch_id)
        print_metrics(final_state, csv_path)

    return final_state
```

---

## 7. CSV Export & Metrics

**File:** `agent_v1/export/csv_writer.py`

```python
import csv
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box
from agent_v1.graph.states import DataSample, AutoTuneState

console = Console()

OUTPUT_DIR = "outputs"


def export_to_csv(samples: list[DataSample], batch_id: str) -> str:
    """
    Export final validated samples to CSV.
    Returns the file path.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{OUTPUT_DIR}/{batch_id}_{timestamp}.csv"

    fieldnames = ["id", "question", "answer", "confidence_score", "topic", "source_url"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "id": sample.id,
                "question": sample.question,
                "answer": sample.answer,
                "confidence_score": round(sample.confidence_score, 3),
                "topic": sample.topic,
                "source_url": sample.source_url
            })

    console.print(f"\n[bold green]✓ CSV exported → {filename}[/bold green]")
    return filename


def print_metrics(state: AutoTuneState, csv_path: str) -> None:
    """
    Print a rich summary table of rag_pipeline metrics.
    """
    validation = state.get("validation_report")
    samples = state.get("final_samples") or []
    approved = state.get("hitl_approved_samples") or []
    rejected = state.get("hitl_rejected_ids") or []
    generated = state.get("sample_batch")

    console.rule("[bold cyan]AutoTune Pipeline — Run Summary[/bold cyan]")

    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="cyan", width=35)
    table.add_column("Value", style="bold white", width=20)

    table.add_row("Batch ID", state.get("batch_id", "—"))
    table.add_row("Conversations processed", str(len(state.get("conversations", []))))
    table.add_row("Topics identified", str(len(state.get("topic_analysis").topics if state.get("topic_analysis") else [])))
    table.add_row("Links fetched", str(len(state.get("link_batch").links if state.get("link_batch") else [])))
    table.add_row("Questions generated", str(len(state.get("question_set").questions if state.get("question_set") else [])))
    table.add_row("Samples generated (Agent 4)", str(len(generated.samples) if generated else 0))
    table.add_row("HITL approved", str(len(approved)))
    table.add_row("HITL rejected", str(len(rejected)))

    if validation:
        table.add_row("Validator passed", str(validation.passed))
        table.add_row("Validator failed", str(validation.failed))

    table.add_row("Final samples in CSV", str(len(samples)))
    table.add_row("Output file", csv_path)

    console.print(table)

    # Print all logs
    console.rule("[dim]Pipeline Logs[/dim]")
    for log in state.get("logs", []):
        console.print(f"  [dim]{log}[/dim]")
```

---

## 8. Running Locally

**File:** `agent_v1/graph/app.py` — replace existing

```python
"""
CLI entry point for the AutoTune rag_pipeline.
Usage: python -m agent_v1.graph.app
"""

from agent_v1.graph.graph import run_pipeline

# ── Sample test data — replace with real disliked conversations ────────────

SAMPLE_CONVERSATIONS = [
    [
        {"role": "user", "content": "What are the eligibility criteria for PM Kisan Samman Nidhi Yojana?"},
        {"role": "assistant", "content": "I'm not sure about the exact eligibility, you may want to check the official website."},
        {"role": "user", "content": "That's not helpful, I need specific details."}
    ],
    [
        {"role": "user", "content": "How do I apply for Ayushman Bharat health insurance scheme?"},
        {"role": "assistant", "content": "Ayushman Bharat is a health scheme. Please visit a government portal for details."},
        {"role": "user", "content": "You should know this. Your answer is too vague."}
    ],
    [
        {"role": "user", "content": "What documents are needed for Pradhan Mantri Awas Yojana application?"},
        {"role": "assistant", "content": "You need some identity documents."},
        {"role": "user", "content": "Which identity documents exactly? This doesn't help me."}
    ]
]


if __name__ == "__main__":
    print("Starting AutoTune AI Pipeline...\n")

    final_state = run_pipeline(
        conversations=SAMPLE_CONVERSATIONS,
        batch_id="test-batch-001"
    )

    print("\nPipeline complete.")
```

```bash
# Run from project root
python -m agent_v1.graph.app
```

---

## 9. Token Budget Strategy

This is critical for staying within limits and avoiding hallucination.

| Stage | Input Chars (approx) | Tokens (approx) | Strategy |
|---|---|---|---|
| Agent 1 (Analyzer) | ~3,200 | ~800 | Truncate conversations to 800 tokens |
| Agent 2 (Link Fetcher) | ~1,200 | ~300 | Only pass topics + search snippets |
| Agent 3 (Question Gen) | ~6,000 | ~1,500 | Trim scraped content to 6000 chars |
| Agent 4 (Sampler) | ~3,000 per sub-batch | ~750 | Sub-batch of 3 questions, fresh context |
| Agent 5 (Validator) | ~2,400 per sub-batch | ~600 | Sub-batch of 3 samples |

**Rules:**
- Never pass the full scraped content to every agent. Pass it once to Agent 3, summarise in Agent 4.
- `gpt-4o-mini` at `max_tokens=1024` for all agents — cheaper, fast, good enough.
- Add `max_tokens=1024` to every `ChatOpenAI()` call in production: `ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=1024)`.

---

## 10. Prompt Reference

**File:** `agent_v1/prompts/prompts.py` — full replacement

```python
from agent_v1.graph.states import DataSample


def analyzer_prompt(conversations_text: str) -> str:
    return f"""You are an AI training data quality analyst.

You will receive 2-3 conversations where a user clicked "Dislike" or "Not Satisfied".

Your job:
1. Determine if the model's responses reveal a GENUINE knowledge or capability gap
2. If yes, identify 3-5 specific training topics that would fix these gaps
3. If the dissatisfaction is about tone/style only (not knowledge), mark as NOT a real issue

Be conservative — only flag as a real issue if the model clearly lacks factual knowledge.

CONVERSATIONS:
{conversations_text}

Respond with a TopicAnalysis object."""


def link_fetcher_prompt(topics: list[str], search_results: str) -> str:
    return f"""You are a research curator.

Topics that need training data: {", ".join(topics)}

Web search results:
{search_results}

Select the 3-5 most relevant, authoritative, and informative links from the above results.
Prefer: official government sites, academic sources, reputable news outlets.
Avoid: forums, opinion pieces, low-quality aggregators.

Respond with a LinkBatch object."""


def question_gen_prompt(topics: list[str], scraped_content: str) -> str:
    return f"""You are an expert at creating training questions for language models.

TOPICS TO COVER: {", ".join(topics)}

REFERENCE CONTENT:
{scraped_content}

Generate exactly 10 training questions based on this content. Requirements:
- Mix factual questions (What is...?), procedural (How do I...?), and edge cases
- Questions must be answerable from the provided content
- Vary difficulty: 3 easy, 4 medium, 3 hard
- Make questions specific, not vague
- Each question should stand alone without needing context

Respond with a QuestionSet object."""


def sampler_prompt(
    questions: list[str],
    topics: list[str],
    context: str,
    source_url: str
) -> str:
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return f"""You are an expert at generating high-quality supervised training data.

REFERENCE CONTEXT:
{context}

TOPICS: {", ".join(topics)}
SOURCE: {source_url}

QUESTIONS TO ANSWER:
{questions_text}

For each question, generate a training sample with:
- id: leave empty (will be assigned)
- question: the question exactly as given
- answer: a complete, accurate, detailed answer (3-5 sentences minimum)
- confidence_score: 0.0-1.0 reflecting how well the context supports this answer
- topic: the most relevant topic from the list
- source_url: use the provided source URL

Respond with a SampleBatch object."""


def validator_prompt(samples: list[DataSample]) -> str:
    samples_text = "\n\n".join(
        f"ID: {s.id}\nQ: {s.question}\nA: {s.answer}\nScore: {s.confidence_score}"
        for s in samples
    )
    return f"""You are a strict quality validator for LLM training data.

Review each sample. For each:
- is_valid: True only if the answer is factually coherent, complete, and not hallucinated
- issues: list specific problems if invalid (e.g. "answer is vague", "contradicts question")
- revised_answer: provide a corrected answer ONLY if you can improve it confidently

SAMPLES TO VALIDATE:
{samples_text}

Be strict. A confidence_score below 0.4 is a red flag. Vague answers are invalid.

Respond with a ValidationReport object."""
```

---

## 11. Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ValidationError: topics max_length exceeded` | Agent 1 returns too many topics | Prompt says "3-5 topics max", Pydantic enforces `max_length=5` |
| `DuckDuckGoSearchRun rate limited` | Too many rapid searches | Add `time.sleep(2)` between batches in `link_fetcher.py` |
| `with_structured_output returns None` | LLM output didn't match schema | Lower temperature, be more explicit in prompt, add retry wrapper |
| `KeyError: hitl_approved_samples` | TypedDict key missing in initial state | Ensure all keys are in the `initial_state` dict in `run_pipeline()` |
| `scrape_url timeout` | Slow site | `max_chars` already limits size; increase `timeout=10` to `timeout=20` |
| Graph hangs at HITL | Waiting for terminal input (correct behavior) | Type `A` to approve all and continue |

### Retry Wrapper (optional but recommended)

```python
import time

def with_retry(fn, retries=3, delay=2):
    """Wrap any LLM call with simple retry logic."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"Retry {attempt+1}/{retries} after error: {e}")
            time.sleep(delay)
```

Usage:
```python
result = with_retry(
    lambda: llm.with_structured_output(TopicAnalysis).invoke(prompt)
)
```

---

## What's Next (Cut 2)

Once the graph runs end-to-end locally:

1. **FastAPI wrapper** — expose `run_pipeline()` as `POST /api/pipeline/start`
2. **Celery Beat** — trigger batch extraction every hour, call the API
3. **HITL as REST** — replace CLI review with `GET /api/pipeline/{batch_id}/samples` + `POST /api/pipeline/{batch_id}/approve`
4. **React dashboard** — display logs, metrics table, sample review UI, CSV download
5. **LangSmith tracing** — every agent run is auto-traced via `LANGSMITH_TRACING=true`

---

*AutoTune AI — Cut 1 Implementation Guide | Built on LangGraph 0.2 + LangChain 0.3*