from typing import TypedDict, List, Optional, Annotated
from pydantic import BaseModel, Field
import operator


# ── Pydantic models — used for structured LLM output ONLY, not graph state ─

class TopicAnalysis(BaseModel):
    is_real_issue: bool = Field(
        description="True if the conversation reveals a genuine model knowledge gap"
    )
    reasoning: str = Field(description="Why this is or isn't a real issue (1-2 sentences)")
    topics: List[str] = Field(
        description="3-5 training topics. Empty list if not a real issue.",
        max_length=5
    )


class WebLink(BaseModel):
    url: str
    title: str
    relevance_reason: str = Field(default="")


class LinkBatch(BaseModel):
    links: List[WebLink] = Field(max_length=5)


class QuestionSet(BaseModel):
    questions: List[str] = Field(min_length=5, max_length=10)


class DataSample(BaseModel):
    id: str = Field(default="")
    question: str
    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    topic: str
    source_url: str


class SampleBatch(BaseModel):
    samples: List[DataSample]


class ValidationResult(BaseModel):
    sample_id: str
    is_valid: bool
    issues: List[str] = Field(default_factory=list)
    revised_answer: Optional[str] = None


class ValidationReport(BaseModel):
    results: List[ValidationResult]
    total: int
    passed: int
    failed: int


# ── Graph State — TypedDict ONLY (LangGraph v1 requirement) ───────────────

class AutoTuneState(TypedDict):
    conversations_text: str          # pre-formatted by backend — plain string
    batch_id: str

    topic_analysis: Optional[dict]
    link_batch: Optional[dict]
    scraped_content: Optional[str]
    question_set: Optional[dict]
    sample_batch: Optional[dict]

    hitl_approved_samples: Optional[List[dict]]
    hitl_rejected_ids: Optional[List[str]]

    validation_report: Optional[dict]
    final_samples: Optional[List[dict]]

    logs: Annotated[List[str], operator.add]
    skip: bool