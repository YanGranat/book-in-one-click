from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from schemas.review import EvidenceLink, PositionRange


class ResearchPoint(BaseModel):
    id: str
    text: str
    location: Optional[PositionRange] = None
    rationale: Optional[str] = None


class ResearchPlan(BaseModel):
    points: List[ResearchPoint] = Field(default_factory=list)


class ResearchIterationNote(BaseModel):
    point_id: str
    step: int
    query: Optional[str] = None
    queries: List[str] = Field(default_factory=list)
    findings: str
    evidence: List[EvidenceLink] = Field(default_factory=list)
    confidence: float = 0.0  # 0..1 subjective confidence
    done: bool = False


class ResearchReport(BaseModel):
    point_id: str
    notes: List[ResearchIterationNote]
    synthesis: str


class Recommendation(BaseModel):
    point_id: str
    action: Literal["keep", "clarify", "rewrite", "remove"]
    explanation: str


class SufficiencyDecision(BaseModel):
    point_id: str
    done: bool
    missing_gaps: List[str] = Field(default_factory=list)
    suggested_queries: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class QueryPack(BaseModel):
    point_id: str
    queries: List[str]


class PrecheckDecision(BaseModel):
    """Deprecated: kept for backward compatibility; not used."""
    point_id: str
    keep: bool
    reason: str


__all__ = [
    "ResearchPoint",
    "ResearchPlan",
    "ResearchIterationNote",
    "ResearchReport",
    "Recommendation",
    "SufficiencyDecision",
    "QueryPack",
    "PrecheckDecision",
]


