from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class EvidenceLink(BaseModel):
    url: str  # HTTP/HTTPS URL as plain string to satisfy schema constraints
    title: Optional[str] = None
    snippet: Optional[str] = None


class PositionRange(BaseModel):
    start: int
    end: int


class CritiqueItem(BaseModel):
    claim_text: str
    location: Optional[PositionRange] = None  # character range in original post text
    verdict: Literal["pass", "fail", "uncertain"]
    reason: str
    proposed_fix: Optional[str] = None
    evidence: List[EvidenceLink] = Field(default_factory=list)


class FactCheckReport(BaseModel):
    summary: str
    items: List[CritiqueItem]


__all__ = [
    "EvidenceLink",
    "PositionRange",
    "CritiqueItem",
    "FactCheckReport",
]


