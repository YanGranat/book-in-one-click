from __future__ import annotations

from typing import List, Literal, Optional, Dict, Any

from pydantic import BaseModel, Field
from schemas.review import EvidenceLink


class ContentItem(BaseModel):
    id: str
    point: str
    notes: Optional[str] = None
    examples: List[str] = Field(default_factory=list)


class SubsectionOutline(BaseModel):
    id: str
    title: str
    content_items: List[ContentItem] = Field(default_factory=list)


class SectionOutline(BaseModel):
    id: str
    title: str
    lead: Optional[str] = None
    subsections: List[SubsectionOutline] = Field(default_factory=list)


class ArticleOutline(BaseModel):
    title: Optional[str] = None
    sections: List[SectionOutline] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    quote: Optional[str] = None
    datum: Optional[str] = None
    date: Optional[str] = None
    person: Optional[str] = None
    source: Optional[EvidenceLink] = None
    confidence: float = 0.0


class EvidencePack(BaseModel):
    subsection_id: str
    items: List[EvidenceItem] = Field(default_factory=list)


class OutlineOperation(BaseModel):
    op: Literal["add", "remove", "merge", "split", "rename", "move", "update_lead"]
    target_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class OutlineChangeList(BaseModel):
    operations: List[OutlineOperation] = Field(default_factory=list)


class DraftChunk(BaseModel):
    subsection_id: str
    title: str
    markdown: str


class LeadChunk(BaseModel):
    scope: Literal["article", "section"]
    section_id: Optional[str] = None
    markdown: str


class TitleProposal(BaseModel):
    title: str


class ArticleTitleLead(BaseModel):
    title: str
    lead_markdown: str


__all__ = [
    "ContentItem",
    "SubsectionOutline",
    "SectionOutline",
    "ArticleOutline",
    "EvidenceItem",
    "EvidencePack",
    "OutlineOperation",
    "OutlineChangeList",
    "DraftChunk",
    "LeadChunk",
    "TitleProposal",
    "ArticleTitleLead",
]


