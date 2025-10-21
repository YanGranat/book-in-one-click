#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BookSubsection(BaseModel):
    id: str
    title: str
    purpose: Optional[str] = None


class BookSection(BaseModel):
    id: str
    title: str
    purpose: Optional[str] = None
    subsections: List[BookSubsection] = Field(default_factory=list)


class BookOutline(BaseModel):
    main_idea: Optional[str] = None
    sections: List[BookSection] = Field(default_factory=list)


class SubsectionPlan(BaseModel):
    section_id: str
    subsection_id: str
    # Plain list of plan points in logical order (3–7 items)
    plan_items: List[str] = Field(default_factory=list)


class BookSectionLead(BaseModel):
    section_id: Optional[str] = None
    lead_markdown: str


__all__ = [
    "BookSubsection",
    "BookSection",
    "BookOutline",
    "SubsectionPlan",
    "BookSectionLead",
]


