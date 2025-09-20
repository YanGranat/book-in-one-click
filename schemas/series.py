from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PostIdea(BaseModel):
    id: str
    title: str
    angle: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class PostIdeaList(BaseModel):
    items: List[PostIdea] = Field(default_factory=list)


class ListSufficiency(BaseModel):
    done: bool
    missing_areas: List[str] = Field(default_factory=list)
    recommended_count: Optional[int] = None


class ExtendResponse(BaseModel):
    items: List[PostIdea] = Field(default_factory=list)


class PrioritizedList(BaseModel):
    items: List[PostIdea] = Field(default_factory=list)


__all__ = [
    "PostIdea",
    "PostIdeaList",
    "ListSufficiency",
    "ExtendResponse",
    "PrioritizedList",
]


