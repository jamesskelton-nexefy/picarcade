"""
Type definitions for Pic Arcade Agentic Backend

Python equivalents of TypeScript types for prompt parsing and workflow orchestration.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# Prompt Parsing Types (Phase 2)
class PromptEntityType(str, Enum):
    PERSON = "person"
    OBJECT = "object"
    STYLE = "style"
    ACTION = "action"
    SETTING = "setting"


class PromptModifierType(str, Enum):
    QUALITY = "quality"
    STYLE = "style"
    LIGHTING = "lighting"
    MOOD = "mood"
    TECHNICAL = "technical"


class PromptReferenceType(str, Enum):
    CELEBRITY = "celebrity"
    ARTWORK = "artwork"
    STYLE = "style"
    BRAND = "brand"


class PromptEntity(BaseModel):
    text: str = Field(description="The extracted entity text")
    type: PromptEntityType = Field(description="Type of entity")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class PromptModifier(BaseModel):
    text: str = Field(description="The modifier text")
    type: PromptModifierType = Field(description="Type of modifier")
    value: Optional[str] = Field(None, description="Parsed value if applicable")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class PromptReference(BaseModel):
    text: str = Field(description="The reference text from prompt")
    type: PromptReferenceType = Field(description="Type of reference")
    search_query: str = Field(description="Optimized search query")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    image_urls: List[str] = Field(default_factory=list, description="Found image URLs")


class ParsedPrompt(BaseModel):
    intent: str = Field(description="Primary intent of the prompt")
    entities: List[PromptEntity] = Field(default_factory=list)
    modifiers: List[PromptModifier] = Field(default_factory=list)
    references: List[PromptReference] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="Overall parsing confidence")


# Generation Types
class GenerationType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    FACE_SWAP = "face_swap"
    STYLE_TRANSFER = "style_transfer"
    IMAGE_EDIT = "image_edit"
    VIDEO_EDIT = "video_edit"


class GenerationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStep(str, Enum):
    PARSE_PROMPT = "parse_prompt"
    RETRIEVE_REFERENCES = "retrieve_references"
    GENERATE_CONTENT = "generate_content"
    EDIT_CONTENT = "edit_content"
    QUALITY_CHECK = "quality_check"
    FINALIZE = "finalize"


class GenerationParameters(BaseModel):
    model: Optional[str] = None
    style: Optional[str] = None
    quality: Optional[str] = Field(None, pattern="^(low|medium|high|ultra)$")
    aspect_ratio: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance: Optional[float] = None
    negative_prompt: Optional[str] = None
    reference_images: List[str] = Field(default_factory=list)


class GenerationRequest(BaseModel):
    id: str
    user_id: str
    prompt: str
    type: GenerationType
    parameters: GenerationParameters
    status: GenerationStatus
    created_at: datetime
    updated_at: datetime


class WorkflowContext(BaseModel):
    prompt: Optional[ParsedPrompt] = None
    references: List[PromptReference] = Field(default_factory=list)
    generated_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    quality_scores: List[Dict[str, Any]] = Field(default_factory=list)
    retry_count: int = 0


class WorkflowState(BaseModel):
    request_id: str
    current_step: WorkflowStep
    steps: List[WorkflowStep]
    context: WorkflowContext
    status: GenerationStatus


# API Response Types
class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Configuration Types
class ApiConfig(BaseModel):
    base_url: str
    timeout: int = 30
    retries: int = 3
    api_key: Optional[str] = None


class OpenAIConfig(ApiConfig):
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2000


class SearchConfig(ApiConfig):
    provider: str = "bing"  # or "google"
    max_results: int = 10 