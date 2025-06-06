"""
Tool System for Pic Arcade Agentic Backend

Implements tools as first-class citizens following the pattern used by
Claude, Cursor, and other leading AI systems. Each tool wraps an external
API or function with standardized interfaces for dynamic invocation.
"""

from .base import Tool, ToolRegistry, ToolResult
from .prompt_tools import PromptParsingTool, PromptOptimizationTool
from .search_tools import PerplexitySearchTool, WebSearchTool, BingImageSearchTool, GoogleImageSearchTool
from .image_tools import (
    FluxImageManipulationTool,
    FluxImageGenerationTool, 
    StyleTransferTool,
    ObjectChangeTool,
    TextEditingTool,
    BackgroundSwapTool,
    CharacterConsistencyTool,
    StableDiffusionImageTool, 
    DALLEImageGenerationTool, 
    ImageEditingTool, 
    FaceSwapTool, 
    QualityAssessmentTool
)
from .video_tools import (
    RunwayVideoTool,
    ReplicateVideoTool,
    VideoEditingTool
)
from .workflow_tools import WorkflowPlanningTool, WorkflowExecutorTool

__all__ = [
    "Tool",
    "ToolRegistry", 
    "ToolResult",
    "PromptParsingTool",
    "PromptOptimizationTool",
    "PerplexitySearchTool",
    "WebSearchTool",
    "BingImageSearchTool",
    "GoogleImageSearchTool",
    "FluxImageManipulationTool",
    "FluxImageGenerationTool",
    "StyleTransferTool",
    "ObjectChangeTool", 
    "TextEditingTool",
    "BackgroundSwapTool",
    "CharacterConsistencyTool",
    "StableDiffusionImageTool",
    "DALLEImageGenerationTool",
    "ImageEditingTool",
    "FaceSwapTool",
    "QualityAssessmentTool",
    "RunwayVideoTool",
    "ReplicateVideoTool",
    "VideoEditingTool",
    "WorkflowPlanningTool",
    "WorkflowExecutorTool"
] 