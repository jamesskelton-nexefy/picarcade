"""
Base Tool System for Pic Arcade

Implements the core tool abstraction pattern where each tool is a wrapper
around an external API or function with standardized interfaces.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Type
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Categories of tools for organization and discovery."""
    PROMPT_PROCESSING = "prompt_processing"
    IMAGE_SEARCH = "image_search"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDITING = "image_editing"
    VIDEO_GENERATION = "video_generation"
    FACE_MANIPULATION = "face_manipulation"
    QUALITY_ASSESSMENT = "quality_assessment"
    WORKFLOW_PLANNING = "workflow_planning"


class ToolResult(BaseModel):
    """Standardized result from tool execution."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Tool(ABC):
    """
    Abstract base class for all tools in the Pic Arcade system.
    
    Each tool wraps an external API or function with standardized interfaces
    for dynamic invocation by agents.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.description = description
        self.category = category
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.config = config or {}
        self._validate_config()
    
    @abstractmethod
    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with given input data.
        
        Args:
            input_data: Input parameters matching the tool's input schema
            
        Returns:
            ToolResult: Standardized result with success/error status
        """
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate tool configuration (API keys, etc.)"""
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data against the tool's schema.
        
        Args:
            input_data: Input to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Basic validation - in production, use jsonschema
            required_fields = self.input_schema.get("required", [])
            for field in required_fields:
                if field not in input_data:
                    logger.error(f"Missing required field '{field}' for tool '{self.name}'")
                    return False
            return True
        except Exception as e:
            logger.error(f"Input validation failed for tool '{self.name}': {e}")
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata for agent reasoning."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "config_keys": list(self.config.keys())
        }


class ToolRegistry:
    """
    Registry for managing and discovering available tools.
    
    Agents use this to find relevant tools for their tasks.
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {}
    
    def register(self, tool: Tool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool
        
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        self._categories[tool.category].append(tool.name)
        
        logger.info(f"Registered tool '{tool.name}' in category '{tool.category}'")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools in a specific category."""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names]
    
    def search_tools(
        self, 
        query: str, 
        category: Optional[ToolCategory] = None
    ) -> List[Tool]:
        """
        Search for tools by description or name.
        
        Args:
            query: Search query
            category: Optional category filter
            
        Returns:
            List of matching tools
        """
        query_lower = query.lower()
        matching_tools = []
        
        tools_to_search = (
            self.get_tools_by_category(category) if category 
            else list(self._tools.values())
        )
        
        for tool in tools_to_search:
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                matching_tools.append(tool)
        
        return matching_tools
    
    def list_all_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_categories(self) -> List[ToolCategory]:
        """Get all tool categories."""
        return list(self._categories.keys())
    
    async def invoke_tool(
        self, 
        tool_name: str, 
        input_data: Dict[str, Any]
    ) -> ToolResult:
        """
        Invoke a tool by name with input validation.
        
        Args:
            tool_name: Name of tool to invoke
            input_data: Input parameters
            
        Returns:
            ToolResult: Result of tool execution
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found in registry"
            )
        
        if not tool.validate_input(input_data):
            return ToolResult(
                success=False,
                error=f"Invalid input for tool '{tool_name}'"
            )
        
        try:
            import time
            start_time = time.time()
            result = await tool.invoke(input_data)
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )


# Global tool registry instance
tool_registry = ToolRegistry() 