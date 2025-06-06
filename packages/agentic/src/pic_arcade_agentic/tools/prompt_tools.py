"""
Prompt Processing Tools for Pic Arcade

Tools that handle prompt parsing, analysis, and optimization using GPT-4o.
"""

import os
import json
import logging
from typing import Dict, Any
from openai import AsyncOpenAI

from .base import Tool, ToolCategory, ToolResult
from ..types import (
    ParsedPrompt,
    PromptEntity,
    PromptModifier,
    PromptReference,
    PromptEntityType,
    PromptModifierType,
    PromptReferenceType
)

logger = logging.getLogger(__name__)


class PromptParsingTool(Tool):
    """
    Tool for parsing user prompts using GPT-4o.
    
    Extracts structured information including intent, entities, modifiers,
    and references for downstream processing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="prompt_parser",
            description="Parse user prompts to extract intent, entities, modifiers, and references using GPT-4o",
            category=ToolCategory.PROMPT_PROCESSING,
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The user prompt to parse"
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "model": {"type": "string", "default": "gpt-4o"},
                            "temperature": {"type": "number", "default": 0.1},
                            "max_tokens": {"type": "integer", "default": 2000}
                        }
                    }
                },
                "required": ["prompt"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "entities": {"type": "array"},
                    "modifiers": {"type": "array"},
                    "references": {"type": "array"},
                    "confidence": {"type": "number"}
                }
            },
            config=config
        )
        
        # Client will be initialized in _validate_config()
    
    def _validate_config(self) -> None:
        """Validate OpenAI API configuration."""
        # Try openai_api_key first, then api_key, then environment variable
        api_key = (self.config.get("openai_api_key") or 
                  self.config.get("api_key") or 
                  os.getenv("OPENAI_API_KEY"))
        if not api_key:
            raise ValueError("OpenAI API key is required for PromptParsingTool")
        
        # Ensure we don't use a Replicate key for OpenAI
        if api_key.startswith("r8_"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required for PromptParsingTool, but only Replicate key found")
        
        self.config["openai_api_key"] = api_key
        
        # Initialize the OpenAI client with the validated key
        self.client = AsyncOpenAI(api_key=api_key)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for GPT-4o parsing."""
        return """You are an expert AI prompt parser for an image/video generation platform. 

Your task is to analyze user prompts and extract structured information for AI generation workflows.

Extract the following information:

1. INTENT: The primary goal (e.g., "generate_portrait", "create_landscape", "edit_image", "style_transfer")

2. ENTITIES: Objects, people, and concepts in the prompt
   - person: Specific people, characters, professions
   - object: Physical items, animals, things
   - style: Art styles, aesthetics, visual approaches
   - action: Verbs, activities, movements
   - setting: Locations, environments, backgrounds

3. MODIFIERS: Quality and style descriptors
   - quality: Resolution, detail level (4K, HD, detailed, etc.)
   - style: Artistic styles (photorealistic, cartoon, oil painting, etc.)
   - lighting: Light conditions (golden hour, studio lighting, dramatic, etc.)
   - mood: Emotional tone (dark, cheerful, mysterious, etc.)
   - technical: Camera/rendering settings (shallow DOF, wide angle, etc.)

4. REFERENCES: Recognizable people, artworks, or brands
   - celebrity: Famous people or characters
   - artwork: Famous paintings, artists, art movements
   - style: Specific visual aesthetics or brands
   - brand: Company styles, product aesthetics

For each entity, modifier, and reference, provide:
- text: The exact text from the prompt
- type: The category
- confidence: Score from 0.0 to 1.0

For references, also provide:
- search_query: Optimized query for image search

Return ONLY valid JSON matching this exact schema:
{
  "intent": "string",
  "entities": [{"text": "string", "type": "string", "confidence": float}],
  "modifiers": [{"text": "string", "type": "string", "value": "string", "confidence": float}],
  "references": [{"text": "string", "type": "string", "search_query": "string", "confidence": float}],
  "confidence": float
}

Be precise and conservative with confidence scores. Only include items you're confident about."""

    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Parse a user prompt using GPT-4o.
        
        Args:
            input_data: Contains 'prompt' and optional 'options'
            
        Returns:
            ToolResult with parsed prompt data
        """
        try:
            prompt = input_data["prompt"]
            options = input_data.get("options", {})
            
            # Prepare messages for GPT-4o
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f"Parse this prompt: {prompt}"}
            ]
            
            # Call GPT-4o API
            response = await self.client.chat.completions.create(
                model=options.get("model", "gpt-4o"),
                messages=messages,
                temperature=options.get("temperature", 0.1),
                max_tokens=options.get("max_tokens", 2000),
                response_format={"type": "json_object"}
            )
            
            # Extract JSON response
            content = response.choices[0].message.content
            if not content:
                return ToolResult(
                    success=False,
                    error="Empty response from GPT-4o"
                )
            
            # Parse and validate JSON
            parsed_data = json.loads(content)
            
            # Convert to structured types
            structured_result = self._convert_to_structured_types(parsed_data)
            
            return ToolResult(
                success=True,
                data=structured_result,
                metadata={
                    "model_used": options.get("model", "gpt-4o"),
                    "prompt_length": len(prompt),
                    "entities_count": len(structured_result.get("entities", [])),
                    "modifiers_count": len(structured_result.get("modifiers", [])),
                    "references_count": len(structured_result.get("references", []))
                }
            )
            
        except json.JSONDecodeError as e:
            return ToolResult(
                success=False,
                error=f"Invalid JSON response from GPT-4o: {e}"
            )
        except Exception as e:
            logger.error(f"Prompt parsing failed: {e}")
            return ToolResult(
                success=False,
                error=f"Prompt parsing failed: {str(e)}"
            )
    
    def _convert_to_structured_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw JSON data to structured types."""
        try:
            # Convert entities
            entities = []
            for entity_data in data.get("entities", []):
                entity = {
                    "text": entity_data["text"],
                    "type": entity_data["type"],
                    "confidence": float(entity_data["confidence"])
                }
                entities.append(entity)
            
            # Convert modifiers
            modifiers = []
            for modifier_data in data.get("modifiers", []):
                modifier = {
                    "text": modifier_data["text"],
                    "type": modifier_data["type"],
                    "value": modifier_data.get("value"),
                    "confidence": float(modifier_data["confidence"])
                }
                modifiers.append(modifier)
            
            # Convert references
            references = []
            for reference_data in data.get("references", []):
                reference = {
                    "text": reference_data["text"],
                    "type": reference_data["type"],
                    "search_query": reference_data["search_query"],
                    "confidence": float(reference_data["confidence"]),
                    "image_urls": []  # Will be populated by search tools
                }
                references.append(reference)
            
            return {
                "intent": data["intent"],
                "entities": entities,
                "modifiers": modifiers,
                "references": references,
                "confidence": float(data["confidence"])
            }
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to convert parsed data: {e}")
            # Return basic fallback
            return {
                "intent": "generate_image",
                "entities": [],
                "modifiers": [],
                "references": [],
                "confidence": 0.1
            }


class PromptOptimizationTool(Tool):
    """
    Tool for optimizing prompts for better AI generation results.
    
    Takes a user prompt and enhances it with better structure, technical terms,
    and style specifications.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="prompt_optimizer",
            description="Optimize user prompts for better AI image/video generation results",
            category=ToolCategory.PROMPT_PROCESSING,
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "generation_type": {
                        "type": "string",
                        "enum": ["image", "video", "portrait", "landscape"],
                        "default": "image"
                    },
                    "style_preference": {"type": "string"},
                    "quality_level": {
                        "type": "string", 
                        "enum": ["standard", "high", "ultra"],
                        "default": "high"
                    }
                },
                "required": ["prompt"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "optimized_prompt": {"type": "string"},
                    "improvements": {"type": "array"},
                    "technical_keywords": {"type": "array"},
                    "confidence": {"type": "number"}
                }
            },
            config=config
        )
        
        # Client will be initialized in _validate_config()
    
    def _validate_config(self) -> None:
        """Validate OpenAI API configuration."""
        # Try openai_api_key first, then api_key, then environment variable
        api_key = (self.config.get("openai_api_key") or 
                  self.config.get("api_key") or 
                  os.getenv("OPENAI_API_KEY"))
        if not api_key:
            raise ValueError("OpenAI API key is required for PromptOptimizationTool")
        
        # Ensure we don't use a Replicate key for OpenAI
        if api_key.startswith("r8_"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required for PromptOptimizationTool, but only Replicate key found")
        
        self.config["openai_api_key"] = api_key
        
        # Initialize the OpenAI client with the validated key
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Optimize a user prompt for better generation results.
        
        Args:
            input_data: Contains prompt and optimization parameters
            
        Returns:
            ToolResult with optimized prompt and metadata
        """
        try:
            prompt = input_data["prompt"]
            generation_type = input_data.get("generation_type", "image")
            quality_level = input_data.get("quality_level", "high")
            
            system_prompt = f"""You are an expert prompt engineer for AI image/video generation.

Your task is to optimize prompts for {generation_type} generation to achieve {quality_level} quality results.

Guidelines:
1. Maintain the core intent and subject matter
2. Add technical photography/art terms when appropriate
3. Include quality modifiers ({quality_level} quality setting)
4. Structure the prompt logically (subject → style → technical specs)
5. Remove redundant or conflicting terms

Return JSON with:
{{
  "optimized_prompt": "enhanced version of the prompt",
  "improvements": ["list of improvements made"],
  "technical_keywords": ["technical terms added"],
  "confidence": confidence_score_0_to_1
}}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Optimize this prompt: {prompt}"}
            ]
            
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result_data = json.loads(content)
            
            return ToolResult(
                success=True,
                data=result_data,
                metadata={
                    "original_length": len(prompt),
                    "optimized_length": len(result_data.get("optimized_prompt", "")),
                    "generation_type": generation_type,
                    "quality_level": quality_level
                }
            )
            
        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            return ToolResult(
                success=False,
                error=f"Prompt optimization failed: {str(e)}"
            ) 