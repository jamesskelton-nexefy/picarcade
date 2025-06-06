"""
Prompt Parsing Agent for Phase 2

Uses GPT-4o to extract structured information from user prompts including:
- Intent classification
- Entity extraction (person, object, style, action, setting)
- Modifier identification (quality, style, lighting, mood, technical)
- Reference detection (celebrity, artwork, style, brand)
"""

import json
import logging
import time
from typing import Dict, Any, Optional
import os
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from ..types import (
    ParsedPrompt,
    PromptEntity,
    PromptModifier,
    PromptReference,
    PromptEntityType,
    PromptModifierType,
    PromptReferenceType,
    OpenAIConfig
)
from ..utils.decision_logger import decision_logger, DecisionType

logger = logging.getLogger(__name__)


class PromptParsingAgent:
    """
    Agent responsible for parsing user prompts using GPT-4o.
    
    Extracts structured information for downstream processing in the
    agentic workflow pipeline.
    """
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize the prompt parsing agent with OpenAI configuration."""
        self.config = config or OpenAIConfig(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            temperature=0.1,
            max_tokens=2000
        )
        
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
            
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.config.api_key)
        
        # LangChain client for structured responses
        self.llm = ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.api_key
        )
    
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

    async def parse_prompt(self, prompt: str) -> ParsedPrompt:
        """
        Parse a user prompt using GPT-4o and return structured information.
        
        Args:
            prompt: The user's input prompt
            
        Returns:
            ParsedPrompt: Structured information extracted from the prompt
            
        Raises:
            Exception: If parsing fails or API call fails
        """
        # Generate request ID for decision tracking
        request_id = f"parse_{int(time.time() * 1000)}"
        
        # Start decision tracking for this parsing operation
        decision_logger.start_decision(
            request_id=request_id,
            agent_name="PromptParsingAgent",
            initial_context={
                "prompt": prompt,
                "prompt_length": len(prompt),
                "parsing_model": self.config.model,
                "temperature": self.config.temperature
            }
        )
        
        start_time = time.time()
        
        try:
            logger.info(f"Parsing prompt: {prompt[:100]}...")
            
            # Log parsing strategy decision
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.PROMPT_PARSING,
                input_data={
                    "prompt": prompt,
                    "strategy": "gpt4o_structured_extraction"
                },
                decision_reasoning=f"Using GPT-4o with structured JSON response to extract intent, entities, modifiers, and references from {len(prompt)} character prompt",
                output_data={
                    "parsing_strategy": "structured_json",
                    "model": self.config.model,
                    "expected_output_types": ["intent", "entities", "modifiers", "references"]
                },
                confidence_score=0.9,
                metadata={
                    "prompt_word_count": len(prompt.split()),
                    "parsing_approach": "llm_structured"
                }
            )
            
            # Prepare messages for GPT-4o
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f"Parse this prompt: {prompt}"}
            ]
            
            # Log API call decision
            api_start_time = time.time()
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.PROMPT_PARSING,
                input_data={
                    "api_call": "openai_chat_completion",
                    "model": self.config.model,
                    "temperature": self.config.temperature
                },
                decision_reasoning="Making structured API call to GPT-4o with JSON response format to ensure consistent parsing output",
                output_data={"api_initiated": True},
                confidence_score=0.95,
                metadata={"response_format": "json_object"}
            )
            
            # Call GPT-4o API
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )
            
            api_time = (time.time() - api_start_time) * 1000
            
            # Extract JSON response
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from GPT-4o")
                
            logger.debug(f"GPT-4o response: {content}")
            
            # Log API response decision
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.PROMPT_PARSING,
                input_data={"raw_response_length": len(content)},
                decision_reasoning="Received structured JSON response from GPT-4o, proceeding to parse and validate",
                output_data={
                    "response_received": True,
                    "response_size": len(content),
                    "api_response_time_ms": api_time
                },
                confidence_score=0.9,
                execution_time_ms=api_time,
                metadata={"api_success": True}
            )
            
            # Parse JSON response
            try:
                parsed_data = json.loads(content)
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON response from GPT-4o: {e}"
                decision_logger.log_decision_step(
                    request_id=request_id,
                    decision_type=DecisionType.ERROR_HANDLING,
                    input_data={"json_error": str(e), "raw_content": content[:200]},
                    decision_reasoning="JSON parsing failed, this indicates malformed response from GPT-4o",
                    output_data={"fallback_strategy": "raise_exception"},
                    confidence_score=0.0,
                    error=error_msg
                )
                raise Exception(error_msg)
            
            # Log JSON parsing success
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.VALIDATION,
                input_data={"parsed_json_keys": list(parsed_data.keys())},
                decision_reasoning="Successfully parsed JSON response, proceeding to validate and convert to structured types",
                output_data={
                    "json_valid": True,
                    "has_intent": "intent" in parsed_data,
                    "has_entities": "entities" in parsed_data,
                    "has_modifiers": "modifiers" in parsed_data,
                    "has_references": "references" in parsed_data
                },
                confidence_score=0.95,
                metadata={"validation_step": "json_structure"}
            )
            
            # Validate and convert to structured types
            parsed_prompt = self._convert_to_parsed_prompt(parsed_data, prompt, request_id)
            
            total_execution_time = (time.time() - start_time) * 1000
            
            # Log successful parsing completion
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.PROMPT_PARSING,
                input_data={"conversion_complete": True},
                decision_reasoning=f"Successfully completed prompt parsing with {len(parsed_prompt.entities)} entities, {len(parsed_prompt.modifiers)} modifiers, and {len(parsed_prompt.references)} references",
                output_data={
                    "intent": parsed_prompt.intent,
                    "entities_extracted": len(parsed_prompt.entities),
                    "modifiers_extracted": len(parsed_prompt.modifiers),
                    "references_extracted": len(parsed_prompt.references),
                    "overall_confidence": parsed_prompt.confidence
                },
                confidence_score=parsed_prompt.confidence,
                execution_time_ms=total_execution_time,
                metadata={
                    "parsing_success": True,
                    "ready_for_downstream": True
                }
            )
            
            # Complete decision tracking
            decision_logger.complete_decision(
                request_id=request_id,
                final_result={
                    "intent": parsed_prompt.intent,
                    "total_extractions": len(parsed_prompt.entities) + len(parsed_prompt.modifiers) + len(parsed_prompt.references),
                    "confidence": parsed_prompt.confidence,
                    "execution_time_ms": total_execution_time
                },
                success=True
            )
            
            return parsed_prompt
            
        except json.JSONDecodeError as e:
            total_execution_time = (time.time() - start_time) * 1000
            error_msg = f"Failed to parse JSON response: {e}"
            
            decision_logger.complete_decision(
                request_id=request_id,
                final_result={"error": error_msg, "execution_time_ms": total_execution_time},
                success=False
            )
            
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            total_execution_time = (time.time() - start_time) * 1000
            error_msg = f"Prompt parsing failed: {e}"
            
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"error_type": type(e).__name__, "error_message": str(e)},
                decision_reasoning="Unexpected error during prompt parsing, terminating with exception",
                output_data={"recovery_action": "raise_exception"},
                confidence_score=0.0,
                execution_time_ms=total_execution_time,
                error=error_msg
            )
            
            decision_logger.complete_decision(
                request_id=request_id,
                final_result={"error": error_msg, "execution_time_ms": total_execution_time},
                success=False
            )
            
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def _convert_to_parsed_prompt(self, data: Dict[str, Any], original_prompt: str, request_id: str) -> ParsedPrompt:
        """Convert raw JSON data to ParsedPrompt with validation."""
        conversion_start_time = time.time()
        
        try:
            # Log conversion start
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.VALIDATION,
                input_data={"raw_data_keys": list(data.keys())},
                decision_reasoning="Starting conversion of raw JSON data to structured ParsedPrompt objects with type validation",
                output_data={"conversion_initiated": True},
                confidence_score=0.9,
                metadata={"validation_step": "type_conversion"}
            )
            
            # Convert entities
            entities = []
            entity_errors = []
            for i, entity_data in enumerate(data.get("entities", [])):
                try:
                    entity = PromptEntity(
                        text=entity_data["text"],
                        type=PromptEntityType(entity_data["type"]),
                        confidence=float(entity_data["confidence"])
                    )
                    entities.append(entity)
                except (KeyError, ValueError, TypeError) as e:
                    entity_errors.append(f"Entity {i}: {e}")
            
            # Convert modifiers
            modifiers = []
            modifier_errors = []
            for i, modifier_data in enumerate(data.get("modifiers", [])):
                try:
                    modifier = PromptModifier(
                        text=modifier_data["text"],
                        type=PromptModifierType(modifier_data["type"]),
                        value=modifier_data.get("value"),
                        confidence=float(modifier_data["confidence"])
                    )
                    modifiers.append(modifier)
                except (KeyError, ValueError, TypeError) as e:
                    modifier_errors.append(f"Modifier {i}: {e}")
            
            # Convert references
            references = []
            reference_errors = []
            for i, reference_data in enumerate(data.get("references", [])):
                try:
                    reference = PromptReference(
                        text=reference_data["text"],
                        type=PromptReferenceType(reference_data["type"]),
                        search_query=reference_data["search_query"],
                        confidence=float(reference_data["confidence"])
                    )
                    references.append(reference)
                except (KeyError, ValueError, TypeError) as e:
                    reference_errors.append(f"Reference {i}: {e}")
            
            # Log validation results
            conversion_time = (time.time() - conversion_start_time) * 1000
            all_errors = entity_errors + modifier_errors + reference_errors
            
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.VALIDATION,
                input_data={
                    "entities_attempted": len(data.get("entities", [])),
                    "modifiers_attempted": len(data.get("modifiers", [])),
                    "references_attempted": len(data.get("references", []))
                },
                decision_reasoning=f"Converted structured data with {len(entities)} entities, {len(modifiers)} modifiers, {len(references)} references. {len(all_errors)} conversion errors encountered.",
                output_data={
                    "entities_converted": len(entities),
                    "modifiers_converted": len(modifiers),
                    "references_converted": len(references),
                    "conversion_errors": len(all_errors),
                    "conversion_success_rate": 1.0 - (len(all_errors) / max(1, len(data.get("entities", [])) + len(data.get("modifiers", [])) + len(data.get("references", []))))
                },
                confidence_score=0.8 if not all_errors else 0.6,
                execution_time_ms=conversion_time,
                metadata={
                    "validation_errors": all_errors[:5] if all_errors else [],  # Log first 5 errors
                    "has_conversion_errors": len(all_errors) > 0
                }
            )
            
            # Create ParsedPrompt
            parsed_prompt = ParsedPrompt(
                intent=data.get("intent", "generate_image"),
                entities=entities,
                modifiers=modifiers,
                references=references,
                confidence=float(data.get("confidence", 0.5))
            )
            
            logger.info(f"Successfully parsed prompt with {len(entities)} entities, "
                       f"{len(modifiers)} modifiers, {len(references)} references")
            
            return parsed_prompt
            
        except (KeyError, ValueError, TypeError) as e:
            error_msg = f"Failed to convert parsed data: {e}"
            conversion_time = (time.time() - conversion_start_time) * 1000
            
            # Log conversion failure
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"conversion_error": str(e)},
                decision_reasoning="Critical conversion error, falling back to basic prompt structure to maintain system stability",
                output_data={
                    "fallback_intent": "generate_image",
                    "fallback_confidence": 0.1
                },
                confidence_score=0.0,
                execution_time_ms=conversion_time,
                error=error_msg,
                metadata={"recovery_strategy": "basic_fallback"}
            )
            
            logger.error(error_msg)
            # Return basic fallback
            return ParsedPrompt(
                intent="generate_image",
                entities=[],
                modifiers=[],
                references=[],
                confidence=0.1
            )
    
    async def parse_batch(self, prompts: list[str]) -> list[ParsedPrompt]:
        """
        Parse multiple prompts in batch for testing.
        
        Args:
            prompts: List of prompt strings
            
        Returns:
            List of ParsedPrompt objects
        """
        results = []
        for prompt in prompts:
            try:
                result = await self.parse_prompt(prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to parse prompt '{prompt[:50]}...': {e}")
                # Add fallback result
                results.append(ParsedPrompt(
                    intent="generate_image",
                    entities=[],
                    modifiers=[],
                    references=[],
                    confidence=0.0
                ))
        return results 