"""
Workflow Planning Tools for Pic Arcade

Tools that handle workflow planning, orchestration, and dynamic tool selection.
Core to the tool-first architecture where agents reason about which tools to use.
"""

import logging
from typing import Dict, Any, List, Optional
import json
from openai import AsyncOpenAI
import os

from .base import Tool, ToolCategory, ToolResult, tool_registry

logger = logging.getLogger(__name__)


class WorkflowPlanningTool(Tool):
    """
    Tool for planning multi-step workflows based on user requests.
    
    Uses GPT-4o to analyze a user's request and determine the sequence of tools
    needed to fulfill it, following the agentic pattern described in the guide.
    Enhanced with conversation context awareness for multi-turn editing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="workflow_planner",
            description="Plan multi-step workflows by selecting and sequencing appropriate tools with conversation context awareness",
            category=ToolCategory.WORKFLOW_PLANNING,
            input_schema={
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "The user's request to plan a workflow for"
                    },
                    "available_tools": {
                        "type": "array",
                        "description": "List of available tools and their descriptions",
                        "items": {"type": "object"}
                    },
                    "conversation_context": {
                        "type": "object",
                        "description": "Conversation context for multi-turn interactions",
                        "properties": {
                            "is_edit": {"type": "boolean"},
                            "edit_type": {"type": "string"},
                            "has_target_image": {"type": "boolean"},
                            "target_image_url": {"type": "string"},
                            "original_prompt": {"type": "string"},
                            "edit_instructions": {"type": "string"}
                        }
                    },
                    "constraints": {
                        "type": "object",
                        "properties": {
                            "max_steps": {"type": "integer", "default": 10},
                            "budget": {"type": "number"},
                            "time_limit": {"type": "number"}
                        }
                    }
                },
                "required": ["user_request"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "workflow_plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "integer"},
                                "tool_name": {"type": "string"},
                                "description": {"type": "string"},
                                "inputs": {"type": "object"},
                                "expected_output": {"type": "string"},
                                "dependencies": {"type": "array"}
                            }
                        }
                    },
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number"},
                    "estimated_time": {"type": "number"},
                    "estimated_cost": {"type": "number"}
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
            raise ValueError("OpenAI API key is required for WorkflowPlanningTool")
        
        # Ensure we don't use a Replicate key for OpenAI
        if api_key.startswith("r8_"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required for WorkflowPlanningTool, but only Replicate key found")
        
        self.config["openai_api_key"] = api_key
        
        # Initialize the OpenAI client with the validated key
        self.client = AsyncOpenAI(api_key=api_key)
    
    def _get_system_prompt(self, available_tools: List[Dict[str, Any]], conversation_context: Dict[str, Any] = None) -> str:
        """Get context-aware system prompt for workflow planning."""
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}\n  Category: {tool.get('category', 'General')}\n  Required inputs: {tool['input_schema'].get('required', [])}\n  Input properties: {list(tool['input_schema'].get('properties', {}).keys())}" 
            for tool in available_tools
        ])
        
        # Context-aware instructions
        context_instructions = ""
        if conversation_context and conversation_context.get("is_edit"):
            context_instructions = f"""
CRITICAL: This is an EDIT REQUEST with conversation context!
- Original image available: {conversation_context.get('target_image_url', 'N/A')}
- Edit type: {conversation_context.get('edit_type', 'unknown')}
- Original prompt: {conversation_context.get('original_prompt', 'N/A')}
- Edit instructions: {conversation_context.get('edit_instructions', 'N/A')}

FOR EDIT REQUESTS:
1. ALWAYS use "flux_image_manipulation" for image-to-image editing (NOT object_change)
2. ALWAYS include the original image URL in tool inputs
3. Create a workflow that edits the existing image, not generates new ones
4. Use the combined context (original + edit instructions) for better results

EDIT WORKFLOW PATTERN:
- Step 1: prompt_parser → parse edit instructions
- Step 2: flux_image_manipulation → with original image + edit prompt
"""
        else:
            context_instructions = """
GENERATION REQUEST (No edit context detected):
- This appears to be a new image generation request
- Use standard generation workflows
- No original image context available
"""
        
        return f"""You are an expert workflow planner for an AI image/video generation platform with conversation context awareness.

{context_instructions}

Your task is to analyze user requests and create step-by-step plans using available tools.

AVAILABLE TOOLS:
{tools_description}

WORKFLOW PLANNING PRINCIPLES:
1. Break down complex requests into logical steps
2. Chain tools together where output of one feeds into another
3. Consider dependencies and proper sequencing
4. Use EXACT tool names and input field names from each tool's schema
5. Optimize for efficiency and quality
6. Handle error cases and fallbacks
7. RESPECT conversation context for multi-turn interactions

TOOL NAMING GUIDE:
- Use "prompt_parser" for parsing prompts
- Use "prompt_optimizer" for optimizing prompts
        - Use "flux_image_manipulation" for ALL image editing and manipulation operations (REQUIRED for edits)
        - Use "flux_image_generation" for pure image generation from text prompts (new images only)
- Use "style_transfer" for art style conversion with reference images
- Use "object_change" ONLY for simple object modifications without existing image context
- Use "text_editing" for replacing text in images
- Use "background_swap" for changing backgrounds
- Use "character_consistency" for maintaining character identity
- Use "runway_video" for premium high-quality video generation (Runway ML)
- Use "replicate_video" for video generation using multiple providers (Google Veo 2, Luma Ray, etc.)
- Use "video_editing" for video editing, enhancement, upscaling, and style transfer
- Use "perplexity_search" for web search
- Use "dalle_image_generation" as fallback image generation

INPUT FIELD MAPPING:
- prompt_parser: "prompt" (required)
- prompt_optimizer: "prompt" (required)
- flux_image_manipulation: "prompt" (required), "operation_type", "image" (for editing operations)
- flux_image_generation: "prompt" (required)
- style_transfer: "image" (required), "style" (required), "prompt"
- object_change: "image" (required), "target_object" (required), "modification" (required)
- text_editing: "image" (required), "new_text" (required)
- background_swap: "image" (required), "new_background" (required)
- character_consistency: "reference_image" (required), "character_description" (required), "new_scenario" (required)
- runway_video: "prompt_text" (required), "model", "ratio", "duration", "prompt_image" (for image-to-video)
- replicate_video: "prompt" (required), "provider", "duration", "quality", "aspect_ratio", "image" (for image-to-video)
- video_editing: "video_url" (required), "operation" (required), "style_prompt", "enhancement_type"
- perplexity_search: "query" (required)

CONTEXT-AWARE WORKFLOW PATTERNS:

1. EDIT EXISTING IMAGE (conversation context available):
   - Step 1: prompt_parser → parse edit instructions
   - Step 2: flux_image_manipulation → with original image + edit prompt
   
2. NEW IMAGE GENERATION (no context):
   - Step 1: prompt_parser → parse creation request
   - Step 2: flux_image_generation → generate new image

3. STYLE TRANSFER (with reference):
   - Step 1: prompt_parser → parse style request
   - Step 2: style_transfer → apply style to existing image

4. COMPLEX EDITING:
   - Step 1: prompt_parser → analyze requirements
   - Step 2: flux_image_manipulation → perform primary edit
   - Step 3: Additional tools if needed

5. VIDEO GENERATION (new video from text):
   - Step 1: prompt_parser → parse video generation request
   - Step 2: prompt_optimizer → optimize for video generation
   - Step 3: replicate_video → generate video using "google_veo2" provider (state-of-the-art)
   IMPORTANT: Use "$step_2" for prompt input - system will auto-extract optimized_prompt

6. PREMIUM VIDEO GENERATION (high quality):
   - Step 1: prompt_parser → parse video request
   - Step 2: runway_video → generate with Runway ML
   IMPORTANT: Use "$step_1" for prompt_text input - system will auto-extract from parsed data

7. IMAGE-TO-VIDEO ANIMATION:
   - Step 1: prompt_parser → parse animation request
   - Step 2: replicate_video → animate image with "image" parameter and "$step_1" for prompt
   IMPORTANT: Provide both "image" URL and prompt for best results

REFERENCE HANDLING:
- Use $user_request to reference the original user input
- Use $step_N to reference output from step N
- Use $step_N.field_name to reference specific fields from step N output
- For edit operations, use the provided original image URL directly

CRITICAL EDITING RULES:
- If conversation_context.is_edit is true, this is ALWAYS an edit operation
- NEVER use object_change for edit operations when original image is available
- ALWAYS use flux_image_manipulation for image-to-image editing
- ALWAYS pass the original image URL to editing tools
- Combine original prompt with edit instructions for best results

CRITICAL: You MUST return a JSON object with this EXACT structure:
{{
  "workflow_plan": [
    {{
      "step": 1,
      "tool_name": "exact_tool_name_from_available_tools",
      "description": "what this step does",
      "inputs": {{"field_name": "value_or_reference"}},
      "expected_output": "what this step produces",
      "dependencies": ["$step_references"]
    }}
  ],
  "reasoning": "explanation of the workflow",
  "confidence": 0.9,
  "estimated_time": 30
}}

DO NOT return a single step object directly. DO NOT return an array directly. 
ALWAYS wrap the steps in a "workflow_plan" array within a JSON object.
The "workflow_plan" key is REQUIRED and must contain an array of step objects.

Example for EDIT operation (COMPLETE JSON object):
{{
  "workflow_plan": [
    {{
      "step": 1,
      "tool_name": "prompt_parser",
      "description": "Parse edit instructions to understand modification requirements",
      "inputs": {{"prompt": "$user_request"}},
      "expected_output": "Parsed edit intent and modification details",
      "dependencies": []
    }},
    {{
      "step": 2,
      "tool_name": "flux_image_manipulation",
      "description": "Apply edit to original image using image-to-image transformation",
      "inputs": {{
        "prompt": "COMBINED_ORIGINAL_AND_EDIT_PROMPT_HERE",
        "image": "ORIGINAL_IMAGE_URL_FROM_CONTEXT",
        "operation_type": "edit"
      }},
      "expected_output": "Modified image with requested changes applied",
      "dependencies": ["$step_1"]
    }}
  ],
  "reasoning": "Context-aware workflow for editing existing image",
  "confidence": 0.9,
  "estimated_time": 45
}}"""

    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Create a workflow plan for a user request.
        
        Args:
            input_data: Contains user request and available tools
            
        Returns:
            ToolResult with workflow plan
        """
        try:
            user_request = input_data["user_request"]
            conversation_context = input_data.get("conversation_context", {})
            
            logger.info("=" * 80)
            logger.info("🧠 WORKFLOW PLANNING STARTING")
            logger.info("=" * 80)
            logger.info(f"🎯 USER REQUEST: '{user_request}'")
            logger.info(f"🔄 CONVERSATION CONTEXT: {conversation_context}")
            
            # Get available tools from registry if not provided
            available_tools = input_data.get("available_tools")
            if not available_tools:
                available_tools = [
                    tool.get_metadata() for tool in tool_registry._tools.values()
                ]
            
            logger.info(f"🛠️  AVAILABLE TOOLS: {[tool['name'] for tool in available_tools]}")
            
            # Create context-aware prompt
            system_prompt = self._get_system_prompt(available_tools, conversation_context)
            
            # Prepare context-enhanced user message
            context_info = ""
            if conversation_context.get("is_edit"):
                context_info = f"""
CONTEXT INFORMATION:
- This is an EDIT operation of an existing image
- Original image: {conversation_context.get('target_image_url', 'Available')}
- Original prompt: {conversation_context.get('original_prompt', 'N/A')}
- Edit type: {conversation_context.get('edit_type', 'general_edit')}

IMPORTANT: Use flux_image_manipulation for image-to-image editing with the original image!
"""
                logger.info(f"📝 EDIT CONTEXT INFO: {context_info}")
            
            user_message = f"{context_info}Plan a workflow for: {user_request}"
            
            logger.info("📤 SENDING TO GPT-4o:")
            logger.info(f"   System prompt length: {len(system_prompt)} chars")
            logger.info(f"   User message: '{user_message}'")
            
            # Prepare messages for GPT-4o
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            logger.info(f"🤖 CALLING GPT-4o for workflow planning...")
            
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,  # Lower temperature for more consistent context-aware planning
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                logger.error("❌ Empty response from GPT-4o")
                return ToolResult(
                    success=False,
                    error="Empty response from GPT-4o"
                )
            
            logger.info(f"📨 RAW GPT-4o RESPONSE: {content}")
            
            raw_plan_data = json.loads(content)
            logger.info(f"📋 PARSED PLAN DATA: {raw_plan_data}")
            
            # Normalize the response to handle different GPT-4o response formats
            plan_data = self._normalize_gpt4o_response(raw_plan_data)
            logger.info(f"📋 NORMALIZED PLAN DATA: {plan_data}")
            
            # Post-process the plan to ensure context compliance
            if conversation_context.get("is_edit"):
                logger.info("🔧 POST-PROCESSING for edit compliance...")
                plan_data = self._ensure_edit_compliance(plan_data, conversation_context)
                logger.info(f"📋 POST-PROCESSED PLAN: {plan_data}")
            
            # Validate the workflow plan before returning
            workflow_plan = plan_data.get("workflow_plan", [])
            if not workflow_plan:
                error_msg = "Workflow planning resulted in empty workflow plan"
                logger.error(f"❌ {error_msg}")
                logger.error(f"📋 Plan data: {plan_data}")
                return ToolResult(
                    success=False,
                    error=error_msg,
                    data=plan_data  # Include the malformed data for debugging
                )
            
            # Log the final workflow steps
            logger.info("📝 FINAL WORKFLOW STEPS:")
            for step in workflow_plan:
                logger.info(f"   Step {step.get('step', '?')}: {step.get('tool_name', '?')} - {step.get('description', '?')}")
                logger.info(f"      Inputs: {step.get('inputs', {})}")
            
            logger.info("=" * 80)
            logger.info("✅ WORKFLOW PLANNING COMPLETED")
            logger.info("=" * 80)
            
            return ToolResult(
                success=True,
                data=plan_data,
                metadata={
                    "user_request": user_request,
                    "tools_considered": len(available_tools),
                    "steps_planned": len(plan_data.get("workflow_plan", [])),
                    "planning_model": "gpt-4o",
                    "context_aware": bool(conversation_context),
                    "is_edit_operation": conversation_context.get("is_edit", False),
                    "has_original_image": conversation_context.get("has_target_image", False)
                }
            )
            
        except Exception as e:
            logger.error(f"💥 WORKFLOW PLANNING FAILED: {e}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return ToolResult(
                success=False,
                error=f"Workflow planning failed: {str(e)}"
            )
    
    def _normalize_gpt4o_response(self, raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize GPT-4o response to handle different response formats.
        
        GPT-4o sometimes returns malformed responses that need to be normalized:
        1. Expected format: {"workflow_plan": [{"step": 1, ...}, {"step": 2, ...}]}
        2. Malformed single step: {"step": 1, "tool_name": "...", ...}
        3. Malformed multiple steps: [{"step": 1, ...}, {"step": 2, ...}]
        4. Partial response missing workflow_plan wrapper
        
        Args:
            raw_response: Raw response from GPT-4o
            
        Returns:
            Normalized response with proper workflow_plan structure
        """
        logger.info(f"🔧 NORMALIZING GPT-4o RESPONSE...")
        
        # Case 1: Already in correct format
        if "workflow_plan" in raw_response and isinstance(raw_response["workflow_plan"], list):
            logger.info(f"   ✅ Response already in correct format with {len(raw_response['workflow_plan'])} steps")
            return raw_response
        
        # Case 2: Single step object returned directly (malformed)
        if "step" in raw_response and "tool_name" in raw_response:
            logger.info(f"   🔧 Detected single step object, wrapping in workflow_plan array")
            normalized = {
                "workflow_plan": [raw_response],
                "reasoning": f"Single step workflow for: {raw_response.get('description', 'N/A')}",
                "confidence": 0.8,
                "estimated_time": 30
            }
            logger.info(f"   📦 Wrapped single step into workflow_plan array")
            return normalized
        
        # Case 3: Array of steps returned directly (missing wrapper)
        if isinstance(raw_response, list):
            logger.info(f"   🔧 Detected array of steps, wrapping in workflow_plan object")
            normalized = {
                "workflow_plan": raw_response,
                "reasoning": f"Multi-step workflow with {len(raw_response)} steps",
                "confidence": 0.8,
                "estimated_time": len(raw_response) * 20
            }
            logger.info(f"   📦 Wrapped {len(raw_response)} steps into workflow_plan object")
            return normalized
        
        # Case 4: Response has workflow_plan key but it's not a list
        if "workflow_plan" in raw_response and not isinstance(raw_response["workflow_plan"], list):
            logger.info(f"   🔧 Detected workflow_plan that's not a list, converting to list")
            plan_value = raw_response["workflow_plan"]
            if isinstance(plan_value, dict) and "step" in plan_value:
                raw_response["workflow_plan"] = [plan_value]
                logger.info(f"   📦 Converted single workflow_plan object to array")
            else:
                logger.warning(f"   ⚠️  Unexpected workflow_plan format: {type(plan_value)}, creating empty plan")
                raw_response["workflow_plan"] = []
            return raw_response
        
        # Case 5: Look for step-like objects in the response
        potential_steps = []
        for key, value in raw_response.items():
            if isinstance(value, dict) and "tool_name" in value:
                logger.info(f"   🔧 Found potential step in key '{key}': {value.get('tool_name')}")
                # Add step number if missing
                if "step" not in value:
                    value["step"] = len(potential_steps) + 1
                potential_steps.append(value)
        
        if potential_steps:
            logger.info(f"   📦 Found {len(potential_steps)} potential steps, creating workflow_plan")
            normalized = {
                "workflow_plan": potential_steps,
                "reasoning": f"Reconstructed workflow from {len(potential_steps)} detected steps",
                "confidence": 0.7,
                "estimated_time": len(potential_steps) * 20
            }
            # Preserve other fields from original response
            for key, value in raw_response.items():
                if key not in normalized and not isinstance(value, dict):
                    normalized[key] = value
            return normalized
        
        # Case 6: Empty or completely malformed response - create minimal valid structure
        logger.warning(f"   ⚠️  Could not normalize response, creating minimal workflow")
        logger.warning(f"   📋 Raw response keys: {list(raw_response.keys())}")
        
        # Try to create a basic workflow if we can extract any useful information
        fallback_workflow = {
            "workflow_plan": [],
            "reasoning": "Unable to parse GPT-4o response, created fallback empty workflow",
            "confidence": 0.1,
            "estimated_time": 0,
            "error": "GPT-4o response normalization failed",
            "original_response": raw_response
        }
        
        logger.warning(f"   📦 Created fallback empty workflow due to parsing failure")
        return fallback_workflow

    def _ensure_edit_compliance(self, plan_data: Dict[str, Any], conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the workflow plan properly handles edit operations with original image context.
        
        Args:
            plan_data: Generated workflow plan
            conversation_context: Conversation context with edit information
            
        Returns:
            Modified plan_data that properly handles edits
        """
        workflow_plan = plan_data.get("workflow_plan", [])
        target_image_url = conversation_context.get("target_image_url")
        original_prompt = conversation_context.get("original_prompt", "")
        edit_instructions = conversation_context.get("edit_instructions", "")
        
        # Find steps that should use the original image
        for step in workflow_plan:
            tool_name = step.get("tool_name", "")
            
            # If using flux_image_manipulation, ensure it has the original image and combined prompt
            if tool_name == "flux_image_manipulation":
                inputs = step.get("inputs", {})
                
                # Ensure original image is included
                if target_image_url and "image" not in inputs:
                    inputs["image"] = target_image_url
                    step["inputs"] = inputs
                
                # Enhance prompt with context
                if "prompt" in inputs and original_prompt:
                    current_prompt = inputs["prompt"]
                    if current_prompt == "$user_request" or edit_instructions in current_prompt:
                        # Create combined prompt for better editing context
                        combined_prompt = f"{original_prompt}, modified to {edit_instructions}"
                        inputs["prompt"] = combined_prompt
                        step["inputs"] = inputs
                
                # Ensure operation type is set for editing
                if "operation_type" not in inputs:
                    inputs["operation_type"] = "edit"
                    step["inputs"] = inputs
            
            # Replace object_change with flux_image_manipulation for edit operations
            elif tool_name == "object_change" and target_image_url:
                step["tool_name"] = "flux_image_manipulation"
                step["description"] = "Modify the image using advanced image-to-image editing"
                
                # Convert object_change inputs to flux_image_manipulation format
                inputs = step.get("inputs", {})
                
                # Create appropriate prompt from object_change parameters
                target_object = inputs.get("target_object", "object")
                modification = inputs.get("modification", edit_instructions)
                edit_prompt = f"{original_prompt}, {modification} the {target_object}"
                
                step["inputs"] = {
                    "prompt": edit_prompt,
                    "image": target_image_url,
                    "operation_type": "edit"
                }
                
                step["expected_output"] = "Modified image with requested changes applied using image-to-image editing"
        
        # Update reasoning to reflect context awareness
        plan_data["reasoning"] = f"Context-aware workflow for editing existing image. Original: '{original_prompt}' + Edit: '{edit_instructions}'"
        
        # Increase confidence for context-aware planning
        if "confidence" in plan_data:
            plan_data["confidence"] = min(plan_data["confidence"] + 0.1, 1.0)
        
        return plan_data


class WorkflowExecutorTool(Tool):
    """
    Tool for executing planned workflows step by step.
    
    Takes a workflow plan and executes each step, managing dependencies
    and data flow between tools.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="workflow_executor",
            description="Execute multi-step workflows by invoking tools in sequence",
            category=ToolCategory.WORKFLOW_PLANNING,
            input_schema={
                "type": "object",
                "properties": {
                    "workflow_plan": {
                        "type": "array",
                        "description": "Workflow plan to execute"
                    },
                    "initial_inputs": {
                        "type": "object",
                        "description": "Initial inputs for the workflow"
                    },
                    "execution_options": {
                        "type": "object",
                        "properties": {
                            "stop_on_error": {"type": "boolean", "default": True},
                            "max_retries": {"type": "integer", "default": 3},
                            "timeout_per_step": {"type": "number", "default": 60}
                        }
                    }
                },
                "required": ["workflow_plan"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "execution_results": {"type": "array"},
                    "final_outputs": {"type": "object"},
                    "execution_status": {"type": "string"},
                    "total_time": {"type": "number"},
                    "errors": {"type": "array"}
                }
            },
            config=config
        )
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        pass  # No specific config needed
    
    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Execute a workflow plan step by step.
        
        Args:
            input_data: Contains workflow plan and execution options
            
        Returns:
            ToolResult with execution results
        """
        try:
            workflow_plan = input_data["workflow_plan"]
            initial_inputs = input_data.get("initial_inputs", {})
            options = input_data.get("execution_options", {})
            
            # Log the complete workflow plan and initial inputs
            logger.info("=" * 80)
            logger.info("🚀 WORKFLOW EXECUTION STARTING")
            logger.info("=" * 80)
            logger.info(f"📋 WORKFLOW PLAN ({len(workflow_plan)} steps):")
            for step in workflow_plan:
                logger.info(f"   Step {step['step']}: {step['tool_name']} - {step['description']}")
                logger.info(f"      Inputs: {step['inputs']}")
                logger.info(f"      Expected: {step['expected_output']}")
            logger.info("=" * 80)
            logger.info(f"🎯 INITIAL INPUTS: {initial_inputs}")
            logger.info("=" * 80)
            
            execution_results = []
            step_outputs = {}
            errors = []
            
            import time
            start_time = time.time()
            
            # Execute each step in sequence
            for step_info in workflow_plan:
                step_num = step_info["step"]
                tool_name = step_info["tool_name"]
                step_inputs = step_info["inputs"]
                
                logger.info(f"\n🔧 EXECUTING STEP {step_num}: {tool_name}")
                logger.info(f"   📥 Raw inputs from plan: {step_inputs}")
                
                try:
                    # Resolve input dependencies
                    resolved_inputs = self._resolve_inputs(
                        step_inputs, step_outputs, initial_inputs
                    )
                    
                    logger.info(f"   ⚙️  Resolved inputs: {resolved_inputs}")
                    
                    # Validate that all required inputs are present
                    tool = tool_registry.get_tool(tool_name)
                    if tool:
                        required_fields = tool.input_schema.get("required", [])
                        missing_fields = [field for field in required_fields if field not in resolved_inputs]
                        
                        if missing_fields:
                            error_msg = f"Missing required fields for {tool_name}: {missing_fields}"
                            logger.warning(f"   ⚠️  {error_msg}")
                            
                            # Try to provide reasonable defaults for common missing fields
                            for field in missing_fields:
                                if field == "prompt" and "user_request" in initial_inputs:
                                    resolved_inputs[field] = initial_inputs["user_request"]
                                    logger.info(f"   🔧 Auto-filled 'prompt' with user_request: {resolved_inputs[field]}")
                                elif field == "query" and "user_request" in initial_inputs:
                                    resolved_inputs[field] = initial_inputs["user_request"]
                                    logger.info(f"   🔧 Auto-filled 'query' with user_request: {resolved_inputs[field]}")
                                else:
                                    resolved_inputs[field] = f"Auto-generated for {field}"
                                    logger.info(f"   🔧 Auto-filled '{field}' with default: {resolved_inputs[field]}")
                            
                            logger.info(f"   ✅ Final resolved inputs after auto-fill: {resolved_inputs}")
                    
                    # Log what we're about to send to the tool
                    logger.info(f"   📤 SENDING TO {tool_name}: {resolved_inputs}")
                    
                    # Execute the tool
                    result = await tool_registry.invoke_tool(tool_name, resolved_inputs)
                    
                    # Log the tool result
                    logger.info(f"   📨 RESULT FROM {tool_name}:")
                    logger.info(f"      Success: {result.success}")
                    if result.success:
                        logger.info(f"      Data: {result.data}")
                        if result.data and isinstance(result.data, dict):
                            # Highlight important data fields
                            for key, value in result.data.items():
                                if key in ['images', 'image_url', 'styled_image', 'optimized_prompt', 'parsed_prompt']:
                                    logger.info(f"      🎯 Key output '{key}': {value}")
                    else:
                        logger.error(f"      ❌ Error: {result.error}")
                    
                    # Store result with better key structure for both simple and complex references
                    step_key = f"step_{step_num}"
                    
                    # Store the complete result data in multiple accessible ways
                    if result.success and result.data:
                        step_outputs[step_key] = result.data
                        
                        # Also store common fields for easy access
                        if isinstance(result.data, dict):
                            # For prompt tools, store common outputs
                            if "parsed_prompt" in result.data:
                                step_outputs[f"{step_key}.parsed_prompt"] = result.data["parsed_prompt"]
                                logger.info(f"   📦 Stored: {step_key}.parsed_prompt = {result.data['parsed_prompt']}")
                            if "optimized_prompt" in result.data:
                                step_outputs[f"{step_key}.optimized_prompt"] = result.data["optimized_prompt"]
                                logger.info(f"   📦 Stored: {step_key}.optimized_prompt = {result.data['optimized_prompt']}")
                            if "images" in result.data:
                                step_outputs[f"{step_key}.images"] = result.data["images"]
                                logger.info(f"   📦 Stored: {step_key}.images = {result.data['images']}")
                            if "styled_image" in result.data:
                                step_outputs[f"{step_key}.styled_image"] = result.data["styled_image"]
                                logger.info(f"   📦 Stored: {step_key}.styled_image = {result.data['styled_image']}")
                            
                            # Store any string values that could be prompts
                            for key, value in result.data.items():
                                if isinstance(value, str) and len(value) > 0:
                                    step_outputs[f"{step_key}.{key}"] = value
                                    logger.info(f"   📦 Stored: {step_key}.{key} = {value}")
                    else:
                        step_outputs[step_key] = {"error": result.error}
                        logger.error(f"   📦 Stored error: {step_key} = {result.error}")
                    
                    logger.info(f"   💾 CURRENT STEP OUTPUTS: {list(step_outputs.keys())}")
                    
                    execution_results.append({
                        "step": step_num,
                        "tool_name": tool_name,
                        "success": result.success,
                        "data": result.data,
                        "error": result.error,
                        "execution_time": getattr(result, 'execution_time', 0),
                        "resolved_inputs": resolved_inputs
                    })
                    
                    # Check for errors
                    if not result.success:
                        error_msg = f"Step {step_num} failed: {result.error}"
                        errors.append(error_msg)
                        logger.error(f"   ❌ STEP FAILED: {error_msg}")
                        
                        if options.get("stop_on_error", True):
                            logger.error(f"   🛑 STOPPING execution due to error")
                            break
                    else:
                        logger.info(f"   ✅ STEP {step_num} COMPLETED SUCCESSFULLY")
                    
                except Exception as e:
                    error_msg = f"Step {step_num} exception: {str(e)}"
                    errors.append(error_msg)
                    logger.error(f"   💥 STEP EXCEPTION: {error_msg}")
                    import traceback
                    logger.error(f"   📋 Traceback: {traceback.format_exc()}")
                    
                    if options.get("stop_on_error", True):
                        logger.error(f"   🛑 STOPPING execution due to exception")
                        break
            
            total_time = time.time() - start_time
            
            # Determine final status
            if errors and options.get("stop_on_error", True):
                status = "failed"
            elif errors:
                status = "completed_with_errors"
            else:
                status = "completed"
            
            logger.info("=" * 80)
            logger.info("🏁 WORKFLOW EXECUTION COMPLETED")
            logger.info(f"   📊 Status: {status}")
            logger.info(f"   ⏱️  Total time: {total_time:.2f}s")
            logger.info(f"   ✅ Steps executed: {len(execution_results)}")
            logger.info(f"   ❌ Errors: {len(errors)}")
            if errors:
                for error in errors:
                    logger.error(f"      - {error}")
            logger.info(f"   📦 Final outputs: {list(step_outputs.keys())}")
            logger.info("=" * 80)
            
            return ToolResult(
                success=True,
                data={
                    "execution_results": execution_results,
                    "final_outputs": step_outputs,
                    "execution_status": status,
                    "total_time": total_time,
                    "errors": errors
                },
                metadata={
                    "steps_executed": len(execution_results),
                    "steps_planned": len(workflow_plan),
                    "success_rate": (len(execution_results) - len(errors)) / len(workflow_plan) if workflow_plan else 0
                }
            )
            
        except Exception as e:
            logger.error(f"💥 WORKFLOW EXECUTION FAILED: {e}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return ToolResult(
                success=False,
                error=f"Workflow execution failed: {str(e)}"
            )
    
    def _resolve_inputs(
        self, 
        step_inputs: Dict[str, Any], 
        step_outputs: Dict[str, Any], 
        initial_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve input dependencies by substituting references to previous outputs.
        
        Args:
            step_inputs: Template inputs for this step
            step_outputs: Outputs from previous steps
            initial_inputs: Initial workflow inputs
            
        Returns:
            Resolved inputs with dependencies substituted
        """
        logger.info(f"   🔍 RESOLVING INPUTS:")
        logger.info(f"      Template inputs: {step_inputs}")
        logger.info(f"      Available step outputs: {list(step_outputs.keys())}")
        logger.info(f"      Available initial inputs: {list(initial_inputs.keys())}")
        
        resolved_inputs = {}
        
        for key, value in step_inputs.items():
            logger.info(f"      🎯 Resolving '{key}' = '{value}'")
            
            if isinstance(value, str):
                # Handle step references like $step_1, $step_2.output_key
                if value.startswith("$"):
                    reference = value[1:]  # Remove $
                    logger.info(f"         🔧 Step reference: '{reference}'")
                    
                    if "." in reference:
                        # Complex reference like step_2.images
                        step_key, output_key = reference.split(".", 1)
                        logger.info(f"         📦 Complex reference: step='{step_key}', key='{output_key}'")
                        
                        if step_key in step_outputs:
                            step_data = step_outputs[step_key]
                            if output_key in step_data:
                                resolved_value = step_data[output_key]
                                logger.info(f"         ✅ Found complex value: '{resolved_value}'")
                            else:
                                resolved_value = step_data
                                logger.info(f"         ⚠️  Key '{output_key}' not found, using step data: '{resolved_value}'")
                        else:
                            resolved_value = value
                            logger.info(f"         ❌ Step '{step_key}' not found, keeping original: '{resolved_value}'")
                    else:
                        # Simple step reference like $step_1
                        if reference in step_outputs:
                            step_data = step_outputs[reference]
                            logger.info(f"         📦 Found step data: {step_data}")
                            
                            # Smart extraction for different data types
                            if isinstance(step_data, dict):
                                # If this is parsed prompt data, extract the meaningful text
                                if 'entities' in step_data and 'intent' in step_data:
                                    # Extract entities and construct a meaningful prompt
                                    entities = step_data.get('entities', [])
                                    modifiers = step_data.get('modifiers', [])
                                    
                                    logger.info(f"         📋 Parsing prompt data with {len(entities)} entities and {len(modifiers)} modifiers")
                                    
                                    if entities or modifiers:
                                        # Extract different types of entities in logical order
                                        objects = [e.get('text', '') for e in entities if e.get('type') == 'object' and e.get('text')]
                                        settings = [e.get('text', '') for e in entities if e.get('type') == 'setting' and e.get('text')]
                                        actions = [e.get('text', '') for e in entities if e.get('type') == 'action' and e.get('text')]
                                        styles = [e.get('text', '') for e in entities if e.get('type') == 'style' and e.get('text')]
                                        persons = [e.get('text', '') for e in entities if e.get('type') == 'person' and e.get('text')]
                                        # Catch any other entity types we might have missed
                                        other_entities = [e.get('text', '') for e in entities if e.get('type') not in ['object', 'setting', 'action', 'style', 'person'] and e.get('text')]
                                        
                                        # Extract style modifiers - prefer 'value' over 'text' for style modifiers
                                        style_modifiers = []
                                        for m in modifiers:
                                            if m.get('type') == 'style':
                                                # Try 'value' first, then 'text', filter out empty strings
                                                modifier_text = m.get('value') or m.get('text', '')
                                                if modifier_text and modifier_text.strip():
                                                    style_modifiers.append(modifier_text.strip())
                                        
                                        logger.info(f"         📝 Extracted components:")
                                        logger.info(f"             Style modifiers: {style_modifiers}")
                                        logger.info(f"             Objects: {objects}")
                                        logger.info(f"             Actions: {actions}")
                                        logger.info(f"             Persons: {persons}")
                                        logger.info(f"             Settings: {settings}")
                                        logger.info(f"             Styles: {styles}")
                                        logger.info(f"             Other entities: {other_entities}")
                                        
                                        # Combine entities in a logical order for image generation
                                        prompt_parts = []
                                        
                                        # Add style modifiers first (like "photo", "photorealistic")
                                        if style_modifiers:
                                            prompt_parts.extend(style_modifiers)
                                        
                                        # Add persons (subjects with identity)
                                        if persons:
                                            prompt_parts.extend(persons)
                                        
                                        # Add objects (main subjects without identity)
                                        if objects:
                                            prompt_parts.extend(objects)
                                        
                                        # Add actions immediately after subjects for better context
                                        if actions:
                                            prompt_parts.extend(actions)
                                        
                                        # Add settings/locations with preposition
                                        if settings:
                                            prompt_parts.extend([f"at {setting}" for setting in settings])
                                        
                                        # Add other styles
                                        if styles:
                                            prompt_parts.extend(styles)
                                        
                                        # Add any other entities we might have missed
                                        if other_entities:
                                            prompt_parts.extend(other_entities)
                                        
                                        # Construct the prompt by combining all parts
                                        if prompt_parts:
                                            resolved_value = ' '.join(prompt_parts)
                                            logger.info(f"         🎯 Successfully constructed prompt from entities: '{resolved_value}'")
                                        else:
                                            # Improved fallback - try to extract any meaningful text
                                            logger.info(f"         ⚠️  No prompt parts extracted, attempting fallback...")
                                            
                                            # Try to get all entity text as fallback
                                            all_entity_texts = [e.get('text', '') for e in entities if e.get('text')]
                                            all_modifier_texts = [m.get('text', '') for m in modifiers if m.get('text')]
                                            
                                            # Combine all available text
                                            all_texts = all_entity_texts + all_modifier_texts
                                            
                                            if all_texts:
                                                resolved_value = ' '.join(all_texts)
                                                logger.info(f"         🎯 Fallback - combined all texts: '{resolved_value}'")
                                            else:
                                                # Last resort - check if we can use the original user request
                                                if 'user_request' in initial_inputs:
                                                    resolved_value = initial_inputs['user_request']
                                                    logger.info(f"         🎯 Last resort fallback - using original user request: '{resolved_value}'")
                                                else:
                                                    # Final fallback - use the raw data as string
                                                    resolved_value = str(step_data)
                                                    logger.info(f"         🎯 Final fallback - raw data: '{resolved_value}'")
                                    else:
                                        resolved_value = str(step_data)
                                        logger.info(f"         ⚠️  No entities found, using string: '{resolved_value}'")
                                else:
                                    # Smart field extraction for prompt-like inputs
                                    # Check if this is being resolved for a prompt field
                                    is_prompt_field = key.lower() in ['prompt', 'prompt_text', 'query', 'text']
                                    logger.info(f"         🔍 Checking if '{key}' is prompt field: {is_prompt_field}")
                                    
                                    if is_prompt_field:
                                        logger.info(f"         🎯 This is a prompt field, extracting optimized string...")
                                        # Try to extract the most relevant string field for prompts
                                        prompt_candidates = [
                                            'optimized_prompt',
                                            'prompt',
                                            'prompt_text', 
                                            'text',
                                            'query',
                                            'description'
                                        ]
                                        
                                        extracted_prompt = None
                                        logger.info(f"         🔍 Searching in step data keys: {list(step_data.keys())}")
                                        for candidate in prompt_candidates:
                                            if candidate in step_data and isinstance(step_data[candidate], str):
                                                extracted_prompt = step_data[candidate]
                                                logger.info(f"         🎯 ✅ EXTRACTED prompt field '{candidate}': '{extracted_prompt}'")
                                                break
                                            else:
                                                logger.info(f"         🔍 Candidate '{candidate}' not found or not string")
                                        
                                        if extracted_prompt:
                                            resolved_value = extracted_prompt
                                            logger.info(f"         🎉 SUCCESS: Using extracted prompt: '{resolved_value}'")
                                        else:
                                            # Fallback to dict for non-prompt fields or if no prompt found
                                            resolved_value = step_data
                                            logger.info(f"         ⚠️  No prompt field found, using full dict for prompt input")
                                    else:
                                        # For non-prompt fields, use the whole dict
                                        resolved_value = step_data
                                        logger.info(f"         ✅ Used dict value for non-prompt field: '{resolved_value}'")
                            else:
                                resolved_value = step_data
                                logger.info(f"         ✅ Used direct value: '{resolved_value}'")
                        else:
                            resolved_value = value
                            logger.info(f"         ❌ Reference '{reference}' not found, keeping original: '{resolved_value}'")
                # Handle initial input references like $user_request
                elif value.startswith("$"):
                    reference = value[1:]
                    logger.info(f"         🔧 Initial input reference: '{reference}'")
                    
                    if reference in initial_inputs:
                        resolved_value = initial_inputs[reference]
                        logger.info(f"         ✅ Found initial input: '{resolved_value}'")
                    else:
                        resolved_value = value
                        logger.info(f"         ❌ Initial input '{reference}' not found, keeping original: '{resolved_value}'")
                else:
                    # Regular string value
                    resolved_value = value
                    logger.info(f"         ✅ Regular string: '{resolved_value}'")
            else:
                # Non-string value, use as-is
                resolved_value = value
                logger.info(f"         ✅ Non-string value: '{resolved_value}'")
            
            resolved_inputs[key] = resolved_value
        
        logger.info(f"      🎯 FINAL RESOLVED INPUTS: {resolved_inputs}")
        return resolved_inputs


class AdaptiveWorkflowTool(Tool):
    """
    Tool for creating adaptive workflows that can modify themselves based on results.
    
    Monitors execution and can replan or adjust the workflow based on intermediate results.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="adaptive_workflow",
            description="Create self-modifying workflows that adapt based on intermediate results",
            category=ToolCategory.WORKFLOW_PLANNING,
            input_schema={
                "type": "object",
                "properties": {
                    "user_request": {"type": "string"},
                    "adaptation_rules": {"type": "array"},
                    "quality_thresholds": {"type": "object"}
                },
                "required": ["user_request"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "final_results": {"type": "object"},
                    "adaptations_made": {"type": "array"},
                    "execution_path": {"type": "array"}
                }
            },
            config=config
        )
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        pass
    
    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        """Execute adaptive workflow."""
        # Advanced implementation for later phases
        return ToolResult(
            success=False,
            error="AdaptiveWorkflowTool not yet implemented - Advanced feature"
        ) 