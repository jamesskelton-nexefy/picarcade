"""
Mem0-Enhanced Tool-First Agent for Pic Arcade

Advanced agent with persistent memory using Mem0 for cross-session context.
Solves the conversation context persistence issue by using enterprise-grade memory.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST, before any other imports
# This ensures API keys are available when tools initialize
agentic_dir = Path(__file__).parent.parent.parent.parent
env_file = agentic_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ Loaded environment from: {env_file}")
else:
    print(f"⚠️ No .env file found at: {env_file}")

# Print loaded API keys for debugging (first 8 chars only)
debug_keys = ["OPENAI_API_KEY", "REPLICATE_API_TOKEN", "MEM0_API_KEY", "PERPLEXITY_API_KEY"]
for key in debug_keys:
    value = os.getenv(key)
    if value:
        print(f"   ✅ {key}: {value[:8]}...")
    else:
        print(f"   ❌ {key}: Not found")

import logging
import time
from typing import Dict, Any, List, Optional
import json
from openai import AsyncOpenAI
from ..tools.base import Tool, ToolRegistry, ToolResult, tool_registry
from ..tools.prompt_tools import PromptParsingTool, PromptOptimizationTool
from ..tools.search_tools import PerplexitySearchTool, WebSearchTool
from ..tools.image_tools import (
    FluxImageManipulationTool, FluxImageGenerationTool, StableDiffusionImageTool, DALLEImageGenerationTool,
    StyleTransferTool, ObjectChangeTool, TextEditingTool, BackgroundSwapTool, CharacterConsistencyTool
)
from ..tools.video_tools import RunwayVideoTool, ReplicateVideoTool, VideoEditingTool
from ..tools.workflow_tools import WorkflowPlanningTool, WorkflowExecutorTool
from ..types import OpenAIConfig
from ..utils.decision_logger import decision_logger, DecisionType
from ..utils.mem0_context import get_mem0_context, create_user_session_id

logger = logging.getLogger(__name__)


class Mem0ToolFirstAgent:
    """
    Enhanced tool-first agent with Mem0 persistent memory for multi-session context.
    
    Provides:
    - Persistent conversation context across API requests
    - Intelligent memory management with Mem0
    - Multi-turn image editing with proper context
    - User preference learning and adaptation
    - Enterprise-grade memory performance and reliability
    """
    
    def __init__(self, config: Optional[OpenAIConfig] = None, mem0_api_key: Optional[str] = None):
        """Initialize the Mem0-enhanced tool-first agent."""
        self.config = config or OpenAIConfig(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            temperature=0.1,
            max_tokens=2000
        )
        
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(api_key=self.config.api_key)
        self.tool_registry = tool_registry
        
        # Initialize Mem0 context
        try:
            self.mem0_context = get_mem0_context(api_key=mem0_api_key)
            logger.info("✅ Mem0 context initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Mem0: {e}")
            logger.info("💡 Set MEM0_API_KEY environment variable or install mem0ai")
            raise
        
        # Initialize and register tools
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register default tools in the registry."""
        print("🔧 Registering tools...")
        
        try:
            # API configuration for tools that need it
            config = {
                "api_key": os.getenv("REPLICATE_API_TOKEN"),
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "perplexity_api_key": os.getenv("PERPLEXITY_API_KEY")
            }
            
            print(f"   Config keys available: {list(k for k, v in config.items() if v)}")
            
            # Register tools one by one with individual error handling
            tools_to_register = []
            
            # Prompt processing tools
            try:
                prompt_parser = PromptParsingTool(config)
                tools_to_register.append(("prompt_parser", prompt_parser))
                print("   ✅ PromptParsingTool created")
            except Exception as e:
                print(f"   ❌ PromptParsingTool failed: {e}")
            
            try:
                prompt_optimizer = PromptOptimizationTool(config)
                tools_to_register.append(("prompt_optimizer", prompt_optimizer))
                print("   ✅ PromptOptimizationTool created")
            except Exception as e:
                print(f"   ❌ PromptOptimizationTool failed: {e}")
            
            # Search tools
            try:
                perplexity_search = PerplexitySearchTool(config)
                tools_to_register.append(("perplexity_search", perplexity_search))
                print("   ✅ PerplexitySearchTool created")
            except Exception as e:
                print(f"   ❌ PerplexitySearchTool failed: {e}")
            
            try:
                web_search = WebSearchTool(config)
                tools_to_register.append(("web_search", web_search))
                print("   ✅ WebSearchTool created")
            except Exception as e:
                print(f"   ❌ WebSearchTool failed: {e}")
            
            # Image generation tools
            try:
                flux_image_manipulation = FluxImageManipulationTool(config)
                tools_to_register.append(("flux_image_manipulation", flux_image_manipulation))
                print("   ✅ FluxImageManipulationTool created")
            except Exception as e:
                print(f"   ❌ FluxImageManipulationTool failed: {e}")
            
            try:
                flux_generation = FluxImageGenerationTool(config)
                tools_to_register.append(("flux_generation", flux_generation))
                print("   ✅ FluxImageGenerationTool created")
            except Exception as e:
                print(f"   ❌ FluxImageGenerationTool failed: {e}")
            
            try:
                stable_diffusion = StableDiffusionImageTool(config)
                tools_to_register.append(("stable_diffusion", stable_diffusion))
                print("   ✅ StableDiffusionImageTool created")
            except Exception as e:
                print(f"   ❌ StableDiffusionImageTool failed: {e}")
            
            try:
                dalle_generation = DALLEImageGenerationTool(config)
                tools_to_register.append(("dalle_generation", dalle_generation))
                print("   ✅ DALLEImageGenerationTool created")
            except Exception as e:
                print(f"   ❌ DALLEImageGenerationTool failed: {e}")
            
            # Specialized Flux editing tools
            try:
                style_transfer = StyleTransferTool(config)
                tools_to_register.append(("style_transfer", style_transfer))
                print("   ✅ StyleTransferTool created")
            except Exception as e:
                print(f"   ❌ StyleTransferTool failed: {e}")
            
            try:
                object_change = ObjectChangeTool(config)
                tools_to_register.append(("object_change", object_change))
                print("   ✅ ObjectChangeTool created")
            except Exception as e:
                print(f"   ❌ ObjectChangeTool failed: {e}")
            
            # Workflow tools
            try:
                workflow_planner = WorkflowPlanningTool(config)
                tools_to_register.append(("workflow_planner", workflow_planner))
                print("   ✅ WorkflowPlanningTool created")
            except Exception as e:
                print(f"   ❌ WorkflowPlanningTool failed: {e}")
            
            try:
                workflow_executor = WorkflowExecutorTool(config)
                tools_to_register.append(("workflow_executor", workflow_executor))
                print("   ✅ WorkflowExecutorTool created")
            except Exception as e:
                print(f"   ❌ WorkflowExecutorTool failed: {e}")
            
            # Video generation tools
            try:
                runway_video = RunwayVideoTool()
                tools_to_register.append(("runway_video", runway_video))
                print("   ✅ RunwayVideoTool created")
            except Exception as e:
                print(f"   ❌ RunwayVideoTool failed: {e}")
            
            try:
                replicate_video = ReplicateVideoTool()
                tools_to_register.append(("replicate_video", replicate_video))
                print("   ✅ ReplicateVideoTool created")
            except Exception as e:
                print(f"   ❌ ReplicateVideoTool failed: {e}")
            
            try:
                video_editing = VideoEditingTool()
                tools_to_register.append(("video_editing", video_editing))
                print("   ✅ VideoEditingTool created")
            except Exception as e:
                print(f"   ❌ VideoEditingTool failed: {e}")
            
            # Register all successfully created tools
            for tool_name, tool in tools_to_register:
                try:
                    self.tool_registry.register(tool)
                    print(f"   ✅ Registered: {tool_name}")
                except Exception as e:
                    print(f"   ❌ Failed to register {tool_name}: {e}")
                
            total_registered = len(self.tool_registry.list_all_tools())
            print(f"🎯 Successfully registered {total_registered} tools")
            
            if total_registered == 0:
                print("⚠️ WARNING: No tools registered! This will cause workflow failures.")
                
        except Exception as e:
            logger.error(f"Failed to register tools: {e}")
            import traceback
            traceback.print_exc()
            # Continue with reduced functionality rather than failing completely
            logger.info("Continuing with limited tool set")
    
    async def process_request(self, user_request: str, user_id: str) -> Dict[str, Any]:
        """
        Process a user request with persistent Mem0 memory context.
        
        Args:
            user_request: The user's natural language request
            user_id: User identifier for memory association
            
        Returns:
            Dictionary containing results and execution metadata
        """
        # Generate request ID for decision tracking
        request_id = f"mem0_agent_{int(time.time() * 1000)}"
        
        # Get conversation summary from Mem0
        conversation_summary = self.mem0_context.get_conversation_summary(user_id)
        
        # Start decision tracking
        decision_logger.start_decision(
            request_id=request_id,
            agent_name="Mem0ToolFirstAgent",
            initial_context={
                "user_request": user_request,
                "user_id": user_id,
                "request_length": len(user_request),
                "available_tools": len(self.tool_registry.list_all_tools()),
                "architecture": "mem0_tool_first",
                "has_conversation_history": conversation_summary["has_recent_images"],
                "memory_source": "mem0",
                "total_memories": conversation_summary["total_generations"]
            }
        )
        
        start_time = time.time()
        
        try:
            # Step 1: Analyze conversation context using Mem0
            context_analysis = await self._analyze_mem0_context(user_request, user_id, request_id)
            
            # Step 2: Plan workflow with Mem0 context
            workflow_plan = await self._plan_workflow(user_request, request_id, context_analysis)
            
            if not workflow_plan["success"]:
                error_msg = workflow_plan.get("error", "Unknown workflow planning error")
                
                # Log workflow planning failure
                decision_logger.log_decision_step(
                    request_id=request_id,
                    decision_type=DecisionType.ERROR_HANDLING,
                    input_data={"workflow_planning_failed": True},
                    decision_reasoning="Workflow planning failed, cannot proceed with request execution",
                    output_data={"error": error_msg},
                    confidence_score=0.0,
                    error=error_msg
                )
                
                decision_logger.complete_decision(
                    request_id=request_id,
                    final_result={"error": error_msg},
                    success=False
                )
                
                return {
                    "success": False,
                    "error": "Failed to plan workflow",
                    "details": error_msg
                }
            
            # Log successful workflow planning
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.WORKFLOW_PLANNING,
                input_data={"workflow_plan_received": True},
                decision_reasoning=f"Successfully planned Mem0-aware workflow with {len(workflow_plan['data']['workflow_plan'])} steps",
                output_data={
                    "workflow_steps": len(workflow_plan['data']['workflow_plan']),
                    "estimated_time": workflow_plan['data'].get('estimated_time', 'unknown'),
                    "tools_required": [step.get('tool_name') for step in workflow_plan['data']['workflow_plan']],
                    "context_aware": context_analysis.get("is_edit", False),
                    "memory_source": "mem0"
                },
                confidence_score=0.9,
                metadata={
                    "planning_success": True,
                    "ready_for_execution": True,
                    "uses_mem0_context": True,
                    "edit_context_found": context_analysis.get("is_edit", False)
                }
            )
            
            # Step 3: Execute the planned workflow
            execution_result = await self._execute_workflow(
                workflow_plan["data"]["workflow_plan"],
                {"user_request": user_request, "context": context_analysis, "user_id": user_id},
                request_id
            )
            
            total_execution_time = (time.time() - start_time) * 1000
            
            # Step 4: Store results in Mem0
            if execution_result["success"]:
                await self._store_generation_result_mem0(
                    user_request, execution_result, request_id, context_analysis, user_id
                )
            
            # Step 5: Process and return results
            result = {
                "success": execution_result["success"],
                "user_request": user_request,
                "user_id": user_id,
                "workflow_plan": workflow_plan["data"],
                "execution_results": execution_result["data"],
                "context_analysis": context_analysis,
                "metadata": {
                    "tools_used": self._extract_tools_used(execution_result),
                    "total_time": execution_result["data"].get("total_time", 0),
                    "execution_status": execution_result["data"].get("execution_status"),
                    "total_execution_time_ms": total_execution_time,
                    "was_edit_operation": context_analysis.get("is_edit", False),
                    "memory_source": "mem0",
                    "memory_entries_used": conversation_summary["total_generations"]
                }
            }
            
            # Log final result
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.WORKFLOW_PLANNING,
                input_data={"processing_complete": True},
                decision_reasoning=f"Completed Mem0-enhanced request processing with persistent memory",
                output_data={
                    "final_success": execution_result["success"],
                    "tools_used_count": len(self._extract_tools_used(execution_result)),
                    "workflow_executed": True,
                    "memory_persisted": True,
                    "context_preserved": context_analysis.get("is_edit", False)
                },
                confidence_score=1.0 if execution_result["success"] else 0.3,
                execution_time_ms=total_execution_time,
                metadata={
                    "processing_complete": True,
                    "architecture_success": "mem0_tool_first_completed",
                    "persistent_memory_used": True
                }
            )
            
            # Complete decision tracking
            decision_logger.complete_decision(
                request_id=request_id,
                final_result={
                    "success": execution_result["success"],
                    "tools_used": len(self._extract_tools_used(execution_result)),
                    "execution_time_ms": total_execution_time,
                    "edit_operation": context_analysis.get("is_edit", False),
                    "memory_source": "mem0"
                },
                success=execution_result["success"]
            )
            
            return result
            
        except Exception as e:
            total_execution_time = (time.time() - start_time) * 1000
            error_msg = f"Mem0 request processing failed: {str(e)}"
            
            # Log processing failure
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"processing_error": str(e)},
                decision_reasoning="Unexpected error during Mem0-enhanced request processing",
                output_data={"error_type": type(e).__name__},
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
            return {
                "success": False,
                "error": error_msg
            }
    
    async def _analyze_mem0_context(self, user_request: str, user_id: str, request_id: str) -> Dict[str, Any]:
        """
        Analyze conversation context using Mem0 persistent memory.
        
        Args:
            user_request: Current user request
            user_id: User identifier
            request_id: Request ID for logging
            
        Returns:
            Context analysis results from Mem0
        """
        context_start_time = time.time()
        
        # Get conversation summary from Mem0
        conversation_summary = self.mem0_context.get_conversation_summary(user_id)
        
        # Log context analysis start
        decision_logger.log_decision_step(
            request_id=request_id,
            decision_type=DecisionType.VALIDATION,
            input_data={
                "user_request": user_request,
                "user_id": user_id,
                "mem0_memories": conversation_summary["total_generations"],
                "recent_images_available": conversation_summary["has_recent_images"]
            },
            decision_reasoning="Analyzing Mem0 persistent memory to detect edit intent and retrieve context",
            output_data={"mem0_analysis_initiated": True},
            confidence_score=0.9,
            metadata={
                "context_check": "mem0_memory",
                "multi_turn_detection": True,
                "memory_source": "mem0"
            }
        )
        
        # Simple intent detection for context analysis
        try:
            prompt_lower = user_request.lower()
            detected_intent = "generate_image"  # default
            
            if any(word in prompt_lower for word in ["edit", "modify", "change", "add", "remove", "put"]):
                detected_intent = "edit_image"
            elif any(word in prompt_lower for word in ["style", "like", "as"]):
                detected_intent = "style_transfer"
                
        except Exception:
            detected_intent = "generate_image"
        
        # Detect edit context using Mem0
        edit_context = self.mem0_context.detect_edit_context(user_id, user_request, detected_intent)
        
        context_time = (time.time() - context_start_time) * 1000
        
        # Log context analysis results
        decision_logger.log_decision_step(
            request_id=request_id,
            decision_type=DecisionType.VALIDATION,
            input_data={"mem0_edit_detection_complete": True},
            decision_reasoning=f"Mem0 analysis {'detected edit intent' if edit_context['is_edit'] else 'detected new generation intent'} with {edit_context['confidence']:.2f} confidence",
            output_data={
                "is_edit": edit_context["is_edit"],
                "edit_type": edit_context.get("edit_type"),
                "has_target_image": edit_context.get("target_image") is not None,
                "context_confidence": edit_context["confidence"],
                "memory_source": edit_context.get("memory_source", "mem0"),
                "mem0_memories_found": conversation_summary["total_generations"]
            },
            confidence_score=edit_context["confidence"],
            execution_time_ms=context_time,
            metadata={
                "context_analysis_complete": True,
                "edit_context_detected": edit_context["is_edit"],
                "mem0_powered": True,
                "persistent_memory": True
            }
        )
        
        return edit_context
    
    async def _store_generation_result_mem0(
        self, 
        user_request: str, 
        execution_result: Dict[str, Any], 
        request_id: str,
        context_analysis: Dict[str, Any],
        user_id: str
    ) -> None:
        """
        Store generation result in Mem0 for persistent memory.
        
        Args:
            user_request: Original user request
            execution_result: Execution results from workflow
            request_id: Request ID
            context_analysis: Context analysis results
            user_id: User identifier
        """
        try:
            # Extract result data
            final_outputs = execution_result.get("data", {}).get("final_outputs", {})
            
            # Determine result type and extract data
            result_type = "unknown"
            result_data = {}
            media_url = None
            
            # Detect if this was video generation based on the tools used or intent
            was_video_generation = False
            if execution_result.get("data", {}).get("execution_results"):
                for step in execution_result["data"]["execution_results"]:
                    tool_name = step.get("tool_name", "")
                    if "video" in tool_name.lower():
                        was_video_generation = True
                        break
            
            # Also check context analysis for video intent
            if (context_analysis.get("edit_instructions", "") or "").lower().find("video") != -1 or \
               (context_analysis.get("original_prompt") or "").lower().find("video") != -1:
                was_video_generation = True
            
            media_type = "video" if was_video_generation else "image"
            print(f"🔍 Extracting {media_type} URL from final_outputs: {list(final_outputs.keys())}")
            
            # Check for media outputs in various formats (images or videos)
            # Method 1: Direct media URL fields
            if was_video_generation:
                # Check video-specific fields
                if "video_url" in final_outputs:
                    media_url = final_outputs["video_url"]
                    print(f"   ✅ Found direct video_url: {media_url[:50]}...")
                elif "videos" in final_outputs:
                    videos = final_outputs["videos"]
                    if isinstance(videos, list) and len(videos) > 0:
                        media_url = videos[0]
                        print(f"   ✅ Found video in videos array: {media_url[:50]}...")
                elif "generated_video" in final_outputs:
                    media_url = final_outputs["generated_video"]
                    print(f"   ✅ Found generated_video: {media_url[:50]}...")
            else:
                # Check image-specific fields
                if "image_url" in final_outputs:
                    media_url = final_outputs["image_url"]
                    print(f"   ✅ Found direct image_url: {media_url[:50]}...")
                elif "modified_image" in final_outputs:
                    media_url = final_outputs["modified_image"]
                    print(f"   ✅ Found modified_image: {media_url[:50]}...")
            
            # Method 2: Media arrays (FluxKontext format)
            if not media_url:
                media_key = "videos" if was_video_generation else "images"
                if media_key in final_outputs:
                    media_items = final_outputs[media_key]
                    print(f"   🔍 Found {media_key} array with {len(media_items) if isinstance(media_items, list) else 'unknown'} items")
                    
                    if isinstance(media_items, list) and len(media_items) > 0:
                        # Extract URL from first item in array
                        first_item = media_items[0]
                        if isinstance(first_item, dict) and "url" in first_item:
                            media_url = first_item["url"]
                            print(f"   ✅ Extracted URL from {media_key}[0]: {media_url[:50]}...")
                        elif isinstance(first_item, str):
                            media_url = first_item
                            print(f"   ✅ Found direct URL in {media_key}[0]: {media_url[:50]}...")
            
            # Method 2b: Check for step-based media (workflow executor format)
            if not media_url:
                media_suffix = '.videos' if was_video_generation else '.images'
                media_key = 'videos' if was_video_generation else 'images'
                
                for key, value in final_outputs.items():
                    if key.endswith(media_suffix) or key == media_key:
                        print(f"   🔍 Found {media_type}s in {key}")
                        if isinstance(value, list) and len(value) > 0:
                            first_item = value[0]
                            if isinstance(first_item, dict) and "url" in first_item:
                                media_url = first_item["url"]
                                print(f"   ✅ Extracted URL from {key}[0]: {media_url[:50]}...")
                                break
                            elif isinstance(first_item, str):
                                media_url = first_item
                                print(f"   ✅ Found direct URL in {key}[0]: {media_url[:50]}...")
                                break
            
            # Method 3: Check all final_outputs for any URL-like strings
            if not media_url:
                print("   🔍 Scanning all final_outputs for URLs...")
                for key, value in final_outputs.items():
                    if isinstance(value, str) and ('http' in value or 'https' in value):
                        media_url = value
                        print(f"   ✅ Found URL in {key}: {media_url[:50]}...")
                        break
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and ('http' in item or 'https' in item):
                                media_url = item
                                print(f"   ✅ Found URL in {key} array: {media_url[:50]}...")
                                break
                            elif isinstance(item, dict):
                                # Check for nested URL in dict (FluxKontext format)
                                if "url" in item and isinstance(item["url"], str):
                                    media_url = item["url"]
                                    print(f"   ✅ Found nested URL in {key}: {media_url[:50]}...")
                                    break
                        if media_url:
                            break
            
            if media_url:
                result_type = media_type  # "video" or "image"
                result_data = {
                    f"{media_type}_url": media_url,
                    "generation_params": final_outputs.get("generation_params", {}),
                    "processing_time": final_outputs.get("processing_time", 0),
                    "media_type": media_type
                }
                
                # If this was an edit, include original context
                if context_analysis.get("is_edit"):
                    result_data["edit_type"] = context_analysis.get("edit_type")
                    result_data["original_image"] = context_analysis.get("target_image")
                    result_data["edit_instructions"] = context_analysis.get("edit_instructions")
                
                print(f"   🎯 Storing {media_type} in Mem0: {media_url[:60]}...")
            else:
                print(f"   ❌ No {media_type} URL found in any format")
                result_type = "text"  # Fallback
                result_data = {"content": f"No {media_type} generated"}
            
            # Store in Mem0
            # Determine proper intent based on what was actually generated
            if was_video_generation:
                intent = context_analysis.get("edit_type", "generate_video") if context_analysis.get("is_edit") else "generate_video"
            else:
                intent = context_analysis.get("edit_type", "generate_image") if context_analysis.get("is_edit") else "generate_image"
            
            success = self.mem0_context.store_generation_result(
                user_id=user_id,
                prompt=user_request,
                intent=intent,
                result_type=result_type,
                result_data=result_data,
                agent_name="Mem0ToolFirstAgent",
                request_id=request_id
            )
            
            # Log storage results
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.VALIDATION,
                input_data={"mem0_storage": True},
                decision_reasoning="Storing generation result in Mem0 for persistent cross-session memory",
                output_data={
                    "storage_success": success,
                    "stored_result_type": result_type,
                    "has_media_url": f"{media_type}_url" in result_data,
                    "media_type": media_type,
                    "memory_source": "mem0",
                    "extracted_media_url": bool(media_url)
                },
                confidence_score=1.0 if success else 0.0,
                metadata={
                    "memory_persistence": "mem0_stored",
                    "enables_future_editing": result_type in ["image", "video"] and media_url,
                    "cross_session_continuity": True,
                    "generation_type": media_type
                }
            )
            
        except Exception as e:
            # Log storage failure
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"mem0_storage_error": str(e)},
                decision_reasoning="Failed to store result in Mem0, future edit operations may not work properly",
                output_data={"error_type": type(e).__name__},
                confidence_score=0.0,
                error=str(e),
                metadata={"memory_persistence": "storage_failed"}
            )
            
            logger.warning(f"Failed to store generation result in Mem0: {e}")
            import traceback
            traceback.print_exc()
    
    async def _plan_workflow(self, user_request: str, parent_request_id: str, context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan a workflow using the enhanced workflow planner with Mem0 context.
        
        Args:
            user_request: User's request
            parent_request_id: Parent request ID for decision tracking
            context_analysis: Context analysis from Mem0
            
        Returns:
            Workflow plan result
        """
        planning_start_time = time.time()
        
        try:
            # Log Mem0-aware tool selection
            decision_logger.log_decision_step(
                request_id=parent_request_id,
                decision_type=DecisionType.TOOL_SELECTION,
                input_data={
                    "tool_search": "workflow_planner",
                    "selection_criteria": "mem0_context_aware_planning",
                    "is_edit_request": context_analysis.get("is_edit", False),
                    "memory_source": "mem0"
                },
                decision_reasoning="Selecting workflow planner with Mem0 context to create optimal workflow for persistent memory-aware operations",
                output_data={"selected_tool": "workflow_planner"},
                confidence_score=0.95,
                metadata={
                    "tool_selection_strategy": "mem0_context_aware",
                    "planning_phase": "workflow_design",
                    "considers_persistent_memory": True
                }
            )
            
            # Get workflow planner tool
            planner = self.tool_registry.get_tool("workflow_planner")
            if not planner:
                error_msg = "Workflow planner tool not available"
                
                decision_logger.log_decision_step(
                    request_id=parent_request_id,
                    decision_type=DecisionType.ERROR_HANDLING,
                    input_data={"missing_tool": "workflow_planner"},
                    decision_reasoning="Critical tool missing - workflow planner not available in registry",
                    output_data={"available_tools": list(self.tool_registry._tools.keys())},
                    confidence_score=0.0,
                    error=error_msg
                )
                
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Prepare Mem0-enhanced input for workflow planner
            planner_input = {
                "user_request": user_request,
                "available_tools": [
                    tool.get_metadata() 
                    for tool in self.tool_registry._tools.values()
                ],
                "conversation_context": {
                    "is_edit": context_analysis.get("is_edit", False),
                    "edit_type": context_analysis.get("edit_type"),
                    "has_target_image": context_analysis.get("target_image") is not None,
                    "target_image_url": context_analysis.get("target_image"),
                    "original_prompt": context_analysis.get("original_prompt"),
                    "edit_instructions": context_analysis.get("edit_instructions", user_request),
                    "memory_source": "mem0",
                    "confidence": context_analysis.get("confidence", 0.0)
                }
            }
            
            # Plan the workflow with Mem0 context
            result = await planner.invoke(planner_input)
            
            # --- PATCH: Inject image for image-to-video if editing ---
            if result.success and context_analysis.get("is_edit") and context_analysis.get("target_image"):
                for step in result.data.get("workflow_plan", []):
                    if step.get("tool_name", "").endswith("video_generation"):
                        # Only add if not already present
                        if "image" not in step["inputs"]:
                            step["inputs"]["image"] = context_analysis["target_image"]
            
            planning_time = (time.time() - planning_start_time) * 1000
            
            if result.success:
                # Log successful planning
                decision_logger.log_decision_step(
                    request_id=parent_request_id,
                    decision_type=DecisionType.WORKFLOW_PLANNING,
                    input_data={"mem0_workflow_planning_complete": True},
                    decision_reasoning=f"Successfully created Mem0-aware workflow plan with {len(result.data.get('workflow_plan', []))} steps",
                    output_data={
                        "planning_success": True,
                        "workflow_steps": len(result.data.get('workflow_plan', [])),
                        "estimated_time": result.data.get('estimated_time'),
                        "complexity": result.data.get('complexity', 'unknown'),
                        "uses_persistent_memory": True,
                        "memory_source": "mem0"
                    },
                    confidence_score=0.9,
                    execution_time_ms=planning_time,
                    metadata={
                        "planning_successful": True,
                        "workflow_complexity": result.data.get('complexity', 'standard'),
                        "mem0_context_aware": True
                    }
                )
            else:
                # Log planning failure
                decision_logger.log_decision_step(
                    request_id=parent_request_id,
                    decision_type=DecisionType.ERROR_HANDLING,
                    input_data={"mem0_workflow_planning_failed": True},
                    decision_reasoning="Mem0-aware workflow planning tool failed to create execution plan",
                    output_data={"planning_error": result.error},
                    confidence_score=0.0,
                    execution_time_ms=planning_time,
                    error=result.error
                )
            
            return {
                "success": result.success,
                "data": result.data,
                "error": result.error
            }
            
        except Exception as e:
            planning_time = (time.time() - planning_start_time) * 1000
            error_msg = f"Mem0 workflow planning failed: {str(e)}"
            
            decision_logger.log_decision_step(
                request_id=parent_request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"mem0_planning_exception": str(e)},
                decision_reasoning="Unexpected error during Mem0-aware workflow planning",
                output_data={"error_type": type(e).__name__},
                confidence_score=0.0,
                execution_time_ms=planning_time,
                error=error_msg
            )
            
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    async def _execute_workflow(
        self, 
        workflow_plan: List[Dict[str, Any]], 
        initial_inputs: Dict[str, Any],
        parent_request_id: str
    ) -> Dict[str, Any]:
        """
        Execute a workflow plan using the workflow executor tool.
        
        Args:
            workflow_plan: List of workflow steps
            initial_inputs: Initial inputs for the workflow (includes user_id)
            parent_request_id: Parent request ID for decision tracking
            
        Returns:
            Execution results
        """
        execution_start_time = time.time()
        
        try:
            # Log execution initiation
            decision_logger.log_decision_step(
                request_id=parent_request_id,
                decision_type=DecisionType.TOOL_SELECTION,
                input_data={
                    "workflow_steps": len(workflow_plan),
                    "execution_tool_search": "workflow_executor",
                    "memory_source": "mem0"
                },
                decision_reasoning="Selecting workflow executor for Mem0-aware workflow execution",
                output_data={
                    "selected_tool": "workflow_executor",
                    "execution_strategy": "sequential_with_mem0_context"
                },
                confidence_score=0.95,
                metadata={
                    "execution_phase": "tool_selection",
                    "workflow_size": len(workflow_plan),
                    "mem0_context_available": True
                }
            )
            
            # Get workflow executor tool
            executor = self.tool_registry.get_tool("workflow_executor")
            if not executor:
                error_msg = "Workflow executor tool not available"
                
                decision_logger.log_decision_step(
                    request_id=parent_request_id,
                    decision_type=DecisionType.ERROR_HANDLING,
                    input_data={"missing_tool": "workflow_executor"},
                    decision_reasoning="Critical tool missing - workflow executor not available",
                    output_data={"execution_blocked": True},
                    confidence_score=0.0,
                    error=error_msg
                )
                
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Execute the workflow with Mem0 context
            result = await executor.invoke({
                "workflow_plan": workflow_plan,
                "initial_inputs": initial_inputs,
                "execution_options": {
                    "stop_on_error": False,  # Continue on errors for better debugging
                    "max_retries": 2,
                    "timeout_per_step": 60
                }
            })
            
            execution_time = (time.time() - execution_start_time) * 1000
            
            if result.success:
                # Log successful execution
                decision_logger.log_decision_step(
                    request_id=parent_request_id,
                    decision_type=DecisionType.WORKFLOW_PLANNING,
                    input_data={"mem0_workflow_execution_complete": True},
                    decision_reasoning=f"Successfully executed Mem0-aware workflow with persistent context",
                    output_data={
                        "execution_success": True,
                        "steps_executed": len(result.data.get("execution_results", [])),
                        "execution_status": result.data.get("execution_status"),
                        "final_outputs": bool(result.data.get("final_outputs")),
                        "memory_source": "mem0"
                    },
                    confidence_score=0.9,
                    execution_time_ms=execution_time,
                    metadata={
                        "execution_successful": True,
                        "workflow_completed": True,
                        "mem0_context_preserved": True
                    }
                )
            else:
                # Log execution failure
                decision_logger.log_decision_step(
                    request_id=parent_request_id,
                    decision_type=DecisionType.ERROR_HANDLING,
                    input_data={"mem0_workflow_execution_failed": True},
                    decision_reasoning="Mem0-aware workflow execution failed",
                    output_data={"execution_error": result.error},
                    confidence_score=0.0,
                    execution_time_ms=execution_time,
                    error=result.error
                )
            
            return {
                "success": result.success,
                "data": result.data,
                "error": result.error
            }
            
        except Exception as e:
            execution_time = (time.time() - execution_start_time) * 1000
            error_msg = f"Mem0 workflow execution failed: {str(e)}"
            
            decision_logger.log_decision_step(
                request_id=parent_request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"mem0_execution_exception": str(e)},
                decision_reasoning="Unexpected error during Mem0-aware workflow execution",
                output_data={"error_type": type(e).__name__},
                confidence_score=0.0,
                execution_time_ms=execution_time,
                error=error_msg
            )
            
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    def _extract_tools_used(self, execution_result: Dict[str, Any]) -> List[str]:
        """Extract list of tools used during execution."""
        tools_used = []
        
        if execution_result.get("success") and execution_result.get("data"):
            execution_results = execution_result["data"].get("execution_results", [])
            tools_used = [
                step["tool_name"] for step in execution_results 
                if step.get("tool_name")
            ]
        
        return tools_used
    
    def get_user_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get memory statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Memory statistics from Mem0
        """
        return self.mem0_context.get_memory_stats(user_id)
    
    async def explain_capabilities(self) -> Dict[str, Any]:
        """
        Explain the agent's capabilities including Mem0 memory features.
        
        Returns:
            Explanation of what the agent can do
        """
        try:
            tools_list = "\n".join([
                f"- {tool.name}: {tool.description}"
                for tool in self.tool_registry._tools.values()
            ])
            
            system_prompt = f"""You are explaining the capabilities of an AI agent that uses Mem0 for persistent memory.

The agent has access to these tools:
{tools_list}

Key features:
- Persistent memory across sessions using Mem0
- Intelligent context detection for multi-turn image editing
- User preference learning and adaptation
- Cross-session conversation continuity

Explain what this agent can do in terms of:
1. Types of requests it can handle
2. How it maintains context between sessions
3. Multi-turn image editing capabilities
4. Memory-powered personalization
5. Benefits of persistent memory architecture

Be concise but comprehensive."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Explain what this Mem0-enhanced agent can do"}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            return {
                "success": True,
                "explanation": response.choices[0].message.content,
                "tools_available": len(self.tool_registry.list_all_tools()),
                "memory_system": "Mem0",
                "persistent_memory": True,
                "cross_session_continuity": True,
                "tool_categories": list(set(
                    tool.category for tool in self.tool_registry._tools.values()
                ))
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to explain capabilities: {str(e)}"
            } 