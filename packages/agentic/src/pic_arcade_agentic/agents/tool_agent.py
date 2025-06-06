"""
Tool-First Agent for Pic Arcade

Demonstrates the tool-first architecture where agents dynamically select
and chain tools based on user requests, following the pattern described
in the tooluse guide.
"""

import logging
import time
from typing import Dict, Any, List, Optional
import json
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file in the agentic package
agentic_dir = Path(__file__).parent.parent.parent.parent
load_dotenv(agentic_dir / ".env")

from ..tools.base import Tool, ToolRegistry, ToolResult, tool_registry
from ..tools.prompt_tools import PromptParsingTool, PromptOptimizationTool
from ..tools.search_tools import PerplexitySearchTool, WebSearchTool
from ..tools.image_tools import (
    FluxImageManipulationTool, FluxImageGenerationTool, StableDiffusionImageTool, DALLEImageGenerationTool,
    StyleTransferTool, ObjectChangeTool, TextEditingTool, BackgroundSwapTool, CharacterConsistencyTool
)
from ..tools.workflow_tools import WorkflowPlanningTool, WorkflowExecutorTool
from ..types import OpenAIConfig
from ..utils.decision_logger import decision_logger, DecisionType
from ..utils.conversation_context import conversation_context

logger = logging.getLogger(__name__)


class ToolFirstAgent:
    """
    Agent that uses the tool-first architecture for dynamic capability selection.
    
    Instead of hardcoded workflows, this agent:
    1. Analyzes user requests
    2. Discovers relevant tools
    3. Plans multi-step workflows
    4. Executes tool chains dynamically
    5. Adapts based on intermediate results
    6. Maintains conversation context for multi-turn interactions
    """
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize the tool-first agent."""
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
        
        # Initialize and register tools
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register default tools in the registry."""
        try:
            # API configuration for tools that need it
            config = {
                "api_key": os.getenv("REPLICATE_API_TOKEN"),
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "perplexity_api_key": os.getenv("PERPLEXITY_API_KEY")
            }
            
            # Prompt processing tools
            prompt_parser = PromptParsingTool(config)
            prompt_optimizer = PromptOptimizationTool(config)
            
            # Search tools (all using Perplexity now)
            perplexity_search = PerplexitySearchTool(config)
            web_search = WebSearchTool(config)  # This is now an alias for Perplexity
            
            # Image generation and editing tools (using Replicate and OpenAI)
            flux_image_manipulation = FluxImageManipulationTool(config)
            flux_generation = FluxImageGenerationTool(config)  # Legacy compatibility
            stable_diffusion = StableDiffusionImageTool(config)
            dalle_generation = DALLEImageGenerationTool(config)
            
            # Specialized Flux editing tools
            style_transfer = StyleTransferTool(config)
            object_change = ObjectChangeTool(config)
            text_editing = TextEditingTool(config)
            background_swap = BackgroundSwapTool(config)
            character_consistency = CharacterConsistencyTool(config)
            
            # Video generation tools
            from ..tools.video_tools import RunwayVideoTool, ReplicateVideoTool, VideoEditingTool
            runway_video = RunwayVideoTool()
            replicate_video = ReplicateVideoTool()
            video_editing = VideoEditingTool()
            
            # Workflow tools
            workflow_planner = WorkflowPlanningTool(config)
            workflow_executor = WorkflowExecutorTool(config)
            
            # Register all tools
            for tool in [
                prompt_parser, prompt_optimizer,
                perplexity_search, web_search,
                flux_image_manipulation, flux_generation, stable_diffusion, dalle_generation,
                style_transfer, object_change, text_editing, background_swap, character_consistency,
                runway_video, replicate_video, video_editing,
                workflow_planner, workflow_executor
            ]:
                self.tool_registry.register(tool)
                
            logger.info(f"Registered {len(self.tool_registry.list_all_tools())} tools")
            
        except Exception as e:
            logger.error(f"Failed to register tools: {e}")
            # Continue with reduced functionality rather than failing completely
            logger.info("Continuing with limited tool set")
    
    async def process_request(self, user_request: str) -> Dict[str, Any]:
        """
        Process a user request using dynamic tool selection with conversation context.
        
        Args:
            user_request: The user's natural language request
            
        Returns:
            Dictionary containing results and execution metadata
        """
        # Generate request ID for decision tracking
        request_id = f"tool_agent_{int(time.time() * 1000)}"
        
        # Start decision tracking
        decision_logger.start_decision(
            request_id=request_id,
            agent_name="ToolFirstAgent",
            initial_context={
                "user_request": user_request,
                "request_length": len(user_request),
                "available_tools": len(self.tool_registry.list_all_tools()),
                "architecture": "tool_first",
                "has_conversation_history": len(conversation_context.generation_results) > 0
            }
        )
        
        start_time = time.time()
        
        try:
            # Step 1: Check conversation context for multi-turn interactions
            context_analysis = await self._analyze_conversation_context(user_request, request_id)
            
            # Step 2: Analyze request and plan workflow (with context)
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
                decision_reasoning=f"Successfully planned workflow with {len(workflow_plan['data']['workflow_plan'])} steps, proceeding to execution",
                output_data={
                    "workflow_steps": len(workflow_plan['data']['workflow_plan']),
                    "estimated_time": workflow_plan['data'].get('estimated_time', 'unknown'),
                    "tools_required": [step.get('tool_name') for step in workflow_plan['data']['workflow_plan']],
                    "context_aware": context_analysis.get("is_edit", False)
                },
                confidence_score=0.9,
                metadata={
                    "planning_success": True,
                    "ready_for_execution": True,
                    "uses_conversation_context": context_analysis.get("is_edit", False)
                }
            )
            
            # Step 3: Execute the planned workflow
            execution_result = await self._execute_workflow(
                workflow_plan["data"]["workflow_plan"],
                {"user_request": user_request, "context": context_analysis},
                request_id
            )
            
            total_execution_time = (time.time() - start_time) * 1000
            
            # Step 4: Store results in conversation context
            if execution_result["success"]:
                await self._store_generation_result(
                    user_request, execution_result, request_id, context_analysis
                )
            
            # Step 5: Process and return results
            result = {
                "success": execution_result["success"],
                "user_request": user_request,
                "workflow_plan": workflow_plan["data"],
                "execution_results": execution_result["data"],
                "context_analysis": context_analysis,
                "metadata": {
                    "tools_used": self._extract_tools_used(execution_result),
                    "total_time": execution_result["data"].get("total_time", 0),
                    "execution_status": execution_result["data"].get("execution_status"),
                    "total_execution_time_ms": total_execution_time,
                    "was_edit_operation": context_analysis.get("is_edit", False)
                }
            }
            
            # Log final result
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.WORKFLOW_PLANNING,
                input_data={"processing_complete": True},
                decision_reasoning=f"Completed tool-first request processing with {execution_result['success']} execution status",
                output_data={
                    "final_success": execution_result["success"],
                    "tools_used_count": len(self._extract_tools_used(execution_result)),
                    "workflow_executed": True,
                    "context_preserved": context_analysis.get("is_edit", False)
                },
                confidence_score=1.0 if execution_result["success"] else 0.3,
                execution_time_ms=total_execution_time,
                metadata={
                    "processing_complete": True,
                    "architecture_success": "tool_first_completed",
                    "conversation_context_used": context_analysis.get("is_edit", False)
                }
            )
            
            # Complete decision tracking
            decision_logger.complete_decision(
                request_id=request_id,
                final_result={
                    "success": execution_result["success"],
                    "tools_used": len(self._extract_tools_used(execution_result)),
                    "execution_time_ms": total_execution_time,
                    "edit_operation": context_analysis.get("is_edit", False)
                },
                success=execution_result["success"]
            )
            
            return result
            
        except Exception as e:
            total_execution_time = (time.time() - start_time) * 1000
            error_msg = f"Request processing failed: {str(e)}"
            
            # Log processing failure
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"processing_error": str(e)},
                decision_reasoning="Unexpected error during request processing, terminating with error response",
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
    
    async def _analyze_conversation_context(self, user_request: str, request_id: str) -> Dict[str, Any]:
        """
        Analyze conversation context to detect multi-turn interactions and editing intents.
        
        Args:
            user_request: Current user request
            request_id: Request ID for logging
            
        Returns:
            Context analysis results
        """
        context_start_time = time.time()
        
        # Get conversation summary
        conversation_summary = conversation_context.get_conversation_summary()
        
        # Log context analysis start
        decision_logger.log_decision_step(
            request_id=request_id,
            decision_type=DecisionType.VALIDATION,
            input_data={
                "user_request": user_request,
                "conversation_history_size": conversation_summary["total_generations"],
                "recent_images_available": conversation_summary["has_recent_images"]
            },
            decision_reasoning="Analyzing conversation context to detect if this is an edit of previous generation or new creation",
            output_data={"context_analysis_initiated": True},
            confidence_score=0.9,
            metadata={
                "context_check": "conversation_history",
                "multi_turn_detection": True
            }
        )
        
        # Quick parsing to get initial intent (lightweight)
        try:
            # Simple intent detection for context analysis
            prompt_lower = user_request.lower()
            detected_intent = "generate_image"  # default
            
            if any(word in prompt_lower for word in ["edit", "modify", "change", "add", "remove", "put"]):
                detected_intent = "edit_image"
            elif any(word in prompt_lower for word in ["style", "like", "as"]):
                detected_intent = "style_transfer"
                
        except Exception:
            detected_intent = "generate_image"
        
        # Detect edit context using conversation context manager
        edit_context = conversation_context.detect_edit_context(user_request, detected_intent)
        
        context_time = (time.time() - context_start_time) * 1000
        
        # Log context analysis results
        decision_logger.log_decision_step(
            request_id=request_id,
            decision_type=DecisionType.VALIDATION,
            input_data={"edit_detection_complete": True},
            decision_reasoning=f"Context analysis {'detected edit intent' if edit_context['is_edit'] else 'detected new generation intent'} with {edit_context['confidence']:.2f} confidence",
            output_data={
                "is_edit": edit_context["is_edit"],
                "edit_type": edit_context.get("edit_type"),
                "has_target_image": edit_context.get("target_image") is not None,
                "context_confidence": edit_context["confidence"],
                "recent_images_count": len(conversation_context.get_recent_images())
            },
            confidence_score=edit_context["confidence"],
            execution_time_ms=context_time,
            metadata={
                "context_analysis_complete": True,
                "edit_context_detected": edit_context["is_edit"],
                "conversation_aware": True
            }
        )
        
        return edit_context
    
    async def _plan_workflow(self, user_request: str, parent_request_id: str, context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan a workflow for the user request using the workflow planning tool with context.
        
        Args:
            user_request: User's request
            parent_request_id: Parent request ID for decision tracking
            context_analysis: Context analysis from conversation history
            
        Returns:
            Workflow plan result
        """
        planning_start_time = time.time()
        
        try:
            # Log context-aware tool selection
            decision_logger.log_decision_step(
                request_id=parent_request_id,
                decision_type=DecisionType.TOOL_SELECTION,
                input_data={
                    "tool_search": "workflow_planner",
                    "selection_criteria": "workflow_planning_capability",
                    "context_aware": True,
                    "is_edit_request": context_analysis.get("is_edit", False)
                },
                decision_reasoning="Selecting workflow planner tool with conversation context to create appropriate workflow for edit vs generation",
                output_data={"selected_tool": "workflow_planner"},
                confidence_score=0.95,
                metadata={
                    "tool_selection_strategy": "context_aware",
                    "planning_phase": "workflow_design",
                    "considers_edit_context": True
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
            
            # Prepare context-enhanced input for workflow planner
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
                    "edit_instructions": context_analysis.get("edit_instructions", user_request)
                }
            }
            
            # Log workflow planning invocation
            decision_logger.log_decision_step(
                request_id=parent_request_id,
                decision_type=DecisionType.WORKFLOW_PLANNING,
                input_data={
                    "planning_tool": "workflow_planner",
                    "available_tools_for_planning": len(self.tool_registry._tools),
                    "context_provided": True,
                    "edit_context": context_analysis.get("is_edit", False)
                },
                decision_reasoning="Invoking context-aware workflow planner to create optimal execution plan considering conversation history and edit requirements",
                output_data={
                    "planning_initiated": True,
                    "context_enhanced": True
                },
                confidence_score=0.9,
                metadata={
                    "planning_tool_selected": True,
                    "tool_registry_size": len(self.tool_registry._tools),
                    "conversation_context_included": True
                }
            )
            
            # Plan the workflow with context
            result = await planner.invoke(planner_input)
            
            planning_time = (time.time() - planning_start_time) * 1000
            
            if result.success:
                # Log successful planning
                decision_logger.log_decision_step(
                    request_id=parent_request_id,
                    decision_type=DecisionType.WORKFLOW_PLANNING,
                    input_data={"workflow_planning_complete": True},
                    decision_reasoning=f"Successfully created context-aware workflow plan with {len(result.data.get('workflow_plan', []))} steps",
                    output_data={
                        "planning_success": True,
                        "workflow_steps": len(result.data.get('workflow_plan', [])),
                        "estimated_time": result.data.get('estimated_time'),
                        "complexity": result.data.get('complexity', 'unknown'),
                        "uses_edit_context": context_analysis.get("is_edit", False)
                    },
                    confidence_score=0.9,
                    execution_time_ms=planning_time,
                    metadata={
                        "planning_successful": True,
                        "workflow_complexity": result.data.get('complexity', 'standard'),
                        "context_aware_planning": True
                    }
                )
            else:
                # Log planning failure
                decision_logger.log_decision_step(
                    request_id=parent_request_id,
                    decision_type=DecisionType.ERROR_HANDLING,
                    input_data={"workflow_planning_failed": True},
                    decision_reasoning="Context-aware workflow planning tool failed to create execution plan",
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
            error_msg = f"Workflow planning failed: {str(e)}"
            
            decision_logger.log_decision_step(
                request_id=parent_request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"planning_exception": str(e)},
                decision_reasoning="Unexpected error during context-aware workflow planning, cannot proceed with execution",
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
    
    async def _store_generation_result(
        self, 
        user_request: str, 
        execution_result: Dict[str, Any], 
        request_id: str,
        context_analysis: Dict[str, Any]
    ) -> None:
        """
        Store generation result in conversation context for future reference.
        
        Args:
            user_request: Original user request
            execution_result: Execution results from workflow
            request_id: Request ID
            context_analysis: Context analysis results
        """
        try:
            # Extract result data
            final_outputs = execution_result.get("data", {}).get("final_outputs", {})
            
            # Determine result type and extract data
            result_type = "unknown"
            result_data = {}
            
            # Check for image outputs
            if "image_url" in final_outputs or "modified_image" in final_outputs:
                result_type = "image"
                result_data = {
                    "image_url": final_outputs.get("image_url") or final_outputs.get("modified_image"),
                    "generation_params": final_outputs.get("generation_params", {}),
                    "processing_time": final_outputs.get("processing_time", 0)
                }
                
                # If this was an edit, include original context
                if context_analysis.get("is_edit"):
                    result_data["edit_type"] = context_analysis.get("edit_type")
                    result_data["original_image"] = context_analysis.get("target_image")
                    result_data["edit_instructions"] = context_analysis.get("edit_instructions")
            
            # Add to conversation context
            generation_result = conversation_context.add_generation_result(
                prompt=user_request,
                intent=context_analysis.get("edit_type", "generate_image") if context_analysis.get("is_edit") else "generate_image",
                result_type=result_type,
                result_data=result_data,
                agent_name="ToolFirstAgent",
                request_id=request_id
            )
            
            # Log context storage
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.VALIDATION,
                input_data={"result_storage": True},
                decision_reasoning="Storing generation result in conversation context for future multi-turn interactions",
                output_data={
                    "stored_result_type": result_type,
                    "has_image_url": "image_url" in result_data,
                    "context_size": len(conversation_context.generation_results)
                },
                confidence_score=1.0,
                metadata={
                    "context_management": "result_stored",
                    "enables_future_editing": result_type == "image"
                }
            )
            
        except Exception as e:
            # Log storage failure
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"storage_error": str(e)},
                decision_reasoning="Failed to store result in conversation context, future edit operations may not work properly",
                output_data={"error_type": type(e).__name__},
                confidence_score=0.0,
                error=str(e),
                metadata={"context_management": "storage_failed"}
            )
            
            logger.warning(f"Failed to store generation result: {e}")
    
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
            initial_inputs: Initial inputs for the workflow
            parent_request_id: Parent request ID for decision tracking
            
        Returns:
            Execution results
        """
        execution_start_time = time.time()
        
        try:
            # Log execution initiation decision
            decision_logger.log_decision_step(
                request_id=parent_request_id,
                decision_type=DecisionType.TOOL_SELECTION,
                input_data={
                    "workflow_steps": len(workflow_plan),
                    "execution_tool_search": "workflow_executor"
                },
                decision_reasoning="Selecting workflow executor tool to execute the planned workflow steps sequentially",
                output_data={
                    "selected_tool": "workflow_executor",
                    "execution_strategy": "sequential_with_error_handling"
                },
                confidence_score=0.95,
                metadata={
                    "execution_phase": "tool_selection",
                    "workflow_size": len(workflow_plan)
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
                    decision_reasoning="Critical tool missing - workflow executor not available, cannot execute planned workflow",
                    output_data={"execution_blocked": True},
                    confidence_score=0.0,
                    error=error_msg
                )
                
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Log workflow execution invocation
            decision_logger.log_decision_step(
                request_id=parent_request_id,
                decision_type=DecisionType.WORKFLOW_PLANNING,
                input_data={
                    "execution_tool": "workflow_executor",
                    "workflow_plan_steps": len(workflow_plan)
                },
                decision_reasoning="Invoking workflow executor with planned steps and configured execution options for robust execution",
                output_data={
                    "execution_initiated": True,
                    "execution_options": {
                        "stop_on_error": False,
                        "max_retries": 2,
                        "timeout_per_step": 60
                    }
                },
                confidence_score=0.9,
                metadata={
                    "execution_configuration": "fault_tolerant",
                    "step_count": len(workflow_plan)
                }
            )
            
            # Execute the workflow
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
                    input_data={"workflow_execution_complete": True},
                    decision_reasoning=f"Successfully executed workflow with {len(workflow_plan)} steps and obtained results",
                    output_data={
                        "execution_success": True,
                        "steps_executed": len(result.data.get("execution_results", [])),
                        "execution_status": result.data.get("execution_status"),
                        "final_outputs": bool(result.data.get("final_outputs"))
                    },
                    confidence_score=0.9,
                    execution_time_ms=execution_time,
                    metadata={
                        "execution_successful": True,
                        "workflow_completed": True
                    }
                )
            else:
                # Log execution failure
                decision_logger.log_decision_step(
                    request_id=parent_request_id,
                    decision_type=DecisionType.ERROR_HANDLING,
                    input_data={"workflow_execution_failed": True},
                    decision_reasoning="Workflow execution failed - tool chain execution encountered errors",
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
            error_msg = f"Workflow execution failed: {str(e)}"
            
            decision_logger.log_decision_step(
                request_id=parent_request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"execution_exception": str(e)},
                decision_reasoning="Unexpected error during workflow execution, unable to complete tool chain",
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
    
    async def discover_tools(self, query: str) -> List[Dict[str, Any]]:
        """
        Discover relevant tools for a query.
        
        Args:
            query: Search query for tools
            
        Returns:
            List of relevant tool metadata
        """
        matching_tools = self.tool_registry.search_tools(query)
        return [tool.get_metadata() for tool in matching_tools]
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get metadata for all available tools."""
        return [
            tool.get_metadata() 
            for tool in self.tool_registry._tools.values()
        ]
    
    async def explain_capabilities(self) -> Dict[str, Any]:
        """
        Explain the agent's capabilities using GPT-4o.
        
        Returns:
            Explanation of what the agent can do
        """
        try:
            tools_list = "\n".join([
                f"- {tool.name}: {tool.description}"
                for tool in self.tool_registry._tools.values()
            ])
            
            system_prompt = f"""You are explaining the capabilities of an AI agent that uses a tool-first architecture.

The agent has access to these tools:
{tools_list}

Explain what this agent can do in terms of:
1. Types of requests it can handle
2. How it chains tools together
3. Examples of complex workflows it can execute
4. Benefits of the tool-first approach

Be concise but comprehensive."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Explain what this agent can do"}
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
                "tool_categories": list(set(
                    tool.category for tool in self.tool_registry._tools.values()
                ))
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to explain capabilities: {str(e)}"
            }


# Example usage and demonstration functions
async def demonstrate_tool_first_approach():
    """
    Demonstrate how the tool-first approach works with example requests.
    """
    agent = ToolFirstAgent()
    
    example_requests = [
        "Parse this prompt: 'Portrait of Emma Stone in Renaissance style'",
        "Find reference images for Leonardo DiCaprio",
        "Create a workflow to generate a portrait like Van Gogh",
        "Put me in Taylor Swift's dress from the awards show"
    ]
    
    print("🔧 TOOL-FIRST AGENT DEMONSTRATION")
    print("=" * 50)
    
    # Show capabilities
    capabilities = await agent.explain_capabilities()
    if capabilities["success"]:
        print("\n📋 Agent Capabilities:")
        print(capabilities["explanation"])
    
    print(f"\n🛠️  Available Tools: {len(agent.get_available_tools())}")
    for tool_meta in agent.get_available_tools():
        print(f"  - {tool_meta['name']}: {tool_meta['description']}")
    
    # Process example requests
    print("\n🚀 Processing Example Requests:")
    
    for i, request in enumerate(example_requests, 1):
        print(f"\n{i}. Request: {request}")
        print("-" * 40)
        
        result = await agent.process_request(request)
        
        if result["success"]:
            workflow = result["workflow_plan"]
            print(f"✅ Planned {len(workflow['workflow_plan'])} steps")
            print(f"⏱️  Estimated time: {workflow.get('estimated_time', 'N/A')}s")
            print(f"🔗 Tools used: {result['metadata']['tools_used']}")
            print(f"📊 Status: {result['metadata']['execution_status']}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_tool_first_approach()) 