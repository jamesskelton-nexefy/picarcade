"""
Workflow Orchestrator for Phase 2

LangGraph-based orchestration for the agentic pipeline:
1. Parse prompts with GPT-4o
2. Retrieve reference images with Bing Search
3. Coordinate workflow state management
"""

import logging
import time
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from datetime import datetime
import uuid

from ..types import (
    WorkflowState,
    WorkflowStep,
    WorkflowContext,
    GenerationStatus,
    ParsedPrompt,
    PromptReference,
    OpenAIConfig,
    SearchConfig
)
from ..agents.prompt_parser import PromptParsingAgent
from ..agents.reference_retriever import ReferenceRetrievalAgent
from ..utils.decision_logger import decision_logger, DecisionType

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    LangGraph-based orchestrator for the agentic pipeline.
    
    Manages the workflow state and coordinates between different agents
    for prompt parsing and reference retrieval.
    """
    
    def __init__(
        self,
        openai_config: Optional[OpenAIConfig] = None,
        search_config: Optional[SearchConfig] = None
    ):
        """Initialize the workflow orchestrator with agent configurations."""
        self.prompt_parser = PromptParsingAgent(openai_config)
        self.reference_retriever = ReferenceRetrievalAgent(search_config)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow for Phase 2.
        
        Returns:
            StateGraph: Configured workflow graph
        """
        # Define the workflow steps
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each step
        workflow.add_node("parse_prompt", self._parse_prompt_node)
        workflow.add_node("retrieve_references", self._retrieve_references_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define the workflow edges
        workflow.set_entry_point("parse_prompt")
        workflow.add_edge("parse_prompt", "retrieve_references")
        workflow.add_edge("retrieve_references", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def process_prompt(
        self, 
        prompt: str, 
        request_id: Optional[str] = None
    ) -> WorkflowState:
        """
        Process a user prompt through the Phase 2 workflow.
        
        Args:
            prompt: User's input prompt
            request_id: Optional request ID (generated if not provided)
            
        Returns:
            WorkflowState: Final state after processing
        """
        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Start decision tracking
        decision_logger.start_decision(
            request_id=request_id,
            agent_name="WorkflowOrchestrator",
            initial_context={
                "prompt": prompt,
                "prompt_length": len(prompt),
                "workflow_type": "phase2_prompt_processing"
            }
        )
        
        start_time = time.time()
        
        try:
            # Log workflow planning decision
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.WORKFLOW_PLANNING,
                input_data={"original_prompt": prompt},
                decision_reasoning="Initiating Phase 2 workflow with 3 steps: parse_prompt -> retrieve_references -> finalize",
                output_data={
                    "planned_steps": ["parse_prompt", "retrieve_references", "finalize"],
                    "workflow_type": "phase2_pipeline"
                },
                confidence_score=1.0,
                metadata={"phase": "2", "pipeline_version": "langgraph"}
            )
            
            # Initialize workflow state
            initial_state = WorkflowState(
                request_id=request_id,
                current_step=WorkflowStep.PARSE_PROMPT,
                steps=[
                    WorkflowStep.PARSE_PROMPT,
                    WorkflowStep.RETRIEVE_REFERENCES,
                    WorkflowStep.FINALIZE
                ],
                context=WorkflowContext(),
                status=GenerationStatus.PROCESSING
            )
            
            # Add the original prompt to context
            initial_state.context.prompt = ParsedPrompt(
                intent="",
                entities=[],
                modifiers=[],
                references=[],
                confidence=0.0
            )
            
            logger.info(f"Starting workflow for request {request_id}")
            
            # Run the workflow
            result = await self.workflow.ainvoke({
                "state": initial_state,
                "original_prompt": prompt
            })
            
            final_state = result["state"]
            final_state.status = GenerationStatus.COMPLETED
            
            execution_time = (time.time() - start_time) * 1000
            
            # Log workflow completion
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.WORKFLOW_PLANNING,
                input_data={"workflow_state": "completed"},
                decision_reasoning="Workflow completed successfully through all steps",
                output_data={
                    "final_status": final_state.status.value,
                    "total_steps_completed": len(final_state.steps),
                    "has_parsed_prompt": final_state.context.prompt is not None,
                    "references_found": len(final_state.context.references) if final_state.context.references else 0
                },
                confidence_score=1.0,
                execution_time_ms=execution_time,
                metadata={"workflow_success": True}
            )
            
            # Complete decision tracking
            decision_logger.complete_decision(
                request_id=request_id,
                final_result={
                    "status": final_state.status.value,
                    "parsed_prompt_intent": final_state.context.prompt.intent if final_state.context.prompt else None,
                    "references_count": len(final_state.context.references) if final_state.context.references else 0,
                    "execution_time_ms": execution_time
                },
                success=True
            )
            
            logger.info(f"Workflow completed for request {request_id}")
            return final_state
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            # Log workflow failure
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"error": error_msg},
                decision_reasoning="Workflow failed due to unexpected error, returning failed state",
                output_data={"error_type": type(e).__name__, "error_message": error_msg},
                confidence_score=0.0,
                execution_time_ms=execution_time,
                error=error_msg
            )
            
            # Complete decision tracking with failure
            decision_logger.complete_decision(
                request_id=request_id,
                final_result={"error": error_msg, "execution_time_ms": execution_time},
                success=False
            )
            
            logger.error(f"Workflow failed for request {request_id}: {e}")
            initial_state.status = GenerationStatus.FAILED
            return initial_state
    
    async def _parse_prompt_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse prompt node in the workflow.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after prompt parsing
        """
        workflow_state = state["state"]
        original_prompt = state["original_prompt"]
        request_id = workflow_state.request_id
        
        start_time = time.time()
        
        # Log node entry decision
        decision_logger.log_decision_step(
            request_id=request_id,
            decision_type=DecisionType.WORKFLOW_PLANNING,
            input_data={
                "node": "parse_prompt",
                "prompt": original_prompt,
                "current_step": workflow_state.current_step.value
            },
            decision_reasoning="Entering parse_prompt node to extract structured information using GPT-4o",
            output_data={"node_started": "parse_prompt"},
            confidence_score=1.0,
            metadata={"workflow_node": "parse_prompt"}
        )
        
        logger.info(f"Parsing prompt for request {workflow_state.request_id}")
        
        try:
            # Parse the prompt using GPT-4o
            parsed_prompt = await self.prompt_parser.parse_prompt(original_prompt)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Update workflow state
            workflow_state.context.prompt = parsed_prompt
            workflow_state.current_step = WorkflowStep.RETRIEVE_REFERENCES
            
            # Log successful parsing decision
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.PROMPT_PARSING,
                input_data={"original_prompt": original_prompt},
                decision_reasoning=f"Successfully parsed prompt with {len(parsed_prompt.references)} references for image retrieval",
                output_data={
                    "intent": parsed_prompt.intent,
                    "entities_count": len(parsed_prompt.entities),
                    "modifiers_count": len(parsed_prompt.modifiers),
                    "references_count": len(parsed_prompt.references),
                    "confidence": parsed_prompt.confidence
                },
                confidence_score=parsed_prompt.confidence,
                execution_time_ms=execution_time,
                metadata={
                    "parsing_agent": "PromptParsingAgent",
                    "has_references": len(parsed_prompt.references) > 0
                }
            )
            
            logger.info(f"Prompt parsed successfully with {len(parsed_prompt.references)} references")
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            # Log parsing failure decision
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"error": error_msg, "node": "parse_prompt"},
                decision_reasoning="Prompt parsing failed, continuing with empty parsed prompt to maintain workflow",
                output_data={
                    "fallback_intent": "generate_image",
                    "fallback_confidence": 0.0
                },
                confidence_score=0.0,
                execution_time_ms=execution_time,
                error=error_msg,
                metadata={"recovery_strategy": "empty_prompt_fallback"}
            )
            
            logger.error(f"Prompt parsing failed: {e}")
            # Continue with empty parsed prompt
            workflow_state.context.prompt = ParsedPrompt(
                intent="generate_image",
                entities=[],
                modifiers=[],
                references=[],
                confidence=0.0
            )
            workflow_state.current_step = WorkflowStep.RETRIEVE_REFERENCES
        
        return {"state": workflow_state, "original_prompt": original_prompt}
    
    async def _retrieve_references_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve references node in the workflow.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after reference retrieval
        """
        workflow_state = state["state"]
        request_id = workflow_state.request_id
        
        start_time = time.time()
        
        # Log node entry decision
        decision_logger.log_decision_step(
            request_id=request_id,
            decision_type=DecisionType.WORKFLOW_PLANNING,
            input_data={
                "node": "retrieve_references",
                "current_step": workflow_state.current_step.value
            },
            decision_reasoning="Entering retrieve_references node to find reference images for parsed references",
            output_data={"node_started": "retrieve_references"},
            confidence_score=1.0,
            metadata={"workflow_node": "retrieve_references"}
        )
        
        logger.info(f"Retrieving references for request {workflow_state.request_id}")
        
        try:
            parsed_prompt = workflow_state.context.prompt
            
            if parsed_prompt and parsed_prompt.references:
                # Log reference retrieval decision
                decision_logger.log_decision_step(
                    request_id=request_id,
                    decision_type=DecisionType.REFERENCE_RETRIEVAL,
                    input_data={
                        "references_to_retrieve": [ref.text for ref in parsed_prompt.references],
                        "reference_types": [ref.type.value for ref in parsed_prompt.references]
                    },
                    decision_reasoning=f"Found {len(parsed_prompt.references)} references to retrieve images for",
                    output_data={"retrieval_initiated": True},
                    confidence_score=0.8,
                    metadata={"retrieval_agent": "ReferenceRetrievalAgent"}
                )
                
                # Retrieve reference images
                updated_references = await self.reference_retriever.retrieve_references(
                    parsed_prompt.references
                )
                
                execution_time = (time.time() - start_time) * 1000
                
                # Update context with retrieved references
                workflow_state.context.references = updated_references
                
                # Count successful retrievals
                successful_retrievals = sum(
                    1 for ref in updated_references 
                    if hasattr(ref, 'image_urls') and ref.image_urls
                )
                
                # Log retrieval results
                decision_logger.log_decision_step(
                    request_id=request_id,
                    decision_type=DecisionType.REFERENCE_RETRIEVAL,
                    input_data={"references_processed": len(updated_references)},
                    decision_reasoning=f"Successfully retrieved images for {successful_retrievals}/{len(updated_references)} references",
                    output_data={
                        "total_references": len(updated_references),
                        "successful_retrievals": successful_retrievals,
                        "retrieval_success_rate": successful_retrievals / len(updated_references) if updated_references else 0
                    },
                    confidence_score=successful_retrievals / len(updated_references) if updated_references else 0,
                    execution_time_ms=execution_time,
                    metadata={"image_search_completed": True}
                )
                
                logger.info(f"Retrieved images for {len(updated_references)} references")
            else:
                execution_time = (time.time() - start_time) * 1000
                
                # Log no references decision
                decision_logger.log_decision_step(
                    request_id=request_id,
                    decision_type=DecisionType.REFERENCE_RETRIEVAL,
                    input_data={"parsed_prompt_has_references": False},
                    decision_reasoning="No references found in parsed prompt, skipping image retrieval",
                    output_data={"references_to_retrieve": 0},
                    confidence_score=1.0,
                    execution_time_ms=execution_time,
                    metadata={"skip_reason": "no_references"}
                )
                
                logger.info("No references found in parsed prompt")
                workflow_state.context.references = []
            
            workflow_state.current_step = WorkflowStep.FINALIZE
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            # Log retrieval failure decision
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"error": error_msg, "node": "retrieve_references"},
                decision_reasoning="Reference retrieval failed, continuing with empty references",
                output_data={"fallback_references": []},
                confidence_score=0.0,
                execution_time_ms=execution_time,
                error=error_msg,
                metadata={"recovery_strategy": "empty_references_fallback"}
            )
            
            logger.error(f"Reference retrieval failed: {e}")
            workflow_state.context.references = []
            workflow_state.current_step = WorkflowStep.FINALIZE
        
        return {"state": workflow_state}
    
    async def _finalize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize workflow node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Final workflow state
        """
        workflow_state = state["state"]
        request_id = workflow_state.request_id
        
        start_time = time.time()
        
        # Log finalization decision
        decision_logger.log_decision_step(
            request_id=request_id,
            decision_type=DecisionType.WORKFLOW_PLANNING,
            input_data={
                "node": "finalize",
                "current_step": workflow_state.current_step.value
            },
            decision_reasoning="Finalizing workflow and preparing summary of all processing results",
            output_data={"node_started": "finalize"},
            confidence_score=1.0,
            metadata={"workflow_node": "finalize"}
        )
        
        logger.info(f"Finalizing workflow for request {workflow_state.request_id}")
        
        # Update final step
        workflow_state.current_step = WorkflowStep.FINALIZE
        
        # Prepare summary data
        parsed_prompt = workflow_state.context.prompt
        references = workflow_state.context.references
        
        execution_time = (time.time() - start_time) * 1000
        
        if parsed_prompt:
            # Log workflow summary
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.WORKFLOW_PLANNING,
                input_data={"finalization_complete": True},
                decision_reasoning="Workflow finalized successfully with complete processing results",
                output_data={
                    "intent": parsed_prompt.intent,
                    "entities_count": len(parsed_prompt.entities),
                    "modifiers_count": len(parsed_prompt.modifiers),
                    "references_with_images": len([r for r in references if hasattr(r, 'image_urls') and r.image_urls]) if references else 0,
                    "workflow_status": "completed"
                },
                confidence_score=parsed_prompt.confidence,
                execution_time_ms=execution_time,
                metadata={
                    "processing_complete": True,
                    "ready_for_generation": True
                }
            )
            
            logger.info(
                f"Workflow summary - Intent: {parsed_prompt.intent}, "
                f"Entities: {len(parsed_prompt.entities)}, "
                f"Modifiers: {len(parsed_prompt.modifiers)}, "
                f"References: {len(references)} with images"
            )
        
        return {"state": workflow_state}
    
    async def process_batch(
        self, 
        prompts: List[str]
    ) -> List[WorkflowState]:
        """
        Process multiple prompts in batch for testing.
        
        Args:
            prompts: List of prompt strings
            
        Returns:
            List of WorkflowState objects
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                request_id = f"batch_{i}_{uuid.uuid4().hex[:8]}"
                result = await self.process_prompt(prompt, request_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process batch prompt {i}: {e}")
                # Add failed state
                failed_state = WorkflowState(
                    request_id=f"batch_{i}_failed",
                    current_step=WorkflowStep.PARSE_PROMPT,
                    steps=[WorkflowStep.PARSE_PROMPT],
                    context=WorkflowContext(),
                    status=GenerationStatus.FAILED
                )
                results.append(failed_state)
        
        return results 