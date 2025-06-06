"""
Tests for Workflow Orchestrator (Phase 2)

Tests LangGraph workflow orchestration for:
- End-to-end prompt processing (parse → retrieve → finalize)
- State management across workflow steps
- Error handling and recovery
- Batch processing capabilities
"""

import pytest
import asyncio
from typing import List
from pic_arcade_agentic.workflow.orchestrator import WorkflowOrchestrator
from pic_arcade_agentic.types import (
    WorkflowState,
    WorkflowStep,
    GenerationStatus,
    ParsedPrompt,
    PromptReference,
    PromptReferenceType
)


class TestWorkflowOrchestrator:
    """Test suite for workflow orchestrator with real API integration."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create workflow orchestrator for testing."""
        return WorkflowOrchestrator()
    
    @pytest.fixture
    def test_prompts(self) -> List[str]:
        """Representative test prompts for workflow testing."""
        return [
            "Create a portrait of Scarlett Johansson in Renaissance style",
            "Generate Emma Stone as a warrior princess with dramatic lighting",
            "Leonardo DiCaprio in Van Gogh's impressionist style",
            "Photorealistic red Ferrari in golden hour lighting",
            "Medieval castle in the style of Lord of the Rings"
        ]
    
    @pytest.mark.asyncio
    async def test_single_prompt_workflow(self, orchestrator):
        """Test complete workflow for a single prompt."""
        prompt = "Create a portrait of Scarlett Johansson in Renaissance style"
        
        final_state = await orchestrator.process_prompt(prompt)
        
        # Validate final state
        assert isinstance(final_state, WorkflowState)
        assert final_state.status == GenerationStatus.COMPLETED
        assert final_state.current_step == WorkflowStep.FINALIZE
        assert final_state.request_id is not None
        
        # Validate workflow steps were executed
        expected_steps = [
            WorkflowStep.PARSE_PROMPT,
            WorkflowStep.RETRIEVE_REFERENCES,
            WorkflowStep.FINALIZE
        ]
        assert final_state.steps == expected_steps
        
        # Validate context contains parsed prompt
        assert final_state.context.prompt is not None
        parsed_prompt = final_state.context.prompt
        assert isinstance(parsed_prompt, ParsedPrompt)
        assert parsed_prompt.intent != ""
        assert parsed_prompt.confidence > 0.0
        
        # For celebrity prompts, should have references with images
        if parsed_prompt.references:
            references = final_state.context.references
            assert len(references) >= len(parsed_prompt.references)
            
            # At least some references should have images
            images_found = sum(len(ref.image_urls) for ref in references)
            if any(ref.type == PromptReferenceType.CELEBRITY for ref in references):
                assert images_found > 0, "No images found for celebrity references"
    
    @pytest.mark.asyncio
    async def test_workflow_state_progression(self, orchestrator):
        """Test that workflow progresses through all expected steps."""
        prompt = "Portrait of Leonardo DiCaprio in oil painting style"
        
        # We'll need to modify orchestrator to track intermediate states
        # For now, just test final state
        final_state = await orchestrator.process_prompt(prompt)
        
        # Should reach final step
        assert final_state.current_step == WorkflowStep.FINALIZE
        assert final_state.status == GenerationStatus.COMPLETED
        
        # Should have processed all steps
        assert WorkflowStep.PARSE_PROMPT in final_state.steps
        assert WorkflowStep.RETRIEVE_REFERENCES in final_state.steps
        assert WorkflowStep.FINALIZE in final_state.steps
    
    @pytest.mark.asyncio
    async def test_batch_workflow_processing(self, orchestrator, test_prompts):
        """Test batch processing of multiple prompts."""
        results = await orchestrator.process_batch(test_prompts)
        
        assert len(results) == len(test_prompts)
        
        successful_workflows = 0
        total_references = 0
        total_images = 0
        
        for i, result in enumerate(results):
            assert isinstance(result, WorkflowState)
            assert result.request_id.startswith("batch_")
            
            if result.status == GenerationStatus.COMPLETED:
                successful_workflows += 1
                
                # Should have parsed prompt
                assert result.context.prompt is not None
                
                # Count references and images
                if result.context.references:
                    total_references += len(result.context.references)
                    total_images += sum(len(ref.image_urls) for ref in result.context.references)
        
        # Require high success rate
        success_rate = successful_workflows / len(test_prompts)
        assert success_rate >= 0.8, f"Success rate {success_rate:.2f} below 80%"
        
        print(f"\nBatch Workflow Test Results:")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Total References: {total_references}")
        print(f"Total Images Found: {total_images}")
    
    @pytest.mark.asyncio
    async def test_celebrity_reference_workflow(self, orchestrator):
        """Test workflow specifically for celebrity references."""
        celebrity_prompts = [
            "Emma Stone as a medieval princess",
            "Ryan Gosling in noir detective style", 
            "Zendaya in futuristic sci-fi setting"
        ]
        
        results = await orchestrator.process_batch(celebrity_prompts)
        
        celebrity_images_found = 0
        
        for result in results:
            if result.status == GenerationStatus.COMPLETED:
                parsed_prompt = result.context.prompt
                references = result.context.references
                
                # Should detect celebrity references
                celebrity_refs = [
                    ref for ref in parsed_prompt.references 
                    if ref.type == PromptReferenceType.CELEBRITY
                ]
                
                if celebrity_refs:
                    # Should have retrieved images for celebrities
                    for ref in references:
                        if ref.type == PromptReferenceType.CELEBRITY:
                            celebrity_images_found += len(ref.image_urls)
        
        # Should find images for celebrity references
        assert celebrity_images_found > 0, "No images found for celebrity references"
    
    @pytest.mark.asyncio
    async def test_error_handling_in_workflow(self, orchestrator):
        """Test workflow error handling and recovery."""
        problematic_prompts = [
            "",  # Empty prompt
            "xyz nonsense invalid prompt 123",  # Nonsensical
            "a",  # Too short
        ]
        
        for prompt in problematic_prompts:
            result = await orchestrator.process_prompt(prompt)
            
            # Should complete without throwing exceptions
            assert isinstance(result, WorkflowState)
            
            # Should reach completion (even with fallback data)
            assert result.status in [GenerationStatus.COMPLETED, GenerationStatus.FAILED]
            
            # Should have some basic structure
            assert result.request_id is not None
            assert result.context is not None
    
    @pytest.mark.asyncio
    async def test_workflow_request_id_generation(self, orchestrator):
        """Test request ID generation and uniqueness."""
        prompt = "Test prompt for ID generation"
        
        # Test auto-generated ID
        result1 = await orchestrator.process_prompt(prompt)
        result2 = await orchestrator.process_prompt(prompt)
        
        assert result1.request_id != result2.request_id
        assert len(result1.request_id) > 10  # Should be substantial
        
        # Test custom ID
        custom_id = "custom-test-id-123"
        result3 = await orchestrator.process_prompt(prompt, custom_id)
        assert result3.request_id == custom_id
    
    @pytest.mark.asyncio
    async def test_workflow_context_preservation(self, orchestrator):
        """Test that context is preserved across workflow steps."""
        prompt = "Portrait of Margot Robbie in Renaissance painting style"
        
        final_state = await orchestrator.process_prompt(prompt)
        
        if final_state.status == GenerationStatus.COMPLETED:
            # Context should have parsed prompt
            assert final_state.context.prompt is not None
            parsed_prompt = final_state.context.prompt
            
            # Context should have references if any were found
            if parsed_prompt.references:
                assert len(final_state.context.references) >= len(parsed_prompt.references)
                
                # References should maintain their properties from parsing
                for i, context_ref in enumerate(final_state.context.references):
                    if i < len(parsed_prompt.references):
                        original_ref = parsed_prompt.references[i]
                        assert context_ref.text == original_ref.text
                        assert context_ref.type == original_ref.type
                        assert context_ref.search_query == original_ref.search_query
    
    @pytest.mark.asyncio
    async def test_workflow_performance(self, orchestrator):
        """Test workflow performance and timing."""
        import time
        
        prompt = "Create a portrait of Tom Hardy in cyberpunk style"
        
        start_time = time.time()
        result = await orchestrator.process_prompt(prompt)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust based on API latency)
        assert processing_time < 30.0, f"Workflow took {processing_time:.2f}s, too slow"
        
        # Should be successful
        assert result.status == GenerationStatus.COMPLETED
        
        print(f"Workflow processing time: {processing_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_different_prompt_types(self, orchestrator):
        """Test workflow with different types of prompts."""
        prompt_types = [
            ("Portrait prompt", "Portrait of Jennifer Lawrence"),
            ("Scene prompt", "Futuristic cityscape with neon lights"),
            ("Style prompt", "Abstract art in Picasso's cubist style"),
            ("Technical prompt", "4K photorealistic mountain landscape"),
            ("Complex prompt", "Emma Watson as Hermione in Hogwarts, magical lighting, oil painting")
        ]
        
        for prompt_type, prompt in prompt_types:
            result = await orchestrator.process_prompt(prompt)
            
            assert result.status == GenerationStatus.COMPLETED, f"Failed for {prompt_type}"
            assert result.context.prompt is not None, f"No parsed prompt for {prompt_type}"
            
            parsed_prompt = result.context.prompt
            assert parsed_prompt.intent != "", f"No intent for {prompt_type}"
            assert parsed_prompt.confidence > 0.0, f"No confidence for {prompt_type}" 