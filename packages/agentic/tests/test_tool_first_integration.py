"""
Integration Tests for Tool-First Architecture with Real Images and API Calls

Tests the complete tool-first workflow using:
- Real high-quality test images from URLs
- Actual OpenAI GPT-4o API calls
- Real Replicate Flux 1.1 Pro Ultra API calls
- Perplexity API for search
- End-to-end workflow validation
"""

import pytest
import asyncio
import os
import time
from typing import List, Dict, Any
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pic_arcade_agentic.agents.tool_agent import ToolFirstAgent
from pic_arcade_agentic.tools import ToolRegistry
from pic_arcade_agentic.tools.image_tools import FluxKontextMaxTool, StyleTransferTool, ObjectChangeTool, TextEditingTool, BackgroundSwapTool, CharacterConsistencyTool
from pic_arcade_agentic.tools.workflow_tools import WorkflowPlanningTool, WorkflowExecutorTool
from pic_arcade_agentic.tools.base import ToolResult


class TestToolFirstIntegration:
    """Integration test suite using real images and API calls."""
    
    @pytest.fixture
    def real_test_images(self) -> Dict[str, str]:
        """High-quality test images from reliable sources."""
        return {
            "portrait": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=800&fit=crop",
            "landscape": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop",
            "business_headshot": "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=600&h=800&fit=crop",
            "street_scene": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=1200&h=800&fit=crop",
            "architecture": "https://images.unsplash.com/photo-1513584684374-8bab748fbf90?w=1200&h=900&fit=crop",
            "person_full_body": "https://images.unsplash.com/photo-1552058544-f2b08422138a?w=600&h=900&fit=crop",
            "food_photo": "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=800&h=800&fit=crop",
            "product_shot": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=800&h=800&fit=crop",
            "nature_scene": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=1200&h=800&fit=crop",
            "urban_lifestyle": "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=800&h=1000&fit=crop"
        }
    
    @pytest.fixture
    def agent(self):
        """Create tool-first agent that manages its own tool registry."""
        return ToolFirstAgent()
    
    @pytest.mark.asyncio
    async def test_verify_test_images_accessible(self, real_test_images):
        """Verify all test images are accessible before running tests."""
        async with httpx.AsyncClient(timeout=30) as client:
            for name, url in real_test_images.items():
                try:
                    response = await client.head(url)
                    assert response.status_code == 200, f"Image {name} not accessible: {url}"
                    
                    content_type = response.headers.get("content-type", "")
                    assert "image" in content_type, f"URL {name} is not an image: {content_type}"
                    
                except Exception as e:
                    pytest.fail(f"Failed to access test image {name}: {e}")
    
    @pytest.mark.asyncio
    async def test_style_transfer_real_image(self, agent, real_test_images):
        """Test style transfer with real portrait image."""
        test_image = real_test_images["portrait"]
        
        request = f"Transform this portrait to watercolor painting style: {test_image}"
        
        start_time = time.time()
        result = await agent.process_request(request)
        processing_time = time.time() - start_time
        
        # Validate successful processing
        assert result.get("success", False), f"Style transfer failed: {result.get('error', 'Unknown error')}"
        
        # Check if we have execution results with images
        execution_results = result.get("execution_results", {})
        if "images" in execution_results:
            images = execution_results["images"]
            assert len(images) > 0
            
            # Validate output image
            output_image = images[0]
            assert "url" in output_image
            assert output_image["url"].startswith("http")
            
            print(f"\n✅ Style Transfer Test Results:")
            print(f"   Processing Time: {processing_time:.1f}s")
            print(f"   Output URL: {output_image['url']}")
            
            if "width" in output_image and "height" in output_image:
                print(f"   Dimensions: {output_image['width']}x{output_image['height']}")
        else:
            print(f"\n⚠️ Style transfer completed but no images in result")
            print(f"   Processing Time: {processing_time:.1f}s")
        
        # Performance validation
        assert processing_time < 120, f"Style transfer took too long: {processing_time}s"
    
    @pytest.mark.asyncio
    async def test_object_change_real_image(self, agent, real_test_images):
        """Test object modification with real person image."""
        test_image = real_test_images["business_headshot"]
        
        request = f"Change the hair color to blonde in this photo: {test_image}"
        
        start_time = time.time()
        result = await agent.process_request(request)
        processing_time = time.time() - start_time
        
        # Validate successful processing
        assert result.get("success", False), f"Object change failed: {result.get('error', 'Unknown error')}"
        
        # Check execution results
        execution_results = result.get("execution_results", {})
        if "images" in execution_results or "final_outputs" in execution_results:
            print(f"\n✅ Object Change Test Results:")
            print(f"   Processing Time: {processing_time:.1f}s")
            print(f"   Workflow completed successfully")
        else:
            print(f"\n⚠️ Object change completed but no images in result")
            print(f"   Processing Time: {processing_time:.1f}s")
        
        # Performance validation
        assert processing_time < 120, f"Object change took too long: {processing_time}s"
    
    @pytest.mark.asyncio
    async def test_background_swap_real_image(self, agent, real_test_images):
        """Test background replacement with real image."""
        test_image = real_test_images["person_full_body"]
        
        request = f"Change the background to a tropical beach scene: {test_image}"
        
        start_time = time.time()
        result = await agent.process_request(request)
        processing_time = time.time() - start_time
        
        # Validate successful processing
        assert result.get("success", False), f"Background swap failed: {result.get('error', 'Unknown error')}"
        
        # Check execution results
        execution_results = result.get("execution_results", {})
        if "images" in execution_results or "final_outputs" in execution_results:
            print(f"\n✅ Background Swap Test Results:")
            print(f"   Processing Time: {processing_time:.1f}s")
            print(f"   Workflow completed successfully")
        else:
            print(f"\n⚠️ Background swap completed but no images in result")
            print(f"   Processing Time: {processing_time:.1f}s")
        
        # Performance validation
        assert processing_time < 120, f"Background swap took too long: {processing_time}s"
    
    @pytest.mark.asyncio
    async def test_text_editing_real_image(self, agent, real_test_images):
        """Test text replacement with real signage image."""
        test_image = real_test_images["street_scene"]
        
        request = f"Replace any text in this image with 'PIC ARCADE': {test_image}"
        
        start_time = time.time()
        result = await agent.process_request(request)
        processing_time = time.time() - start_time
        
        # Validate processing (may not succeed if no text detected)
        if result.get("success", False):
            execution_results = result.get("execution_results", {})
            if "images" in execution_results or "final_outputs" in execution_results:
                print(f"\n✅ Text Editing Test Results:")
                print(f"   Processing Time: {processing_time:.1f}s")
                print(f"   Workflow completed successfully")
            else:
                print(f"\n⚠️ Text editing completed but no images in result")
                print(f"   Processing Time: {processing_time:.1f}s")
        else:
            print(f"\n⚠️ Text Editing Test - Processing failed: {result.get('error', 'Unknown error')}")
    
    @pytest.mark.asyncio
    async def test_character_consistency_real_image(self, agent, real_test_images):
        """Test character consistency with real person image."""
        test_image = real_test_images["portrait"]
        
        request = f"Generate this person in different poses (sitting, standing) while maintaining their appearance: {test_image}"
        
        start_time = time.time()
        result = await agent.process_request(request)
        processing_time = time.time() - start_time
        
        # Validate successful processing
        assert result.get("success", False), f"Character consistency failed: {result.get('error', 'Unknown error')}"
        
        # Check execution results
        execution_results = result.get("execution_results", {})
        if "images" in execution_results or "final_outputs" in execution_results:
            print(f"\n✅ Character Consistency Test Results:")
            print(f"   Processing Time: {processing_time:.1f}s")
            print(f"   Workflow completed successfully")
        else:
            print(f"\n⚠️ Character consistency completed but no images in result")
            print(f"   Processing Time: {processing_time:.1f}s")
        
        # Performance validation
        assert processing_time < 120, f"Character consistency took too long: {processing_time}s"
    
    @pytest.mark.asyncio
    async def test_complex_multi_step_workflow(self, agent, real_test_images):
        """Test complex multi-step editing workflow with real image."""
        test_image = real_test_images["business_headshot"]
        
        request = f"""Transform this corporate headshot with multiple changes: 
        1. Convert to watercolor painting style
        2. Change hair to blonde curly
        3. Add a tropical beach background
        4. Ensure professional quality
        
        Image: {test_image}"""
        
        start_time = time.time()
        result = await agent.process_request(request)
        processing_time = time.time() - start_time
        
        # Validate workflow execution
        assert result.get("success", False), f"Complex workflow failed: {result.get('error', 'Unknown error')}"
        
        # Check workflow metadata
        workflow_plan = result.get("workflow_plan", {})
        metadata = result.get("metadata", {})
        
        print(f"\n✅ Complex Workflow Test Results:")
        print(f"   Processing Time: {processing_time:.1f}s")
        
        if "workflow_plan" in workflow_plan:
            steps = workflow_plan["workflow_plan"]
            print(f"   Workflow Steps: {len(steps)}")
            for i, step in enumerate(steps, 1):
                print(f"   Step {i}: {step.get('tool_name', 'unknown')}")
        
        if "tools_used" in metadata:
            print(f"   Tools Used: {metadata['tools_used']}")
        
        # Check for final output
        execution_results = result.get("execution_results", {})
        if "images" in execution_results and execution_results["images"]:
            output_image = execution_results["images"][0]
            print(f"   Final Output: {output_image.get('url', 'No URL')}")
        
        # Performance validation
        assert processing_time < 180, f"Complex workflow took too long: {processing_time}s"
    
    @pytest.mark.asyncio
    async def test_image_generation_from_scratch(self, agent):
        """Test pure image generation without input image."""
        request = "Generate a professional portrait of a businesswoman in modern office setting, photorealistic style"
        
        start_time = time.time()
        result = await agent.process_request(request)
        processing_time = time.time() - start_time
        
        # Validate successful generation
        assert result.get("success", False), f"Image generation failed: {result.get('error', 'Unknown error')}"
        
        # Check execution results
        execution_results = result.get("execution_results", {})
        if "images" in execution_results or "final_outputs" in execution_results:
            print(f"\n✅ Image Generation Test Results:")
            print(f"   Processing Time: {processing_time:.1f}s")
            print(f"   Workflow completed successfully")
        else:
            print(f"\n⚠️ Image generation completed but no images in result")
            print(f"   Processing Time: {processing_time:.1f}s")
        
        # Performance validation
        assert processing_time < 120, f"Image generation took too long: {processing_time}s"
    
    @pytest.mark.asyncio
    async def test_batch_image_processing(self, agent, real_test_images):
        """Test processing multiple images in sequence."""
        test_requests = [
            f"Apply oil painting style to this portrait: {real_test_images['portrait']}",
            f"Change to a futuristic cityscape background: {real_test_images['business_headshot']}",
            f"Convert to black and white vintage style: {real_test_images['landscape']}"
        ]
        
        results = []
        total_start_time = time.time()
        
        for i, request in enumerate(test_requests):
            print(f"\nProcessing batch item {i+1}/3...")
            start_time = time.time()
            
            result = await agent.process_request(request)
            processing_time = time.time() - start_time
            
            results.append({
                "success": result.get("success", False),
                "processing_time": processing_time,
                "has_output": bool(result.get("execution_results", {}))
            })
            
            # Small delay to avoid API rate limits
            await asyncio.sleep(2)
        
        total_time = time.time() - total_start_time
        
        # Validate batch results
        successful_count = sum(1 for r in results if r["success"])
        assert successful_count >= 2, f"Only {successful_count}/3 batch items succeeded"
        
        avg_processing_time = sum(r["processing_time"] for r in results) / len(results)
        
        print(f"\n✅ Batch Processing Test Results:")
        print(f"   Success Rate: {successful_count}/3 ({successful_count/3:.1%})")
        print(f"   Total Time: {total_time:.1f}s")
        print(f"   Average Processing Time: {avg_processing_time:.1f}s")
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_image(self, agent):
        """Test error handling with invalid image URLs."""
        invalid_requests = [
            "Apply style transfer to this image: https://invalid-url.com/fake.jpg",
            "Change background: not-a-url",
            "Edit this image: https://httpstat.us/404"
        ]
        
        for request in invalid_requests:
            result = await agent.process_request(request)
            
            # Should handle gracefully without crashing
            assert isinstance(result, dict)
            if not result.get("success", False):
                assert "error" in result
                print(f"✅ Properly handled invalid request: {result['error']}")
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, agent, real_test_images):
        """Test performance benchmarks for different operations."""
        test_cases = [
            ("style_transfer", f"Convert to impressionist painting: {real_test_images['portrait']}"),
            ("object_change", f"Change shirt color to red: {real_test_images['business_headshot']}"),
            ("background_swap", f"Add mountain landscape background: {real_test_images['person_full_body']}"),
            ("generation", "Generate a sunset over mountains, photorealistic")
        ]
        
        performance_results = []
        
        for operation_type, request in test_cases:
            print(f"\nBenchmarking {operation_type}...")
            
            start_time = time.time()
            result = await agent.process_request(request)
            processing_time = time.time() - start_time
            
            performance_results.append({
                "operation": operation_type,
                "success": result.get("success", False),
                "time": processing_time,
                "has_output": bool(result.get("execution_results", {}))
            })
            
            # Rate limiting
            await asyncio.sleep(3)
        
        # Performance validation
        successful_ops = [r for r in performance_results if r["success"]]
        assert len(successful_ops) >= 3, "At least 3/4 operations should succeed"
        
        avg_time = sum(r["time"] for r in successful_ops) / len(successful_ops)
        assert avg_time < 90, f"Average processing time {avg_time:.1f}s too slow"
        
        print(f"\n✅ Performance Benchmark Results:")
        print(f"   Success Rate: {len(successful_ops)}/4")
        print(f"   Average Time: {avg_time:.1f}s")
        
        for result in performance_results:
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {result['operation']}: {result['time']:.1f}s")
    
    @pytest.mark.asyncio
    async def test_api_key_validation(self):
        """Test that all required API keys are properly configured."""
        required_keys = [
            ("REPLICATE_API_TOKEN", "Flux image generation"),
            ("OPENAI_API_KEY", "GPT-4o prompt processing"),
        ]
        
        missing_keys = []
        
        for key_name, purpose in required_keys:
            key_value = os.getenv(key_name)
            if not key_value or len(key_value) < 10:
                missing_keys.append((key_name, purpose))
        
        if missing_keys:
            missing_list = "\n".join([f"  - {key}: {purpose}" for key, purpose in missing_keys])
            pytest.skip(f"Missing required API keys:\n{missing_list}")
        
        print("✅ All required API keys are configured") 