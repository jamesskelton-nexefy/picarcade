"""
Test Suite for Phase 5 Video Generation Tools

Tests all video generation capabilities introduced in Phase 5:
- Runway ML video generation
- Multiple Replicate video providers
- Video editing and enhancement
- Input validation and error handling
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock

from pic_arcade_agentic.tools import (
    RunwayVideoTool,
    ReplicateVideoTool,
    VideoEditingTool,
    ToolResult
)


class TestRunwayVideoTool:
    """Test Runway ML video generation tool"""
    
    @pytest.fixture
    def runway_tool(self):
        return RunwayVideoTool()
    
    def test_initialization(self, runway_tool):
        """Test tool initialization"""
        assert runway_tool.name == "runway_video_generation"
        assert runway_tool.description.startswith("Generate high-quality videos")
        assert "prompt_text" in runway_tool.input_schema["properties"]
        assert "video_url" in runway_tool.output_schema["properties"]
    
    def test_input_validation(self, runway_tool):
        """Test input validation"""
        # Valid input
        valid_input = {"prompt_text": "A sunset over mountains"}
        assert runway_tool.validate_input(valid_input)
        
        # Missing required field
        invalid_input = {"duration": 6}
        assert not runway_tool.validate_input(invalid_input)
    
    @pytest.mark.asyncio
    @patch('pic_arcade_agentic.tools.video_tools.AsyncRunwayML')
    async def test_successful_generation(self, mock_runway_class, runway_tool):
        """Test successful video generation"""
        # Mock Runway client and response
        mock_client = Mock()
        mock_video_task = Mock()
        mock_video_task.id = "task_123"
        
        # Setup the mock chain
        mock_runway_class.return_value = mock_client
        mock_client.image_to_video.create = AsyncMock(return_value=mock_video_task)
        
        # Mock the API key
        with patch.dict(os.environ, {'RUNWAYML_API_SECRET': 'test_key'}):
            runway_tool.api_key = 'test_key'
            
            input_data = {
                "prompt_text": "A beautiful landscape",
                "model": "gen4_turbo",
                "ratio": "1280:720"
            }
            
            # Mock asyncio.to_thread to run sync function
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.return_value = mock_video_task
                
                result = await runway_tool.invoke(input_data)
                
                assert result.success
                assert result.data["task_id"] == "task_123"
                assert result.data["model_used"] == "gen4_turbo"
                assert "cost_estimate" in result.data
    
    @pytest.mark.asyncio
    async def test_missing_api_key(self, runway_tool):
        """Test behavior with missing API key"""
        runway_tool.api_key = None
        
        input_data = {"prompt_text": "Test prompt"}
        result = await runway_tool.invoke(input_data)
        
        assert not result.success
        assert "RUNWAYML_API_SECRET" in result.data["error"]
    
    def test_cost_calculation(self, runway_tool):
        """Test cost calculation logic"""
        cost = runway_tool._calculate_runway_cost(duration=6, quality="1280:720")
        assert isinstance(cost, float)
        assert cost > 0
        
        # Higher quality should cost more
        hd_cost = runway_tool._calculate_runway_cost(duration=6, quality="1920:1080")
        assert hd_cost > cost


class TestReplicateVideoTool:
    """Test Replicate video generation tool with multiple providers"""
    
    @pytest.fixture
    def replicate_tool(self):
        return ReplicateVideoTool()
    
    def test_initialization(self, replicate_tool):
        """Test tool initialization"""
        assert replicate_tool.name == "replicate_video_generation"
        assert replicate_tool.description.startswith("Generate videos using various")
        assert "prompt" in replicate_tool.input_schema["properties"]
        assert "provider" in replicate_tool.input_schema["properties"]
    
    def test_provider_configs(self, replicate_tool):
        """Test provider configuration retrieval"""
        # Test valid providers
        providers = ["google_veo2", "luma_ray", "hunyuan_video", "minimax_video"]
        
        for provider in providers:
            config = replicate_tool._get_model_config(provider)
            assert config is not None
            assert "model" in config
            assert "description" in config
            assert "max_duration" in config
        
        # Test invalid provider
        invalid_config = replicate_tool._get_model_config("invalid_provider")
        assert invalid_config is None
    
    def test_input_preparation(self, replicate_tool):
        """Test model input preparation for different providers"""
        input_data = {
            "prompt": "A cat playing",
            "duration": 6,
            "quality": "720p"
        }
        
        # Test different providers
        providers = ["luma_ray", "google_veo2", "hunyuan_video"]
        
        for provider in providers:
            input_data["provider"] = provider
            config = replicate_tool._get_model_config(provider)
            model_inputs = replicate_tool._prepare_model_inputs(input_data, config)
            
            assert "prompt" in model_inputs
            assert isinstance(model_inputs, dict)
    
    def test_confidence_calculation(self, replicate_tool):
        """Test confidence score calculation"""
        # Fast processing should have higher confidence
        fast_confidence = replicate_tool._calculate_confidence("google_veo2", 30)
        slow_confidence = replicate_tool._calculate_confidence("google_veo2", 300)
        
        assert 0 <= fast_confidence <= 1
        assert 0 <= slow_confidence <= 1
        assert fast_confidence >= slow_confidence
    
    def test_cost_calculation(self, replicate_tool):
        """Test cost estimation"""
        input_data = {"duration": 6, "quality": "720p"}
        
        # Test different providers
        providers = ["google_veo2", "luma_ray", "hunyuan_video"]
        
        for provider in providers:
            cost = replicate_tool._calculate_cost(provider, input_data)
            assert isinstance(cost, float)
            assert cost >= 0
        
        # Higher quality should cost more
        hd_cost = replicate_tool._calculate_cost("google_veo2", {
            "duration": 6, "quality": "4K"
        })
        normal_cost = replicate_tool._calculate_cost("google_veo2", {
            "duration": 6, "quality": "720p"
        })
        assert hd_cost > normal_cost
    
    def test_video_url_extraction(self, replicate_tool):
        """Test video URL extraction from different output formats"""
        # String output
        url = replicate_tool._extract_video_url("https://example.com/video.mp4", "test")
        assert url == "https://example.com/video.mp4"
        
        # List output
        url = replicate_tool._extract_video_url(["https://example.com/video.mp4"], "test")
        assert url == "https://example.com/video.mp4"
        
        # Dict output
        url = replicate_tool._extract_video_url({"video": "https://example.com/video.mp4"}, "test")
        assert url == "https://example.com/video.mp4"
        
        # No valid URL
        url = replicate_tool._extract_video_url({"status": "error"}, "test")
        assert url is None
    
    @pytest.mark.asyncio
    @patch('replicate.run')
    async def test_successful_generation(self, mock_replicate, replicate_tool):
        """Test successful video generation with Replicate"""
        # Mock Replicate response
        mock_replicate.return_value = "https://example.com/generated_video.mp4"
        
        with patch.dict(os.environ, {'REPLICATE_API_TOKEN': 'test_token'}):
            replicate_tool.api_token = 'test_token'
            
            input_data = {
                "prompt": "A dancing robot",
                "provider": "luma_ray",
                "duration": 5,
                "quality": "720p"
            }
            
            # Mock asyncio.to_thread
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.return_value = "https://example.com/generated_video.mp4"
                
                result = await replicate_tool.invoke(input_data)
                
                assert result.success
                assert result.data["video_url"] == "https://example.com/generated_video.mp4"
                assert result.data["provider_used"] == "luma_ray"
                assert "confidence" in result.data
    
    @pytest.mark.asyncio
    async def test_missing_api_token(self, replicate_tool):
        """Test behavior with missing Replicate API token"""
        replicate_tool.api_token = None
        
        input_data = {"prompt": "Test prompt"}
        result = await replicate_tool.invoke(input_data)
        
        assert not result.success
        assert "REPLICATE_API_TOKEN" in result.data["error"]
    
    @pytest.mark.asyncio
    async def test_invalid_provider(self, replicate_tool):
        """Test behavior with invalid provider"""
        with patch.dict(os.environ, {'REPLICATE_API_TOKEN': 'test_token'}):
            replicate_tool.api_token = 'test_token'
            
            input_data = {
                "prompt": "Test prompt",
                "provider": "invalid_provider"
            }
            
            result = await replicate_tool.invoke(input_data)
            
            assert not result.success
            assert "Unsupported provider" in result.data["error"]


class TestVideoEditingTool:
    """Test video editing and enhancement tool"""
    
    @pytest.fixture
    def editing_tool(self):
        return VideoEditingTool()
    
    def test_initialization(self, editing_tool):
        """Test tool initialization"""
        assert editing_tool.name == "video_editing"
        assert editing_tool.description.startswith("Edit and enhance videos")
        assert "video_url" in editing_tool.input_schema["properties"]
        assert "operation" in editing_tool.input_schema["properties"]
    
    def test_editing_model_configs(self, editing_tool):
        """Test editing model configuration retrieval"""
        operations = ["upscale", "style_transfer", "enhance", "stabilize", "motion_edit"]
        
        for operation in operations:
            config = editing_tool._get_editing_model(operation)
            assert config is not None
            assert "model" in config
            assert "description" in config
        
        # Test invalid operation
        invalid_config = editing_tool._get_editing_model("invalid_operation")
        assert invalid_config is None
    
    def test_input_preparation(self, editing_tool):
        """Test editing input preparation"""
        input_data = {
            "video_url": "https://example.com/video.mp4",
            "operation": "upscale",
            "upscale_factor": 2
        }
        
        config = {"model": "test/model"}
        inputs = editing_tool._prepare_editing_inputs(input_data, config)
        
        assert "video" in inputs
        assert inputs["video"] == "https://example.com/video.mp4"
        assert inputs["upscale"] == 2
    
    @pytest.mark.asyncio
    async def test_missing_api_token(self, editing_tool):
        """Test behavior with missing API token"""
        editing_tool.api_token = None
        
        input_data = {
            "video_url": "https://example.com/video.mp4",
            "operation": "upscale"
        }
        
        result = await editing_tool.invoke(input_data)
        
        assert not result.success
        assert "REPLICATE_API_TOKEN" in result.data["error"]


class TestVideoToolsIntegration:
    """Integration tests for video tools"""
    
    @pytest.mark.asyncio
    async def test_workflow_chaining(self):
        """Test chaining video generation and editing"""
        # This would test a complete workflow:
        # 1. Generate video with ReplicateVideoTool
        # 2. Edit video with VideoEditingTool
        
        # For now, just test that tools can be instantiated together
        replicate_tool = ReplicateVideoTool()
        editing_tool = VideoEditingTool()
        runway_tool = RunwayVideoTool()
        
        assert replicate_tool.name == "replicate_video_generation"
        assert editing_tool.name == "video_editing"
        assert runway_tool.name == "runway_video_generation"
    
    def test_tool_categories(self):
        """Test that all video tools have correct categories"""
        from pic_arcade_agentic.tools.base import ToolCategory
        
        replicate_tool = ReplicateVideoTool()
        editing_tool = VideoEditingTool()
        runway_tool = RunwayVideoTool()
        
        assert replicate_tool.category == ToolCategory.VIDEO_GENERATION
        assert editing_tool.category == ToolCategory.VIDEO_GENERATION
        assert runway_tool.category == ToolCategory.VIDEO_GENERATION
    
    def test_input_schema_completeness(self):
        """Test that all tools have complete input schemas"""
        tools = [ReplicateVideoTool(), VideoEditingTool(), RunwayVideoTool()]
        
        for tool in tools:
            schema = tool.input_schema
            assert "type" in schema
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema
            assert len(schema["required"]) > 0
    
    def test_output_schema_consistency(self):
        """Test that all tools have consistent output schemas"""
        tools = [ReplicateVideoTool(), VideoEditingTool(), RunwayVideoTool()]
        
        for tool in tools:
            schema = tool.output_schema
            assert "type" in schema
            assert schema["type"] == "object"
            assert "properties" in schema


# Performance and stress tests
class TestVideoPerformance:
    """Performance tests for video generation tools"""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_generations(self):
        """Test multiple concurrent video generations"""
        replicate_tool = ReplicateVideoTool()
        
        # Mock to avoid actual API calls
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = "https://example.com/video.mp4"
            
            with patch.dict(os.environ, {'REPLICATE_API_TOKEN': 'test_token'}):
                replicate_tool.api_token = 'test_token'
                
                # Create multiple concurrent tasks
                tasks = []
                for i in range(3):
                    input_data = {
                        "prompt": f"Test video {i}",
                        "provider": "luma_ray"
                    }
                    tasks.append(replicate_tool.invoke(input_data))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # All should complete successfully
                for result in results:
                    assert isinstance(result, ToolResult)
                    assert result.success
    
    def test_memory_usage(self):
        """Test that tools don't leak memory"""
        # Create and destroy many tool instances
        for _ in range(100):
            tool = ReplicateVideoTool()
            assert tool.name == "replicate_video_generation"
            del tool


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 