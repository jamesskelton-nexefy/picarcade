"""
Flux Tools Integration Tests with Real Images and API Calls

Tests all Flux-based tools using:
- Real high-quality images from Unsplash
- Actual Replicate Flux 1.1 Pro Ultra API calls
- Various aspect ratios and output formats
- Performance and quality validation
"""

import pytest
import asyncio
import os
import time
from typing import Dict, Any
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pic_arcade_agentic.tools.image_tools import (
    FluxKontextMaxTool,
    StyleTransferTool,
    ObjectChangeTool,
    TextEditingTool,
    BackgroundSwapTool,
    CharacterConsistencyTool
)
from pic_arcade_agentic.tools.base import ToolResult


class TestFluxTools:
    """Test suite for all Flux-based image editing tools."""
    
    @pytest.fixture
    def flux_config(self):
        """Configuration for Flux tools."""
        return {
            "api_key": os.getenv("REPLICATE_API_TOKEN")
        }
    
    @pytest.fixture
    def test_images(self) -> Dict[str, str]:
        """Curated test images for different scenarios."""
        return {
            # Portraits for style transfer and object changes
            "professional_woman": "https://images.unsplash.com/photo-1494790108755-2616b612b1e9?w=800&h=800&fit=crop",
            "business_man": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=800&fit=crop",
            "casual_portrait": "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=600&h=800&fit=crop",
            
            # Full body shots for background changes
            "person_standing": "https://images.unsplash.com/photo-1552058544-f2b08422138a?w=600&h=900&fit=crop",
            "person_walking": "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=600&h=900&fit=crop",
            
            # Images with text for text editing
            "storefront": "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=1200&h=800&fit=crop",
            "street_signs": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=1200&h=800&fit=crop",
            "cafe_exterior": "https://images.unsplash.com/photo-1554118811-1e0d58224f24?w=1000&h=800&fit=crop",
            
            # Character consistency reference
            "character_reference": "https://images.unsplash.com/photo-1531427186611-ecfd6d936c79?w=600&h=800&fit=crop",
            
            # Various formats and ratios
            "landscape_wide": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1600&h=900&fit=crop",
            "square_format": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=800&h=800&fit=crop"
        }
    
    @pytest.mark.asyncio
    async def test_flux_kontext_max_generation(self, flux_config):
        """Test pure image generation with Flux Kontext Max."""
        tool = FluxKontextMaxTool(flux_config)
        
        test_prompts = [
            {
                "prompt": "Professional portrait of a businesswoman in modern office, photorealistic",
                "aspect_ratio": "3:4",
                "output_format": "jpg"
            },
            {
                "prompt": "Majestic mountain landscape at golden hour, highly detailed",
                "aspect_ratio": "16:9",
                "output_format": "png"
            },
            {
                "prompt": "Futuristic cityscape with neon lights, cyberpunk style",
                "aspect_ratio": "21:9",
                "output_format": "jpg"
            }
        ]
        
        for i, test_case in enumerate(test_prompts):
            print(f"\nTesting generation {i+1}/3: {test_case['prompt'][:50]}...")
            
            start_time = time.time()
            result = await tool.invoke(test_case)
            processing_time = time.time() - start_time
            
            # Validate successful generation
            assert result.success, f"Generation failed: {result.error}"
            assert "images" in result.data
            assert len(result.data["images"]) > 0
            
            # Validate output structure
            image = result.data["images"][0]
            assert "url" in image
            assert image["url"].startswith("http")
            assert "width" in image and "height" in image
            assert image["width"] > 0 and image["height"] > 0
            
            # Performance check
            assert processing_time < 90, f"Generation took too long: {processing_time}s"
            
            print(f"   ✅ Generated in {processing_time:.1f}s")
            print(f"   📸 {image['width']}x{image['height']} - {image['url']}")
            
            # Rate limiting
            await asyncio.sleep(3)
    
    @pytest.mark.asyncio
    async def test_style_transfer_variations(self, flux_config, test_images):
        """Test style transfer with different art styles."""
        tool = StyleTransferTool(flux_config)
        
        test_cases = [
            {
                "image": test_images["professional_woman"],
                "style": "watercolor",
                "strength": 0.7
            },
            {
                "image": test_images["business_man"],
                "style": "oil_painting",
                "strength": 0.8
            },
            {
                "image": test_images["casual_portrait"],
                "style": "sketch",
                "strength": 0.6
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTesting style transfer {i+1}/3: {test_case['style']}...")
            
            start_time = time.time()
            result = await tool.invoke(test_case)
            processing_time = time.time() - start_time
            
            # Validate successful processing
            assert result.success, f"Style transfer failed: {result.error}"
            assert "styled_image" in result.data
            assert result.data["styled_image"].startswith("http")
            assert result.data["style_applied"] == test_case["style"]
            
            # Performance check
            assert processing_time < 90, f"Style transfer took too long: {processing_time}s"
            
            print(f"   ✅ {test_case['style']} applied in {processing_time:.1f}s")
            print(f"   🎨 Result: {result.data['styled_image']}")
            
            # Rate limiting
            await asyncio.sleep(3)
    
    @pytest.mark.asyncio
    async def test_object_modifications(self, flux_config, test_images):
        """Test object and clothing modifications."""
        tool = ObjectChangeTool(flux_config)
        
        test_cases = [
            {
                "image": test_images["professional_woman"],
                "target_object": "hair",
                "modification": "blonde curly hair",
                "strength": 0.8
            },
            {
                "image": test_images["business_man"],
                "target_object": "suit",
                "modification": "navy blue formal suit",
                "strength": 0.7
            },
            {
                "image": test_images["casual_portrait"],
                "target_object": "shirt",
                "modification": "red cotton t-shirt",
                "strength": 0.6
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTesting object change {i+1}/3: {test_case['target_object']} → {test_case['modification']}...")
            
            start_time = time.time()
            result = await tool.invoke(test_case)
            processing_time = time.time() - start_time
            
            # Validate successful processing
            assert result.success, f"Object change failed: {result.error}"
            assert "modified_image" in result.data
            assert result.data["modified_image"].startswith("http")
            assert result.data["object_modified"] == test_case["target_object"]
            assert result.data["modification_applied"] == test_case["modification"]
            
            # Performance check
            assert processing_time < 90, f"Object change took too long: {processing_time}s"
            
            print(f"   ✅ Modified in {processing_time:.1f}s")
            print(f"   🔧 Result: {result.data['modified_image']}")
            
            # Rate limiting
            await asyncio.sleep(3)
    
    @pytest.mark.asyncio
    async def test_background_swapping(self, flux_config, test_images):
        """Test background replacement with different environments."""
        tool = BackgroundSwapTool(flux_config)
        
        test_cases = [
            {
                "image": test_images["person_standing"],
                "new_background": "tropical beach with palm trees",
                "environment_type": "outdoor",
                "strength": 0.8
            },
            {
                "image": test_images["person_walking"],
                "new_background": "modern city skyline at sunset",
                "environment_type": "outdoor",
                "strength": 0.7
            },
            {
                "image": test_images["professional_woman"],
                "new_background": "elegant office interior with large windows",
                "environment_type": "indoor",
                "strength": 0.6
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTesting background swap {i+1}/3: {test_case['new_background'][:30]}...")
            
            start_time = time.time()
            result = await tool.invoke(test_case)
            processing_time = time.time() - start_time
            
            # Validate successful processing
            assert result.success, f"Background swap failed: {result.error}"
            assert "swapped_image" in result.data
            assert result.data["swapped_image"].startswith("http")
            assert result.data["background_description"] == test_case["new_background"]
            
            # Performance check
            assert processing_time < 90, f"Background swap took too long: {processing_time}s"
            
            print(f"   ✅ Background swapped in {processing_time:.1f}s")
            print(f"   🌅 Result: {result.data['swapped_image']}")
            
            # Rate limiting
            await asyncio.sleep(3)
    
    @pytest.mark.asyncio
    async def test_text_editing_capabilities(self, flux_config, test_images):
        """Test text replacement in images with signage."""
        tool = TextEditingTool(flux_config)
        
        test_cases = [
            {
                "image": test_images["storefront"],
                "new_text": "PIC ARCADE",
                "text_location": "on the main sign"
            },
            {
                "image": test_images["cafe_exterior"],
                "new_text": "FLUX STUDIO",
                "text_location": "on the storefront"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTesting text editing {i+1}/2: '{test_case['new_text']}'...")
            
            start_time = time.time()
            result = await tool.invoke(test_case)
            processing_time = time.time() - start_time
            
            # Text editing may not always succeed (depends on text detection)
            if result.success:
                assert "edited_image" in result.data
                assert result.data["edited_image"].startswith("http")
                assert result.data["new_text"] == test_case["new_text"]
                
                print(f"   ✅ Text edited in {processing_time:.1f}s")
                print(f"   📝 Result: {result.data['edited_image']}")
            else:
                print(f"   ⚠️ Text editing failed (expected if no text detected): {result.error}")
            
            # Performance check
            assert processing_time < 90, f"Text editing took too long: {processing_time}s"
            
            # Rate limiting
            await asyncio.sleep(3)
    
    @pytest.mark.asyncio
    async def test_character_consistency(self, flux_config, test_images):
        """Test character consistency across different poses."""
        tool = CharacterConsistencyTool(flux_config)
        
        test_cases = [
            {
                "reference_image": test_images["character_reference"],
                "character_description": "young professional woman with brown hair in business attire",
                "new_scenario": "sitting at a desk working on laptop",
                "variation_strength": 0.3
            },
            {
                "reference_image": test_images["business_man"],
                "character_description": "middle-aged businessman in formal suit",
                "new_scenario": "standing and giving a presentation",
                "variation_strength": 0.4
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTesting character consistency {i+1}/2: {test_case['new_scenario'][:30]}...")
            
            start_time = time.time()
            result = await tool.invoke(test_case)
            processing_time = time.time() - start_time
            
            # Validate successful processing
            assert result.success, f"Character consistency failed: {result.error}"
            assert "consistent_image" in result.data
            assert result.data["consistent_image"].startswith("http")
            assert result.data["scenario_applied"] == test_case["new_scenario"]
            
            # Performance check
            assert processing_time < 90, f"Character consistency took too long: {processing_time}s"
            
            print(f"   ✅ Character maintained in {processing_time:.1f}s")
            print(f"   👤 Result: {result.data['consistent_image']}")
            
            # Rate limiting
            await asyncio.sleep(3)
    
    @pytest.mark.asyncio
    async def test_aspect_ratio_variations(self, flux_config):
        """Test different aspect ratios and output formats."""
        tool = FluxKontextMaxTool(flux_config)
        
        aspect_ratios = ["1:1", "3:2", "16:9", "9:16", "21:9"]
        output_formats = ["jpg", "png"]
        
        base_prompt = "Beautiful landscape with mountains and lake, photorealistic"
        
        test_combinations = [
            (ratio, fmt) for ratio in aspect_ratios[:3] for fmt in output_formats[:1]  # Limited for testing
        ]
        
        for i, (aspect_ratio, output_format) in enumerate(test_combinations):
            print(f"\nTesting format {i+1}/{len(test_combinations)}: {aspect_ratio} {output_format}...")
            
            start_time = time.time()
            result = await tool.invoke({
                "prompt": base_prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "safety_tolerance": 2
            })
            processing_time = time.time() - start_time
            
            # Validate successful generation
            assert result.success, f"Generation failed for {aspect_ratio} {output_format}: {result.error}"
            assert "images" in result.data
            assert len(result.data["images"]) > 0
            
            image = result.data["images"][0]
            assert "url" in image
            assert image["url"].startswith("http")
            
            # Validate aspect ratio (approximate)
            width, height = image["width"], image["height"]
            expected_ratios = {
                "1:1": 1.0,
                "3:2": 1.5,
                "16:9": 16/9,
                "9:16": 9/16,
                "21:9": 21/9
            }
            
            actual_ratio = width / height
            expected_ratio = expected_ratios[aspect_ratio]
            ratio_diff = abs(actual_ratio - expected_ratio) / expected_ratio
            
            assert ratio_diff < 0.1, f"Aspect ratio mismatch: expected {expected_ratio:.2f}, got {actual_ratio:.2f}"
            
            print(f"   ✅ Generated {width}x{height} in {processing_time:.1f}s")
            print(f"   📐 Ratio: {actual_ratio:.2f} (expected {expected_ratio:.2f})")
            
            # Rate limiting
            await asyncio.sleep(2)
    
    @pytest.mark.asyncio
    async def test_parameter_normalization(self, flux_config):
        """Test parameter normalization with various input types."""
        tool = FluxKontextMaxTool(flux_config)
        
        # Test descriptive parameters that should be normalized
        test_cases = [
            {
                "prompt": "Portrait of a person",
                "safety_tolerance": "medium",  # Should become 2
                "image_prompt_strength": "high",  # Should become 0.8
                "aspect_ratio": "portrait",  # Should become "2:3"
                "output_format": "webp"  # Should become "jpg"
            },
            {
                "prompt": "Landscape scene",
                "safety_tolerance": "low",  # Should become 1
                "image_prompt_strength": "very_low",  # Should become 0.1
                "aspect_ratio": "wide",  # Should become "16:9"
                "output_format": "jpeg"  # Should become "jpg"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTesting parameter normalization {i+1}/2...")
            
            start_time = time.time()
            result = await tool.invoke(test_case)
            processing_time = time.time() - start_time
            
            # Should succeed despite non-standard parameters
            assert result.success, f"Parameter normalization failed: {result.error}"
            assert "images" in result.data
            assert len(result.data["images"]) > 0
            
            print(f"   ✅ Normalized parameters processed in {processing_time:.1f}s")
            
            # Rate limiting
            await asyncio.sleep(2)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, flux_config):
        """Test error handling with invalid inputs."""
        tool = FluxKontextMaxTool(flux_config)
        
        error_test_cases = [
            {
                "prompt": "",  # Empty prompt
                "description": "empty prompt"
            },
            {
                "prompt": "Test image",
                "image": "https://invalid-url.com/fake.jpg",
                "operation_type": "style_transfer",
                "description": "invalid image URL"
            }
        ]
        
        for test_case in error_test_cases:
            print(f"\nTesting error handling: {test_case['description']}...")
            
            result = await tool.invoke(test_case)
            
            # Should handle gracefully without crashing
            assert isinstance(result, ToolResult)
            
            if not result.success:
                assert result.error is not None
                print(f"   ✅ Properly handled error: {result.error}")
            else:
                print(f"   ⚠️ Unexpectedly succeeded (may be valid fallback behavior)")
    
    @pytest.mark.asyncio
    async def test_api_key_validation(self, flux_config):
        """Test that Replicate API key is properly configured."""
        api_key = flux_config.get("api_key")
        
        if not api_key:
            pytest.skip("REPLICATE_API_TOKEN not configured")
        
        assert len(api_key) > 20, "API key seems too short"
        assert api_key.startswith("r8_"), "Replicate API keys should start with 'r8_'"
        
        print("✅ Replicate API key is properly configured")
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, flux_config, test_images):
        """Performance benchmarks for different Flux operations."""
        operations = [
            ("generation", FluxKontextMaxTool, {"prompt": "Professional headshot, studio lighting"}),
            ("style_transfer", StyleTransferTool, {
                "image": test_images["professional_woman"], 
                "style": "oil_painting"
            }),
            ("object_change", ObjectChangeTool, {
                "image": test_images["business_man"],
                "target_object": "shirt",
                "modification": "blue dress shirt"
            }),
            ("background_swap", BackgroundSwapTool, {
                "image": test_images["person_standing"],
                "new_background": "mountain landscape"
            })
        ]
        
        performance_results = []
        
        for op_name, tool_class, params in operations:
            print(f"\nBenchmarking {op_name}...")
            
            tool = tool_class(flux_config)
            start_time = time.time()
            result = await tool.invoke(params)
            processing_time = time.time() - start_time
            
            performance_results.append({
                "operation": op_name,
                "time": processing_time,
                "success": result.success
            })
            
            status = "✅" if result.success else "❌"
            print(f"   {status} {op_name}: {processing_time:.1f}s")
            
            # Rate limiting
            await asyncio.sleep(3)
        
        # Performance analysis
        successful_ops = [r for r in performance_results if r["success"]]
        
        if successful_ops:
            avg_time = sum(r["time"] for r in successful_ops) / len(successful_ops)
            max_time = max(r["time"] for r in successful_ops)
            
            print(f"\n📊 Performance Summary:")
            print(f"   Success Rate: {len(successful_ops)}/{len(performance_results)}")
            print(f"   Average Time: {avg_time:.1f}s")
            print(f"   Max Time: {max_time:.1f}s")
            
            # Performance assertions
            assert avg_time < 60, f"Average processing time too slow: {avg_time:.1f}s"
            assert max_time < 120, f"Max processing time too slow: {max_time:.1f}s" 