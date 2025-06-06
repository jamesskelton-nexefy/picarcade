"""
Real-World Demo Test - Complete End-to-End Workflow

Demonstrates the full Pic Arcade workflow using:
- Real professional images from Unsplash
- Actual API calls to OpenAI, Replicate, Perplexity
- Complex multi-step image editing scenarios
- Performance and quality validation
"""

import pytest
import asyncio
import os
import time
import json
from typing import List, Dict, Any
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pic_arcade_agentic.agents.tool_agent import ToolFirstAgent
from pic_arcade_agentic.tools import ToolRegistry
from pic_arcade_agentic.tools.image_tools import *
from pic_arcade_agentic.tools.workflow_tools import *
from pic_arcade_agentic.tools.prompt_tools import *
from pic_arcade_agentic.tools.search_tools import *


class TestRealWorldDemo:
    """Real-world demonstration of complete Pic Arcade capabilities."""
    
    @pytest.fixture
    def demo_images(self) -> Dict[str, Dict[str, str]]:
        """Professional demo images categorized by use case."""
        return {
            "marketing_campaign": {
                "corporate_headshot": "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=600&h=800&fit=crop",
                "product_photo": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=800&h=800&fit=crop",
                "brand_billboard": "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=1200&h=800&fit=crop"
            },
            "social_media": {
                "lifestyle_portrait": "https://images.unsplash.com/photo-1494790108755-2616b612b1e9?w=800&h=800&fit=crop",
                "travel_photo": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop",
                "food_styling": "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=800&h=800&fit=crop"
            },
            "e_commerce": {
                "fashion_model": "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=600&h=900&fit=crop",
                "product_showcase": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=1200&h=800&fit=crop",
                "store_interior": "https://images.unsplash.com/photo-1554118811-1e0d58224f24?w=1000&h=800&fit=crop"
            },
            "creative_content": {
                "artistic_portrait": "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=600&h=800&fit=crop",
                "architecture": "https://images.unsplash.com/photo-1513584684374-8bab748fbf90?w=1200&h=900&fit=crop",
                "nature_scene": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=1200&h=800&fit=crop"
            }
        }
    
    @pytest.fixture
    def agent_setup(self):
        """Set up complete agent with all tools."""
        # The ToolFirstAgent creates and manages its own tool registry
        return ToolFirstAgent()
    
    @pytest.mark.asyncio
    async def test_marketing_campaign_workflow(self, agent_setup, demo_images):
        """Test complete marketing campaign image editing workflow."""
        agent = agent_setup
        images = demo_images["marketing_campaign"]
        
        print("\n🎯 MARKETING CAMPAIGN WORKFLOW DEMO")
        print("=" * 60)
        
        campaigns = [
            {
                "name": "Executive Portrait Enhancement",
                "image": images["corporate_headshot"],
                "request": f"""Transform this corporate headshot for executive marketing:
                1. Convert to professional oil painting style
                2. Change the background to a modern office with city view
                3. Enhance the lighting to be more dramatic
                4. Ensure the result maintains executive gravitas
                
                Image: {images["corporate_headshot"]}""",
                "expected_time": 45
            },
            {
                "name": "Product Rebranding",
                "image": images["product_photo"],
                "request": f"""Rebrand this product photo:
                1. Change any visible text to 'PIC ARCADE'
                2. Apply luxury brand aesthetic with golden accents
                3. Create elegant studio lighting
                4. Output in square format for social media
                
                Image: {images["product_photo"]}""",
                "expected_time": 60
            },
            {
                "name": "Billboard Advertisement",
                "image": images["brand_billboard"],
                "request": f"""Create billboard advertisement:
                1. Replace storefront text with 'PIC ARCADE - AI STUDIO'
                2. Add futuristic neon lighting effects
                3. Convert to night scene with dramatic atmosphere
                4. Optimize for wide format display
                
                Image: {images["brand_billboard"]}""",
                "expected_time": 50
            }
        ]
        
        campaign_results = []
        total_start_time = time.time()
        
        for i, campaign in enumerate(campaigns):
            print(f"\n📸 Campaign {i+1}/3: {campaign['name']}")
            print("-" * 40)
            
            start_time = time.time()
            result = await agent.process_request(campaign["request"])
            processing_time = time.time() - start_time
            
            campaign_result = {
                "name": campaign["name"],
                "success": result.success,
                "processing_time": processing_time,
                "expected_time": campaign["expected_time"],
                "performance_score": campaign["expected_time"] / processing_time if result.success else 0
            }
            
            if result.success:
                print(f"   ✅ SUCCESS in {processing_time:.1f}s (expected {campaign['expected_time']}s)")
                
                if "images" in result.data and result.data["images"]:
                    output_url = result.data["images"][0]["url"]
                    print(f"   🖼️  Output: {output_url}")
                    campaign_result["output_url"] = output_url
                
                if "workflow_steps" in result.metadata:
                    steps = result.metadata["workflow_steps"]
                    print(f"   🔧 Workflow: {len(steps)} steps")
                    campaign_result["workflow_steps"] = len(steps)
                
                # Performance analysis
                if processing_time <= campaign["expected_time"]:
                    print(f"   🏆 EXCELLENT performance (under target)")
                elif processing_time <= campaign["expected_time"] * 1.5:
                    print(f"   👍 GOOD performance (within 1.5x target)")
                else:
                    print(f"   ⚠️  SLOW performance (over 1.5x target)")
            else:
                print(f"   ❌ FAILED: {result.error}")
            
            campaign_results.append(campaign_result)
            
            # Rate limiting between campaigns
            await asyncio.sleep(5)
        
        total_time = time.time() - total_start_time
        
        # Campaign Analysis
        successful_campaigns = [c for c in campaign_results if c["success"]]
        success_rate = len(successful_campaigns) / len(campaigns)
        
        if successful_campaigns:
            avg_time = sum(c["processing_time"] for c in successful_campaigns) / len(successful_campaigns)
            avg_performance_score = sum(c["performance_score"] for c in successful_campaigns) / len(successful_campaigns)
        else:
            avg_time = 0
            avg_performance_score = 0
        
        print(f"\n📊 MARKETING CAMPAIGN RESULTS")
        print("=" * 40)
        print(f"Success Rate: {success_rate:.1%} ({len(successful_campaigns)}/{len(campaigns)})")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Average Processing Time: {avg_time:.1f}s")
        print(f"Performance Score: {avg_performance_score:.2f}")
        
        # Quality assertions
        assert success_rate >= 0.67, f"Marketing campaign success rate too low: {success_rate:.1%}"
        assert avg_time < 90, f"Average processing time too slow: {avg_time:.1f}s"
        
        return campaign_results
    
    @pytest.mark.asyncio
    async def test_social_media_content_creation(self, agent_setup, demo_images):
        """Test social media content creation workflow."""
        agent = agent_setup
        images = demo_images["social_media"]
        
        print("\n📱 SOCIAL MEDIA CONTENT CREATION DEMO")
        print("=" * 60)
        
        content_requests = [
            {
                "platform": "Instagram",
                "image": images["lifestyle_portrait"],
                "request": f"""Create Instagram-ready lifestyle content:
                1. Apply vibrant, Instagram-worthy color grading
                2. Add trendy bokeh background effect
                3. Ensure 1:1 square aspect ratio
                4. Enhance skin tones for social media appeal
                
                Image: {images["lifestyle_portrait"]}"""
            },
            {
                "platform": "Pinterest",
                "image": images["travel_photo"],
                "request": f"""Design Pinterest travel pin:
                1. Convert to inspiring vintage travel poster style
                2. Add text overlay space at bottom
                3. Use 2:3 Pinterest-optimized aspect ratio
                4. Enhance colors for maximum engagement
                
                Image: {images["travel_photo"]}"""
            },
            {
                "platform": "TikTok",
                "image": images["food_styling"],
                "request": f"""Create TikTok food content:
                1. Apply dynamic, eye-catching filter
                2. Enhance colors to make food more appetizing
                3. Format for 9:16 vertical video thumbnail
                4. Add subtle motion blur effect for dynamism
                
                Image: {images["food_styling"]}"""
            }
        ]
        
        social_results = []
        
        for i, content in enumerate(content_requests):
            print(f"\n📲 Platform {i+1}/3: {content['platform']}")
            print("-" * 30)
            
            start_time = time.time()
            result = await agent.process_request(content["request"])
            processing_time = time.time() - start_time
            
            platform_result = {
                "platform": content["platform"],
                "success": result.success,
                "processing_time": processing_time
            }
            
            if result.success:
                print(f"   ✅ {content['platform']} content created in {processing_time:.1f}s")
                
                if "images" in result.data and result.data["images"]:
                    image_data = result.data["images"][0]
                    print(f"   📐 Dimensions: {image_data['width']}x{image_data['height']}")
                    print(f"   🔗 URL: {image_data['url']}")
                    platform_result["output_url"] = image_data["url"]
                    platform_result["dimensions"] = f"{image_data['width']}x{image_data['height']}"
            else:
                print(f"   ❌ Failed: {result.error}")
            
            social_results.append(platform_result)
            await asyncio.sleep(3)
        
        # Social Media Analysis
        successful_platforms = [r for r in social_results if r["success"]]
        print(f"\n📈 SOCIAL MEDIA RESULTS")
        print("=" * 30)
        
        for result in social_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['platform']}: {result.get('processing_time', 0):.1f}s")
        
        assert len(successful_platforms) >= 2, "At least 2/3 platforms should succeed"
        
        return social_results
    
    @pytest.mark.asyncio
    async def test_e_commerce_product_enhancement(self, agent_setup, demo_images):
        """Test e-commerce product enhancement workflow."""
        agent = agent_setup
        images = demo_images["e_commerce"]
        
        print("\n🛒 E-COMMERCE PRODUCT ENHANCEMENT DEMO")
        print("=" * 60)
        
        product_enhancements = [
            {
                "category": "Fashion",
                "image": images["fashion_model"],
                "request": f"""Enhance fashion product photography:
                1. Change model's outfit to elegant black dress
                2. Add professional studio lighting
                3. Create clean white background
                4. Ensure colors are accurate for online shopping
                
                Image: {images["fashion_model"]}"""
            },
            {
                "category": "Lifestyle",
                "image": images["product_showcase"],
                "request": f"""Create lifestyle product showcase:
                1. Add warm, inviting home environment background
                2. Enhance product visibility and appeal
                3. Apply soft, natural lighting
                4. Optimize for product catalog display
                
                Image: {images["product_showcase"]}"""
            },
            {
                "category": "Retail",
                "image": images["store_interior"],
                "request": f"""Enhance retail space presentation:
                1. Brighten and modernize the interior
                2. Add 'PIC ARCADE' branding to visible signage
                3. Create welcoming, premium atmosphere
                4. Optimize lighting for customer appeal
                
                Image: {images["store_interior"]}"""
            }
        ]
        
        ecommerce_results = []
        
        for i, enhancement in enumerate(product_enhancements):
            print(f"\n🏪 Category {i+1}/3: {enhancement['category']}")
            print("-" * 30)
            
            start_time = time.time()
            result = await agent.process_request(enhancement["request"])
            processing_time = time.time() - start_time
            
            category_result = {
                "category": enhancement["category"],
                "success": result.success,
                "processing_time": processing_time
            }
            
            if result.success:
                print(f"   ✅ {enhancement['category']} enhanced in {processing_time:.1f}s")
                
                if "images" in result.data and result.data["images"]:
                    output_url = result.data["images"][0]["url"]
                    print(f"   🎯 Enhanced Product: {output_url}")
                    category_result["output_url"] = output_url
            else:
                print(f"   ❌ Enhancement failed: {result.error}")
            
            ecommerce_results.append(category_result)
            await asyncio.sleep(3)
        
        # E-commerce Analysis
        successful_enhancements = [r for r in ecommerce_results if r["success"]]
        success_rate = len(successful_enhancements) / len(product_enhancements)
        
        print(f"\n💼 E-COMMERCE RESULTS")
        print("=" * 25)
        print(f"Enhancement Success Rate: {success_rate:.1%}")
        
        if successful_enhancements:
            avg_time = sum(r["processing_time"] for r in successful_enhancements) / len(successful_enhancements)
            print(f"Average Enhancement Time: {avg_time:.1f}s")
        
        assert success_rate >= 0.67, f"E-commerce enhancement success rate too low: {success_rate:.1%}"
        
        return ecommerce_results
    
    @pytest.mark.asyncio
    async def test_creative_artistic_workflow(self, agent_setup, demo_images):
        """Test creative and artistic transformation workflow."""
        agent = agent_setup
        images = demo_images["creative_content"]
        
        print("\n🎨 CREATIVE ARTISTIC TRANSFORMATION DEMO")
        print("=" * 60)
        
        artistic_transformations = [
            {
                "style": "Fine Art Portrait",
                "image": images["artistic_portrait"],
                "request": f"""Create fine art masterpiece:
                1. Transform to Renaissance painting style
                2. Add dramatic chiaroscuro lighting like Rembrandt
                3. Enhance classical composition and depth
                4. Create museum-quality artistic interpretation
                
                Image: {images["artistic_portrait"]}"""
            },
            {
                "style": "Architectural Visualization",
                "image": images["architecture"],
                "request": f"""Create architectural art piece:
                1. Convert to detailed architectural drawing style
                2. Add technical blueprint elements overlay
                3. Enhance structural details and geometry
                4. Apply monochromatic professional aesthetic
                
                Image: {images["architecture"]}"""
            },
            {
                "style": "Nature Art",
                "image": images["nature_scene"],
                "request": f"""Transform to nature art:
                1. Apply impressionist painting style like Monet
                2. Enhance natural colors and light play
                3. Add artistic brush stroke effects
                4. Create gallery-worthy landscape painting
                
                Image: {images["nature_scene"]}"""
            }
        ]
        
        artistic_results = []
        
        for i, transformation in enumerate(artistic_transformations):
            print(f"\n🖼️  Style {i+1}/3: {transformation['style']}")
            print("-" * 35)
            
            start_time = time.time()
            result = await agent.process_request(transformation["request"])
            processing_time = time.time() - start_time
            
            art_result = {
                "style": transformation["style"],
                "success": result.success,
                "processing_time": processing_time
            }
            
            if result.success:
                print(f"   ✅ {transformation['style']} created in {processing_time:.1f}s")
                
                if "images" in result.data and result.data["images"]:
                    output_url = result.data["images"][0]["url"]
                    print(f"   🎭 Artistic Result: {output_url}")
                    art_result["output_url"] = output_url
            else:
                print(f"   ❌ Transformation failed: {result.error}")
            
            artistic_results.append(art_result)
            await asyncio.sleep(4)  # Longer delay for complex artistic transformations
        
        # Artistic Analysis
        successful_art = [r for r in artistic_results if r["success"]]
        success_rate = len(successful_art) / len(artistic_transformations)
        
        print(f"\n🎪 CREATIVE TRANSFORMATION RESULTS")
        print("=" * 40)
        print(f"Artistic Success Rate: {success_rate:.1%}")
        
        if successful_art:
            avg_time = sum(r["processing_time"] for r in successful_art) / len(successful_art)
            print(f"Average Creation Time: {avg_time:.1f}s")
            
            for result in successful_art:
                print(f"   🎨 {result['style']}: {result['processing_time']:.1f}s")
        
        assert success_rate >= 0.67, f"Creative transformation success rate too low: {success_rate:.1%}"
        
        return artistic_results
    
    @pytest.mark.asyncio
    async def test_complete_demo_suite(self, agent_setup, demo_images):
        """Run complete demo showcasing all capabilities."""
        print("\n🌟 COMPLETE PIC ARCADE DEMO SUITE")
        print("=" * 70)
        print("Testing all workflows with real images and API calls")
        print("=" * 70)
        
        # Run all demo workflows
        demo_start_time = time.time()
        
        marketing_results = await self.test_marketing_campaign_workflow(agent_setup, demo_images)
        social_results = await self.test_social_media_content_creation(agent_setup, demo_images)
        ecommerce_results = await self.test_e_commerce_product_enhancement(agent_setup, demo_images)
        artistic_results = await self.test_creative_artistic_workflow(agent_setup, demo_images)
        
        total_demo_time = time.time() - demo_start_time
        
        # Overall Analysis
        all_results = {
            "marketing": marketing_results,
            "social_media": social_results,
            "ecommerce": ecommerce_results,
            "artistic": artistic_results
        }
        
        total_tests = sum(len(results) for results in all_results.values())
        total_successes = sum(
            len([r for r in results if r.get("success", False)]) 
            for results in all_results.values()
        )
        
        overall_success_rate = total_successes / total_tests
        
        print(f"\n🏆 COMPLETE DEMO SUITE RESULTS")
        print("=" * 50)
        print(f"Overall Success Rate: {overall_success_rate:.1%} ({total_successes}/{total_tests})")
        print(f"Total Demo Time: {total_demo_time:.1f}s")
        print(f"Average per Test: {total_demo_time/total_tests:.1f}s")
        
        # Workflow breakdown
        for workflow_name, results in all_results.items():
            workflow_successes = len([r for r in results if r.get("success", False)])
            workflow_rate = workflow_successes / len(results)
            print(f"   {workflow_name.title()}: {workflow_rate:.1%} ({workflow_successes}/{len(results)})")
        
        # Final assertions
        assert overall_success_rate >= 0.70, f"Overall success rate too low: {overall_success_rate:.1%}"
        assert total_demo_time < 600, f"Demo took too long: {total_demo_time:.1f}s"
        
        print(f"\n✨ DEMO COMPLETE - Pic Arcade is ready for professional use!")
        
        return {
            "overall_success_rate": overall_success_rate,
            "total_time": total_demo_time,
            "workflow_results": all_results
        } 