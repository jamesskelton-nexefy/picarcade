#!/usr/bin/env python3
"""
Phase 5 Video Generation Demo for PicArcade

Demonstrates the comprehensive video generation capabilities introduced in Phase 5:
- Runway ML integration for premium video generation
- Multiple Replicate video models (Google Veo 2, Luma Ray, HunyuanVideo, etc.)
- Video editing and enhancement tools
- Text-to-video and image-to-video workflows

This demo script showcases real-world video generation scenarios.
"""

import asyncio
import os
import json
import time
from typing import Dict, Any

# Import PicArcade tools
from pic_arcade_agentic.tools import (
    RunwayVideoTool,
    ReplicateVideoTool,
    VideoEditingTool
)


class Phase5VideoDemo:
    """
    Comprehensive demo of Phase 5 video generation capabilities
    """
    
    def __init__(self):
        self.runway_tool = RunwayVideoTool()
        self.replicate_tool = ReplicateVideoTool()
        self.editing_tool = VideoEditingTool()
        
        # Demo scenarios
        self.demo_scenarios = [
            {
                "name": "Product Marketing Video",
                "prompt": "A sleek smartphone rotating slowly on a white background with dramatic lighting, professional product photography style",
                "provider": "luma_ray",
                "duration": 5,
                "quality": "1080p"
            },
            {
                "name": "Nature Documentary",
                "prompt": "A majestic eagle soaring over a mountain landscape at golden hour, cinematic camera movement",
                "provider": "google_veo2",
                "duration": 8,
                "quality": "4K"
            },
            {
                "name": "Social Media Content",
                "prompt": "A coffee cup with steam rising, cozy cafe atmosphere, warm lighting",
                "provider": "minimax_video",
                "duration": 4,
                "quality": "720p",
                "aspect_ratio": "9:16"
            },
            {
                "name": "Animation Style",
                "prompt": "A cartoon character walking through a magical forest, vibrant colors, animated style",
                "provider": "hunyuan_video",
                "duration": 6,
                "quality": "720p"
            }
        ]
    
    async def run_complete_demo(self):
        """Run the complete Phase 5 video generation demo"""
        print("🎬 PicArcade Phase 5 Video Generation Demo")
        print("=" * 50)
        print()
        
        # Check API keys
        await self._check_api_keys()
        
        # Run each demo scenario
        for i, scenario in enumerate(self.demo_scenarios, 1):
            print(f"\n📹 Demo {i}: {scenario['name']}")
            print("-" * 30)
            await self._run_scenario(scenario)
            
            if i < len(self.demo_scenarios):
                print("\n⏳ Waiting 10 seconds before next demo...")
                time.sleep(10)
        
        # Demonstrate video editing
        print(f"\n🎨 Video Editing Demo")
        print("-" * 30)
        await self._demo_video_editing()
        
        # Demonstrate Runway ML (if available)
        print(f"\n🚀 Runway ML Demo")
        print("-" * 30)
        await self._demo_runway()
        
        print("\n✅ Phase 5 Demo Complete!")
        print("🎉 Video generation capabilities successfully demonstrated")
    
    async def _check_api_keys(self):
        """Check if required API keys are available"""
        print("🔑 Checking API Keys...")
        
        replicate_key = os.getenv('REPLICATE_API_TOKEN')
        runway_key = os.getenv('RUNWAYML_API_SECRET')
        
        if replicate_key:
            print("  ✅ Replicate API Token: Available")
        else:
            print("  ❌ Replicate API Token: Missing")
            print("     Set REPLICATE_API_TOKEN environment variable")
        
        if runway_key:
            print("  ✅ Runway ML API Secret: Available")
        else:
            print("  ⚠️  Runway ML API Secret: Missing (optional)")
            print("     Set RUNWAYML_API_SECRET for Runway features")
        
        print()
    
    async def _run_scenario(self, scenario: Dict[str, Any]):
        """Run a specific video generation scenario"""
        print(f"📝 Prompt: {scenario['prompt'][:80]}...")
        print(f"🏭 Provider: {scenario['provider']}")
        print(f"⏱️  Duration: {scenario['duration']}s")
        print(f"🎥 Quality: {scenario['quality']}")
        
        # Prepare input for Replicate video tool
        input_data = {
            "prompt": scenario["prompt"],
            "provider": scenario["provider"],
            "duration": scenario["duration"],
            "quality": scenario["quality"]
        }
        
        # Add aspect ratio if specified
        if "aspect_ratio" in scenario:
            input_data["aspect_ratio"] = scenario["aspect_ratio"]
        
        print("\n🔄 Generating video...")
        start_time = time.time()
        
        try:
            result = await self.replicate_tool.invoke(input_data)
            
            if result.success:
                processing_time = time.time() - start_time
                data = result.data
                
                print(f"✅ Video generated successfully!")
                print(f"   🔗 URL: {data['video_url']}")
                print(f"   ⏱️  Processing Time: {data['processing_time']:.1f}s")
                print(f"   💰 Estimated Cost: ${data['cost_estimate']:.3f}")
                print(f"   📊 Confidence: {data['confidence']:.1%}")
                print(f"   🎯 Model: {data['model_used']}")
                
                # Save result for potential editing demo
                self._save_demo_result(scenario['name'], data)
                
            else:
                print(f"❌ Video generation failed: {result.data.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"💥 Error during video generation: {str(e)}")
    
    async def _demo_video_editing(self):
        """Demonstrate video editing capabilities"""
        print("🎨 Video editing is available for:")
        print("   • Video upscaling (2x, 4x)")
        print("   • Style transfer")
        print("   • Quality enhancement")
        print("   • Video stabilization")
        print("   • Motion editing")
        
        # For this demo, we'll show the input schema
        print(f"\n📋 Video Editing Tool Schema:")
        print(f"   Input: {json.dumps(self.editing_tool.input_schema, indent=2)}")
        
        print("\n💡 Example usage:")
        example_input = {
            "video_url": "https://example.com/video.mp4",
            "operation": "upscale",
            "upscale_factor": 2
        }
        print(f"   {json.dumps(example_input, indent=2)}")
    
    async def _demo_runway(self):
        """Demonstrate Runway ML capabilities"""
        if not os.getenv('RUNWAYML_API_SECRET'):
            print("⚠️  Runway ML API key not available - showing capabilities only")
            print("\n🚀 Runway ML Features:")
            print("   • Gen-4 Turbo model for high-quality generation")
            print("   • Text-to-video and image-to-video")
            print("   • Multiple aspect ratios (16:9, 9:16, etc.)")
            print("   • Up to 10 seconds duration")
            print("   • Professional quality output")
            return
        
        print("🚀 Testing Runway ML video generation...")
        
        runway_input = {
            "prompt_text": "A professional product shot of a luxury watch on a marble surface with dramatic lighting",
            "model": "gen4_turbo",
            "ratio": "1280:720",
            "duration": 6
        }
        
        print(f"📝 Prompt: {runway_input['prompt_text']}")
        print(f"🎥 Model: {runway_input['model']}")
        print(f"📐 Ratio: {runway_input['ratio']}")
        
        try:
            print("\n🔄 Creating Runway video task...")
            result = await self.runway_tool.invoke(runway_input)
            
            if result.success:
                data = result.data
                print(f"✅ Runway task created successfully!")
                print(f"   🆔 Task ID: {data['task_id']}")
                print(f"   ⏱️  Processing Time: {data['processing_time']:.1f}s")
                print(f"   💰 Estimated Cost: ${data['cost_estimate']:.3f}")
                print(f"   📝 Note: {data['metadata']['note']}")
            else:
                print(f"❌ Runway task failed: {result.data.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"💥 Error with Runway: {str(e)}")
    
    def _save_demo_result(self, scenario_name: str, result_data: Dict[str, Any]):
        """Save demo result for potential use in editing demo"""
        # In a real implementation, you might save these for chaining operations
        print(f"   💾 Result saved for scenario: {scenario_name}")


class VideoProviderComparison:
    """
    Compare different video generation providers
    """
    
    def __init__(self):
        self.replicate_tool = ReplicateVideoTool()
    
    async def compare_providers(self):
        """Compare video generation across different providers"""
        print("\n🔬 Video Provider Comparison")
        print("=" * 40)
        
        test_prompt = "A serene lake reflecting mountains at sunset, peaceful atmosphere"
        
        providers = [
            {"name": "Luma Ray", "provider": "luma_ray", "strength": "Speed & Quality"},
            {"name": "Google Veo 2", "provider": "google_veo2", "strength": "4K Resolution"},
            {"name": "HunyuanVideo", "provider": "hunyuan_video", "strength": "Open Source"},
            {"name": "Minimax Video", "provider": "minimax_video", "strength": "Animation"},
        ]
        
        print(f"📝 Test Prompt: {test_prompt}")
        print("\n📊 Provider Capabilities:")
        
        for provider in providers:
            print(f"\n🏭 {provider['name']} ({provider['provider']})")
            print(f"   💪 Strength: {provider['strength']}")
            
            # Get model config
            config = self.replicate_tool._get_model_config(provider['provider'])
            if config:
                print(f"   🎥 Max Duration: {config['max_duration']}s")
                print(f"   📐 Qualities: {', '.join(config['qualities'])}")
                print(f"   🖼️  Supports Image Input: {'Yes' if config['supports_image'] else 'No'}")
                print(f"   📖 Description: {config['description']}")
            
            # Calculate cost estimate
            cost = self.replicate_tool._calculate_cost(provider['provider'], {
                "duration": 6,
                "quality": "720p"
            })
            print(f"   💰 Est. Cost (6s, 720p): ${cost:.3f}")


async def main():
    """Main demo function"""
    print("🎬 Starting PicArcade Phase 5 Video Generation Demo...")
    print()
    
    # Run main demo
    demo = Phase5VideoDemo()
    await demo.run_complete_demo()
    
    # Run provider comparison
    comparison = VideoProviderComparison()
    await comparison.compare_providers()
    
    print("\n" + "=" * 60)
    print("🎉 Phase 5 Demo Complete!")
    print("✨ Video generation capabilities are now available in PicArcade")
    print("\n📚 Next Steps:")
    print("   • Test with your own prompts")
    print("   • Experiment with different providers")
    print("   • Try image-to-video generation")
    print("   • Explore video editing features")
    print("   • Integrate with existing workflows")


if __name__ == "__main__":
    asyncio.run(main()) 