#!/usr/bin/env python3
"""
Direct Tool Test - Verify Model Metadata

Test individual tools directly to see the model metadata being returned.
"""

import asyncio
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pic_arcade_agentic.tools.image_tools import StyleTransferTool, ObjectChangeTool


async def test_style_transfer_direct():
    """Test StyleTransferTool directly to see metadata output."""
    
    print("🔧 DIRECT TOOL TEST - StyleTransferTool")
    print("=" * 50)
    
    # Initialize tool
    config = {
        "api_key": os.getenv("REPLICATE_API_TOKEN")
    }
    
    tool = StyleTransferTool(config)
    
    # Test input (using a real test image)
    test_input = {
        "image": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=800&fit=crop",
        "style": "watercolor",
        "prompt": "convert to watercolor style",
        "strength": 0.7
    }
    
    print(f"📝 Input: {test_input}")
    print(f"🚀 Calling tool...")
    
    try:
        result = await tool.invoke(test_input)
        
        print(f"\n📊 Tool Result:")
        print(f"   Success: {result.success}")
        
        if result.success and result.data:
            print(f"   Data Keys: {list(result.data.keys())}")
            
            if "styled_image" in result.data:
                print(f"   Styled Image URL: {result.data['styled_image'][:50]}...")
            
            if "style_applied" in result.data:
                print(f"   Style Applied: {result.data['style_applied']}")
                
            if "processing_time" in result.data:
                print(f"   Processing Time: {result.data['processing_time']:.1f}s")
        
        if result.metadata:
            print(f"\n🤖 Metadata:")
            for key, value in result.metadata.items():
                if key == "model_used":
                    print(f"   🎯 Model Used: {value}")
                elif key == "enhanced_prompt":
                    print(f"   📝 Enhanced Prompt: {value[:100]}...")
                else:
                    print(f"   {key}: {value}")
        
        if result.error:
            print(f"\n❌ Error: {result.error}")
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")


async def test_object_change_direct():
    """Test ObjectChangeTool directly to see metadata output."""
    
    print("\n🔧 DIRECT TOOL TEST - ObjectChangeTool")
    print("=" * 50)
    
    # Initialize tool
    config = {
        "api_key": os.getenv("REPLICATE_API_TOKEN")
    }
    
    tool = ObjectChangeTool(config)
    
    # Test input (using a real test image)
    test_input = {
        "image": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=800&fit=crop",
        "target_object": "hair",
        "modification": "blonde curly hair",
        "strength": 0.8
    }
    
    print(f"📝 Input: {test_input}")
    print(f"🚀 Calling tool...")
    
    try:
        result = await tool.invoke(test_input)
        
        print(f"\n📊 Tool Result:")
        print(f"   Success: {result.success}")
        
        if result.success and result.data:
            print(f"   Data Keys: {list(result.data.keys())}")
            
            if "modified_image" in result.data:
                print(f"   Modified Image URL: {result.data['modified_image'][:50]}...")
            
            if "object_modified" in result.data:
                print(f"   Object Modified: {result.data['object_modified']}")
                
            if "processing_time" in result.data:
                print(f"   Processing Time: {result.data['processing_time']:.1f}s")
        
        if result.metadata:
            print(f"\n🤖 Metadata:")
            for key, value in result.metadata.items():
                if key == "model_used":
                    print(f"   🎯 Model Used: {value}")
                elif key == "prompt":
                    print(f"   📝 Generated Prompt: {value[:100]}...")
                else:
                    print(f"   {key}: {value}")
        
        if result.error:
            print(f"\n❌ Error: {result.error}")
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")


async def main():
    """Run direct tool test."""
    
    # Check API key
    api_key = os.getenv("REPLICATE_API_TOKEN")
    if not api_key:
        print("❌ Missing REPLICATE_API_TOKEN")
        return
    
    print(f"✅ API Key configured: {api_key[:10]}...")
    
    await test_style_transfer_direct()
    await test_object_change_direct()


if __name__ == "__main__":
    asyncio.run(main()) 