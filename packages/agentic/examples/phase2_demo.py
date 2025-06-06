#!/usr/bin/env python3
"""
Phase 2 Demonstration Script

Shows the complete workflow of prompt parsing and reference retrieval
using real API integrations with GPT-4o and Bing Search.
"""

import asyncio
import os
import sys
from typing import List

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pic_arcade_agentic import (
    PromptParsingAgent,
    ReferenceRetrievalAgent,
    WorkflowOrchestrator
)


async def demo_prompt_parsing():
    """Demonstrate prompt parsing capabilities."""
    print("🔍 PHASE 2 DEMO: Prompt Parsing Agent")
    print("=" * 50)
    
    agent = PromptParsingAgent()
    
    test_prompts = [
        "Create a portrait of Scarlett Johansson in Renaissance style",
        "Generate Emma Stone as a warrior princess with dramatic lighting",
        "Photorealistic Ferrari on mountain road, golden hour lighting",
        "Abstract art in Picasso's cubist style with vibrant colors"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: {prompt}")
        print("-" * 60)
        
        try:
            result = await agent.parse_prompt(prompt)
            
            print(f"Intent: {result.intent}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Entities ({len(result.entities)}):")
            for entity in result.entities:
                print(f"  - {entity.text} ({entity.type}) [{entity.confidence:.2f}]")
            
            print(f"Modifiers ({len(result.modifiers)}):")
            for modifier in result.modifiers:
                print(f"  - {modifier.text} ({modifier.type}) [{modifier.confidence:.2f}]")
            
            print(f"References ({len(result.references)}):")
            for reference in result.references:
                print(f"  - {reference.text} ({reference.type})")
                print(f"    Search: {reference.search_query}")
                
        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_reference_retrieval():
    """Demonstrate reference retrieval capabilities."""
    print("\n\n🖼️  PHASE 2 DEMO: Reference Retrieval Agent")
    print("=" * 50)
    
    agent = ReferenceRetrievalAgent()
    
    celebrity_tests = [
        "Leonardo DiCaprio",
        "Emma Stone", 
        "Ryan Gosling"
    ]
    
    print(f"\nTesting {len(celebrity_tests)} celebrity references:")
    
    try:
        results = await agent.retrieve_celebrity_references(celebrity_tests)
        
        for prompt, image_urls in results:
            print(f"\n📸 {prompt}:")
            print(f"   Found {len(image_urls)} images")
            for j, url in enumerate(image_urls, 1):
                print(f"   {j}. {url}")
                
    except Exception as e:
        print(f"❌ Error: {e}")


async def demo_full_workflow():
    """Demonstrate complete workflow orchestration."""
    print("\n\n🔄 PHASE 2 DEMO: Full Workflow Orchestration")
    print("=" * 50)
    
    orchestrator = WorkflowOrchestrator()
    
    test_prompts = [
        "Portrait of Margot Robbie in Van Gogh impressionist style",
        "Generate Tom Hardy as a medieval knight in dramatic lighting"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Processing: {prompt}")
        print("-" * 60)
        
        try:
            final_state = await orchestrator.process_prompt(prompt)
            
            print(f"Status: {final_state.status}")
            print(f"Current Step: {final_state.current_step}")
            print(f"Request ID: {final_state.request_id}")
            
            if final_state.context.prompt:
                parsed = final_state.context.prompt
                print(f"\nParsed Results:")
                print(f"  Intent: {parsed.intent}")
                print(f"  Entities: {len(parsed.entities)}")
                print(f"  Modifiers: {len(parsed.modifiers)}")
                print(f"  References: {len(parsed.references)}")
            
            if final_state.context.references:
                total_images = sum(len(ref.image_urls) for ref in final_state.context.references)
                print(f"  Images Found: {total_images}")
                
                for ref in final_state.context.references:
                    if ref.image_urls:
                        print(f"    {ref.text}: {len(ref.image_urls)} images")
                        
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Run the complete Phase 2 demonstration."""
    print("🚀 PIC ARCADE - PHASE 2 DEMONSTRATION")
    print("LangGraph Agentic Pipeline with Real API Integration")
    print("=" * 60)
    
    # Check API keys
    required_keys = ["OPENAI_API_KEY", "BING_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"❌ Missing API keys: {', '.join(missing_keys)}")
        print("Please set the required environment variables.")
        return
    
    print("✅ API keys found")
    
    try:
        # Run demonstrations
        await demo_prompt_parsing()
        await demo_reference_retrieval()
        await demo_full_workflow()
        
        print("\n\n🎉 PHASE 2 DEMONSTRATION COMPLETE!")
        print("All components working with real API integration.")
        print("\nNext: Phase 3 - Image Generation with Flux API")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 