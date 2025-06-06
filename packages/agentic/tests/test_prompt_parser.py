"""
Tests for Prompt Parsing Agent (Phase 2)

Tests GPT-4o prompt parsing with 50+ prompts to validate:
- Intent extraction
- Entity identification (person, object, style, action, setting)  
- Modifier detection (quality, style, lighting, mood, technical)
- Reference recognition (celebrity, artwork, style, brand)
"""

import pytest
import asyncio
import os
from typing import List
from pic_arcade_agentic.agents.prompt_parser import PromptParsingAgent
from pic_arcade_agentic.types import (
    ParsedPrompt,
    PromptEntityType,
    PromptModifierType,
    PromptReferenceType
)


class TestPromptParsingAgent:
    """Test suite for prompt parsing agent with real GPT-4o API calls."""
    
    @pytest.fixture
    def agent(self):
        """Create prompt parsing agent for testing."""
        return PromptParsingAgent()
    
    @pytest.fixture
    def test_prompts(self) -> List[str]:
        """50+ test prompts covering various scenarios."""
        return [
            # Celebrity portraits
            "Create a portrait of Scarlett Johansson in Renaissance style",
            "Generate Emma Stone as a warrior princess",
            "Ryan Gosling in the style of Van Gogh",
            "Jennifer Lawrence in cyberpunk aesthetic", 
            "Tom Hardy as a medieval knight",
            "Margot Robbie in art deco style",
            "Chris Evans in noir lighting",
            "Zendaya as an anime character",
            "Leonardo DiCaprio in Picasso's style",
            "Natalie Portman as a space explorer",
            
            # Object and scene generation
            "A futuristic cityscape with neon lights and flying cars",
            "Photorealistic red sports car in golden hour lighting",
            "Medieval castle on a mountain peak, dramatic clouds",
            "Cozy cottage with fireplace, warm ambient lighting",
            "High-tech laboratory with glowing equipment",
            "Enchanted forest with mystical creatures",
            "Vintage train station in sepia tones",
            "Modern minimalist living room, 4K quality",
            "Steampunk airship in cloudy sky",
            "Underwater palace with bioluminescent details",
            
            # Art style references
            "Landscape painting in Monet's impressionist style",
            "Portrait in the style of Frida Kahlo",
            "Abstract composition like Jackson Pollock",
            "Surreal scene inspired by Salvador Dali",
            "Gothic cathedral in Edward Hopper's style",
            "Still life in Cézanne's post-impressionist manner",
            "Figure drawing in Egon Schiele's style",
            "Cityscape in Roy Lichtenstein's pop art style",
            "Nature scene in Georgia O'Keeffe's style",
            "Portrait in Rembrandt's chiaroscuro technique",
            
            # Technical specifications
            "Ultra HD 8K photorealistic mountain landscape",
            "Low poly 3D character design",
            "High contrast black and white street photography",
            "Shallow depth of field portrait with bokeh",
            "Wide angle architectural photography",
            "Macro photography of water droplets",
            "Long exposure night sky with stars",
            "Studio lighting portrait setup",
            "Film grain vintage photography aesthetic",
            "Hyperrealistic digital painting technique",
            
            # Complex multi-element prompts
            "Emma Watson as Hermione in Hogwarts library, magical lighting, oil painting style",
            "Robert Downey Jr as Iron Man in Marvel comic book art style",
            "Gal Gadot as Wonder Woman in ancient Greek temple, cinematic lighting",
            "Christian Bale as Batman in Gotham City, noir atmosphere",
            "Scarlett Johansson as Black Widow in action pose, digital art",
            "Chris Hemsworth as Thor in Asgard, epic fantasy style",
            "Mark Ruffalo as Hulk in destroyed city, dramatic lighting",
            "Jeremy Renner as Hawkeye on rooftop, sunset lighting",
            "Chadwick Boseman as Black Panther in Wakanda, vibrant colors",
            "Brie Larson as Captain Marvel in space, cosmic background",
            
            # Brand and style references  
            "Logo design in Apple's minimalist aesthetic",
            "Product photography in Nike's dynamic style",
            "Fashion shoot in Vogue magazine style",
            "Car advertisement in BMW's luxury aesthetic",
            "Technology product in Tesla's futuristic style",
            "Interior design in IKEA's Scandinavian style",
            "Food photography in Jamie Oliver's rustic style",
            "Architecture in Frank Lloyd Wright's organic style",
            "Graphic design in Bauhaus movement style",
            "Typography in Helvetica modernist style"
        ]
    
    @pytest.mark.asyncio
    async def test_parse_single_prompt(self, agent):
        """Test parsing a single prompt structure."""
        prompt = "Create a portrait of Scarlett Johansson in Renaissance style"
        
        result = await agent.parse_prompt(prompt)
        
        # Validate basic structure
        assert isinstance(result, ParsedPrompt)
        assert result.intent != ""
        assert result.confidence > 0.0
        assert result.confidence <= 1.0
        
        # Should detect celebrity reference
        celebrity_refs = [r for r in result.references if r.type == PromptReferenceType.CELEBRITY]
        assert len(celebrity_refs) > 0
        assert any("scarlett" in r.text.lower() for r in celebrity_refs)
        
        # Should detect style modifier or reference
        style_items = [m for m in result.modifiers if m.type == PromptModifierType.STYLE]
        style_refs = [r for r in result.references if r.type == PromptReferenceType.STYLE]
        assert len(style_items) > 0 or len(style_refs) > 0
    
    @pytest.mark.asyncio
    async def test_batch_prompt_parsing(self, agent, test_prompts):
        """Test parsing all 50+ prompts to validate comprehensive functionality."""
        results = await agent.parse_batch(test_prompts)
        
        assert len(results) == len(test_prompts)
        
        # Track statistics
        successful_parses = 0
        total_entities = 0
        total_modifiers = 0
        total_references = 0
        intent_types = set()
        
        for i, result in enumerate(results):
            # Basic validation
            assert isinstance(result, ParsedPrompt)
            assert result.confidence >= 0.0
            assert result.confidence <= 1.0
            
            if result.confidence > 0.1:  # Exclude fallback results
                successful_parses += 1
                total_entities += len(result.entities)
                total_modifiers += len(result.modifiers) 
                total_references += len(result.references)
                intent_types.add(result.intent)
                
                # Validate intent is meaningful
                assert result.intent != ""
                assert len(result.intent) > 3
        
        # Require at least 80% success rate
        success_rate = successful_parses / len(test_prompts)
        assert success_rate >= 0.8, f"Success rate {success_rate:.2f} below 80%"
        
        # Validate entity extraction
        avg_entities = total_entities / successful_parses if successful_parses > 0 else 0
        assert avg_entities >= 1.0, f"Average entities {avg_entities:.2f} too low"
        
        # Validate modifier detection
        avg_modifiers = total_modifiers / successful_parses if successful_parses > 0 else 0
        assert avg_modifiers >= 0.5, f"Average modifiers {avg_modifiers:.2f} too low"
        
        # Should detect references in appropriate prompts
        avg_references = total_references / successful_parses if successful_parses > 0 else 0
        assert avg_references >= 0.3, f"Average references {avg_references:.2f} too low"
        
        # Should have diverse intent types
        assert len(intent_types) >= 3, f"Only {len(intent_types)} intent types detected"
        
        print(f"\nPrompt Parsing Test Results:")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Entities: {avg_entities:.1f}")
        print(f"Average Modifiers: {avg_modifiers:.1f}")
        print(f"Average References: {avg_references:.1f}")
        print(f"Intent Types: {len(intent_types)}")
    
    @pytest.mark.asyncio
    async def test_entity_type_detection(self, agent):
        """Test detection of specific entity types."""
        test_cases = [
            ("Portrait of Emma Stone", PromptEntityType.PERSON),
            ("Red Ferrari on highway", PromptEntityType.OBJECT),
            ("Impressionist painting style", PromptEntityType.STYLE),
            ("Dancing in the rain", PromptEntityType.ACTION), 
            ("Medieval castle courtyard", PromptEntityType.SETTING)
        ]
        
        for prompt, expected_type in test_cases:
            result = await agent.parse_prompt(prompt)
            
            # Should detect entity of expected type
            found_type = any(e.type == expected_type for e in result.entities)
            assert found_type, f"Failed to detect {expected_type} in '{prompt}'"
    
    @pytest.mark.asyncio
    async def test_modifier_type_detection(self, agent):
        """Test detection of specific modifier types."""
        test_cases = [
            ("4K ultra HD resolution", PromptModifierType.QUALITY),
            ("Oil painting artistic style", PromptModifierType.STYLE),
            ("Golden hour warm lighting", PromptModifierType.LIGHTING),
            ("Dark mysterious atmosphere", PromptModifierType.MOOD),
            ("Shallow depth of field", PromptModifierType.TECHNICAL)
        ]
        
        for prompt, expected_type in test_cases:
            result = await agent.parse_prompt(prompt)
            
            # Should detect modifier of expected type
            found_type = any(m.type == expected_type for m in result.modifiers)
            assert found_type, f"Failed to detect {expected_type} in '{prompt}'"
    
    @pytest.mark.asyncio
    async def test_reference_type_detection(self, agent):
        """Test detection of specific reference types.""" 
        test_cases = [
            ("Portrait of Leonardo DiCaprio", PromptReferenceType.CELEBRITY),
            ("In the style of Starry Night", PromptReferenceType.ARTWORK),
            ("Minimalist Apple aesthetic", PromptReferenceType.BRAND),
            ("Art Deco style design", PromptReferenceType.STYLE)
        ]
        
        for prompt, expected_type in test_cases:
            result = await agent.parse_prompt(prompt)
            
            # Should detect reference of expected type
            found_type = any(r.type == expected_type for r in result.references)
            assert found_type, f"Failed to detect {expected_type} in '{prompt}'"
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self, agent):
        """Test confidence scoring accuracy."""
        clear_prompt = "Portrait of Emma Stone in Renaissance style"
        vague_prompt = "Make something nice and pretty"
        
        clear_result = await agent.parse_prompt(clear_prompt)
        vague_result = await agent.parse_prompt(vague_prompt)
        
        # Clear prompt should have higher confidence
        assert clear_result.confidence > vague_result.confidence
        assert clear_result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_search_query_optimization(self, agent):
        """Test that references include optimized search queries."""
        prompt = "Portrait of Scarlett Johansson in Van Gogh style"
        
        result = await agent.parse_prompt(prompt)
        
        # Should have references with search queries
        references = result.references
        assert len(references) > 0
        
        for reference in references:
            assert reference.search_query != ""
            assert len(reference.search_query) >= len(reference.text)
            
            # Search query should be more specific
            if reference.type == PromptReferenceType.CELEBRITY:
                assert "portrait" in reference.search_query.lower() or \
                       "photo" in reference.search_query.lower()
    
    @pytest.mark.asyncio 
    async def test_empty_and_invalid_prompts(self, agent):
        """Test handling of edge cases."""
        edge_cases = [
            "",  # Empty prompt
            "a",  # Single character
            "12345",  # Numbers only
            "!!!???",  # Special characters only
            "The the the the the"  # Repetitive words
        ]
        
        for prompt in edge_cases:
            result = await agent.parse_prompt(prompt)
            
            # Should handle gracefully without throwing
            assert isinstance(result, ParsedPrompt)
            assert result.confidence >= 0.0
            
            # Fallback should have low confidence
            if not prompt.strip():
                assert result.confidence < 0.5 