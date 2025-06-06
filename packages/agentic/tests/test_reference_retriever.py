"""
Tests for Reference Retrieval Agent (Phase 2)

Tests Bing Search API integration with 10 celebrity prompts to validate:
- Image search functionality
- Top 3 image URLs include correct results
- CLIP-based ranking (basic implementation)
- Search query optimization
"""

import pytest
import asyncio
import os
from typing import List, Tuple
from pic_arcade_agentic.agents.reference_retriever import ReferenceRetrievalAgent
from pic_arcade_agentic.types import (
    PromptReference,
    PromptReferenceType,
    SearchConfig
)


class TestReferenceRetrievalAgent:
    """Test suite for reference retrieval agent with real Bing API calls."""
    
    @pytest.fixture
    def agent(self):
        """Create reference retrieval agent for testing."""
        return ReferenceRetrievalAgent()
    
    @pytest.fixture
    def celebrity_prompts(self) -> List[str]:
        """10 celebrity prompts for testing reference retrieval."""
        return [
            "Scarlett Johansson",
            "Leonardo DiCaprio", 
            "Emma Stone",
            "Ryan Gosling",
            "Jennifer Lawrence",
            "Tom Hardy",
            "Margot Robbie",
            "Chris Evans",
            "Zendaya",
            "Natalie Portman"
        ]
    
    @pytest.mark.asyncio
    async def test_single_celebrity_reference(self, agent):
        """Test retrieving images for a single celebrity reference."""
        reference = PromptReference(
            text="Scarlett Johansson",
            type=PromptReferenceType.CELEBRITY,
            search_query="Scarlett Johansson portrait headshot photo",
            confidence=0.9
        )
        
        updated_references = await agent.retrieve_references([reference])
        
        assert len(updated_references) == 1
        updated_ref = updated_references[0]
        
        # Should have image URLs
        assert len(updated_ref.image_urls) > 0
        assert len(updated_ref.image_urls) <= 3  # Top 3 results
        
        # URLs should be valid
        for url in updated_ref.image_urls:
            assert url.startswith("http")
            assert any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp'])
    
    @pytest.mark.asyncio
    async def test_celebrity_batch_retrieval(self, agent, celebrity_prompts):
        """Test retrieving reference images for 10 celebrity prompts."""
        results = await agent.retrieve_celebrity_references(celebrity_prompts)
        
        assert len(results) == len(celebrity_prompts)
        
        successful_retrievals = 0
        total_images = 0
        
        for prompt, image_urls in results:
            assert prompt in celebrity_prompts
            
            if len(image_urls) > 0:
                successful_retrievals += 1
                total_images += len(image_urls)
                
                # Should have at most 3 images
                assert len(image_urls) <= 3
                
                # URLs should be valid
                for url in image_urls:
                    assert url.startswith("http")
                    assert isinstance(url, str)
                    assert len(url) > 10  # Reasonable URL length
        
        # Require at least 80% success rate
        success_rate = successful_retrievals / len(celebrity_prompts)
        assert success_rate >= 0.8, f"Success rate {success_rate:.2f} below 80%"
        
        # Should find images for most celebrities
        avg_images = total_images / successful_retrievals if successful_retrievals > 0 else 0
        assert avg_images >= 1.0, f"Average images per success {avg_images:.2f} too low"
        
        print(f"\nReference Retrieval Test Results:")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Total Images Found: {total_images}")
        print(f"Average Images per Success: {avg_images:.1f}")
    
    @pytest.mark.asyncio
    async def test_search_query_optimization(self, agent):
        """Test search query optimization for different reference types."""
        test_cases = [
            (PromptReferenceType.CELEBRITY, "Emma Stone", "portrait headshot photo"),
            (PromptReferenceType.ARTWORK, "Starry Night", "artwork painting art museum"),
            (PromptReferenceType.STYLE, "Art Deco", "style aesthetic visual design"),
            (PromptReferenceType.BRAND, "Apple", "logo brand design official")
        ]
        
        for ref_type, query, expected_terms in test_cases:
            optimized = agent._optimize_search_query(query, ref_type)
            
            # Should contain original query
            assert query.lower() in optimized.lower()
            
            # Should add relevant terms
            for term in expected_terms.split():
                assert term in optimized.lower()
            
            # Should be longer than original
            assert len(optimized) > len(query)
    
    @pytest.mark.asyncio
    async def test_image_ranking(self, agent):
        """Test image ranking algorithm."""
        # Mock image data
        test_images = [
            {
                "url": "https://example.com/small.jpg",
                "width": 200,
                "height": 200,
                "encoding_format": "JPEG",
                "name": "small image",
                "host_page_url": "https://example.com"
            },
            {
                "url": "https://example.com/large.jpg", 
                "width": 1200,
                "height": 1200,
                "encoding_format": "JPEG",
                "name": "high quality portrait",
                "host_page_url": "https://wikipedia.org/page"
            },
            {
                "url": "https://example.com/medium.png",
                "width": 600,
                "height": 600,
                "encoding_format": "PNG",
                "name": "medium size",
                "host_page_url": "https://imdb.com/name"
            }
        ]
        
        reference = PromptReference(
            text="test person",
            type=PromptReferenceType.CELEBRITY,
            search_query="test person portrait",
            confidence=0.9
        )
        
        ranked_urls = await agent._rank_images(test_images, reference)
        
        # Should return URLs
        assert len(ranked_urls) > 0
        assert len(ranked_urls) <= 3
        
        # All should be valid URLs
        for url in ranked_urls:
            assert url.startswith("https://example.com/")
            assert url.endswith((".jpg", ".png"))
    
    @pytest.mark.asyncio
    async def test_multiple_reference_types(self, agent):
        """Test retrieving different types of references."""
        references = [
            PromptReference(
                text="Leonardo DiCaprio",
                type=PromptReferenceType.CELEBRITY,
                search_query="Leonardo DiCaprio actor portrait",
                confidence=0.9
            ),
            PromptReference(
                text="Mona Lisa",
                type=PromptReferenceType.ARTWORK,
                search_query="Mona Lisa painting Leonardo da Vinci",
                confidence=0.8
            ),
            PromptReference(
                text="minimalist design",
                type=PromptReferenceType.STYLE,
                search_query="minimalist design style aesthetic",
                confidence=0.7
            )
        ]
        
        updated_references = await agent.retrieve_references(references)
        
        assert len(updated_references) == 3
        
        # Each should maintain its type and properties
        for i, updated_ref in enumerate(updated_references):
            original_ref = references[i]
            assert updated_ref.type == original_ref.type
            assert updated_ref.text == original_ref.text
            assert updated_ref.search_query == original_ref.search_query
            
            # Should have attempted to find images (may be empty for some types)
            assert isinstance(updated_ref.image_urls, list)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling for invalid references."""
        # Test with invalid/empty references
        edge_cases = [
            PromptReference(
                text="",
                type=PromptReferenceType.CELEBRITY,
                search_query="",
                confidence=0.0
            ),
            PromptReference(
                text="nonexistent person xyz 123",
                type=PromptReferenceType.CELEBRITY,
                search_query="nonexistent person xyz 123",
                confidence=0.1
            )
        ]
        
        # Should handle gracefully without throwing
        updated_references = await agent.retrieve_references(edge_cases)
        
        assert len(updated_references) == 2
        
        for ref in updated_references:
            # Should return reference even if no images found
            assert isinstance(ref, PromptReference)
            assert isinstance(ref.image_urls, list)
    
    @pytest.mark.asyncio
    async def test_image_url_validation(self, agent):
        """Test that returned image URLs are valid and accessible."""
        reference = PromptReference(
            text="Tom Hanks",
            type=PromptReferenceType.CELEBRITY,
            search_query="Tom Hanks actor portrait photo",
            confidence=0.9
        )
        
        updated_references = await agent.retrieve_references([reference])
        
        if updated_references and updated_references[0].image_urls:
            image_urls = updated_references[0].image_urls
            
            for url in image_urls:
                # URL format validation
                assert url.startswith(("http://", "https://"))
                assert "." in url  # Should have domain
                
                # Should look like an image URL
                url_lower = url.lower()
                has_image_extension = any(
                    ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']
                )
                has_image_param = any(
                    param in url_lower for param in ['image', 'photo', 'pic', 'img']
                )
                
                # Should indicate it's an image either by extension or parameters
                assert has_image_extension or has_image_param, f"URL doesn't look like image: {url}"
    
    @pytest.mark.asyncio
    async def test_search_api_configuration(self, agent):
        """Test that search API is properly configured."""
        # Verify API key is set
        assert agent.config.api_key is not None
        assert len(agent.config.api_key) > 10  # Reasonable API key length
        
        # Verify headers are set
        assert "Ocp-Apim-Subscription-Key" in agent.headers
        assert agent.headers["Ocp-Apim-Subscription-Key"] == agent.config.api_key
        
        # Verify configuration
        assert agent.config.provider == "bing"
        assert agent.config.base_url == "https://api.bing.microsoft.com/v7.0/images/search"
        assert agent.config.max_results <= 50  # Bing API limit 