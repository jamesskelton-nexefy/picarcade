"""
Search Tools for Pic Arcade

Tools that handle image and web search using Perplexity API.
"""

import os
import logging
from typing import Dict, Any, List
import httpx

from .base import Tool, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


class PerplexitySearchTool(Tool):
    """
    Tool for searching images and information using Perplexity API.
    
    Finds reference images for celebrities, artworks, styles, and brands
    with intelligent query optimization and comprehensive results.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="perplexity_search",
            description="Search for images and information using Perplexity API with intelligent analysis",
            category=ToolCategory.IMAGE_SEARCH,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for images and information"
                    },
                    "search_type": {
                        "type": "string", 
                        "enum": ["images", "web", "comprehensive"],
                        "default": "comprehensive"
                    },
                    "reference_type": {
                        "type": "string", 
                        "enum": ["celebrity", "artwork", "style", "brand", "general"],
                        "default": "general"
                    },
                    "focus": {
                        "type": "string",
                        "description": "Specific focus for the search (e.g., 'high quality photos', 'recent images')",
                        "default": "high quality reference images"
                    }
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "images": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "source": {"type": "string"},
                                "rank_score": {"type": "number"}
                            }
                        }
                    },
                    "web_results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "snippet": {"type": "string"}
                            }
                        }
                    },
                    "summary": {"type": "string"},
                    "total_results": {"type": "integer"}
                }
            },
            config=config
        )
    
    def _validate_config(self) -> None:
        """Validate Perplexity API configuration."""
        api_key = self.config.get("api_key") or os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("Perplexity API key is required for PerplexitySearchTool")
        self.config["api_key"] = api_key
        
        # Initialize headers after validation
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def _optimize_search_query(self, query: str, reference_type: str, focus: str) -> str:
        """Optimize search query based on reference type and focus."""
        query = query.strip()
        
        # Base optimizations by reference type
        type_optimizations = {
            "celebrity": f"Find high-quality photos and images of {query}",
            "artwork": f"Find {query} artwork, paintings, and artistic references",
            "style": f"Find examples and references of {query} style in visual art",
            "brand": f"Find official {query} brand imagery and visual identity",
            "general": f"Find high-quality images and visual references for {query}"
        }
        
        base_query = type_optimizations.get(reference_type, f"Find images of {query}")
        
        # Add focus specification
        if focus and focus != "high quality reference images":
            optimized_query = f"{base_query}. Focus on: {focus}"
        else:
            optimized_query = f"{base_query}. Focus on high-quality, clear reference images suitable for AI generation."
        
        return optimized_query
    
    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Search using Perplexity API.
        
        Args:
            input_data: Search parameters
            
        Returns:
            ToolResult with search results including images and context
        """
        try:
            query = input_data["query"]
            search_type = input_data.get("search_type", "comprehensive")
            reference_type = input_data.get("reference_type", "general")
            focus = input_data.get("focus", "high quality reference images")
            
            # Optimize query for Perplexity
            optimized_query = self._optimize_search_query(query, reference_type, focus)
            
            # Prepare request payload
            payload = {
                "model": "llama-3.1-sonar-large-128k-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a search assistant that finds high-quality images and comprehensive information. When searching for images, provide direct URLs when possible and describe the images found. Always provide accurate, recent information."
                    },
                    {
                        "role": "user",
                        "content": optimized_query
                    }
                ],
                "max_tokens": 4000,
                "temperature": 0.2,
                "top_p": 0.9,
                "search_domain_filter": ["perplexity.ai"],
                "return_images": True,
                "return_related_questions": False,
                "search_recency_filter": "month",
                "top_k": 0,
                "stream": False,
                "presence_penalty": 0,
                "frequency_penalty": 1
            }
            
            # Make API request
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    json=payload,
                    headers=self.headers
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Extract response content
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                citations = data.get("citations", [])
                
                # Parse images and web results from response
                images = self._extract_images_from_response(content, citations)
                web_results = self._extract_web_results_from_citations(citations)
                
                return ToolResult(
                    success=True,
                    data={
                        "images": images,
                        "web_results": web_results,
                        "summary": content,
                        "total_results": len(images) + len(web_results)
                    },
                    metadata={
                        "original_query": query,
                        "optimized_query": optimized_query,
                        "reference_type": reference_type,
                        "search_type": search_type,
                        "model_used": "llama-3.1-sonar-large-128k-online"
                    }
                )
                
        except httpx.HTTPError as e:
            return ToolResult(
                success=False,
                error=f"HTTP error during Perplexity search: {e}"
            )
        except Exception as e:
            logger.error(f"Perplexity search failed: {e}")
            return ToolResult(
                success=False,
                error=f"Search failed: {str(e)}"
            )
    
    def _extract_images_from_response(self, content: str, citations: List[str]) -> List[Dict[str, Any]]:
        """Extract image information from Perplexity response."""
        images = []
        
        # Look for image URLs in citations
        for i, citation in enumerate(citations):
            if any(ext in citation.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                images.append({
                    "url": citation,
                    "title": f"Reference Image {i+1}",
                    "description": "Image found via Perplexity search",
                    "source": citation,
                    "rank_score": 1.0 - (i * 0.1)  # Decreasing score by position
                })
        
        # If no direct image URLs, extract from content analysis
        if not images and "image" in content.lower():
            # Create placeholder entries based on content mentions
            image_mentions = content.lower().count("image")
            for i in range(min(image_mentions, 3)):
                images.append({
                    "url": "",  # Will need to be populated by follow-up searches
                    "title": f"Referenced Image {i+1}",
                    "description": f"Image reference found in search results",
                    "source": "perplexity_analysis",
                    "rank_score": 0.8 - (i * 0.1)
                })
        
        return images
    
    def _extract_web_results_from_citations(self, citations: List[str]) -> List[Dict[str, Any]]:
        """Extract web results from citations."""
        web_results = []
        
        for citation in citations:
            # Skip image URLs
            if not any(ext in citation.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                web_results.append({
                    "title": citation.split('/')[-1] if '/' in citation else citation,
                    "url": citation,
                    "snippet": "Referenced source found via Perplexity search"
                })
        
        return web_results


# Keep this as an alias for backward compatibility
class WebSearchTool(PerplexitySearchTool):
    """Alias for PerplexitySearchTool to maintain compatibility."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "web_search"
        self.description = "Search the web using Perplexity API for comprehensive information"


# Legacy compatibility - these will redirect to Perplexity
class BingImageSearchTool(PerplexitySearchTool):
    """Legacy compatibility tool that redirects to Perplexity."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "bing_image_search"
        self.description = "Image search using Perplexity API (replaces Bing)"


class GoogleImageSearchTool(PerplexitySearchTool):
    """Legacy compatibility tool that redirects to Perplexity."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "google_image_search"
        self.description = "Image search using Perplexity API (replaces Google)" 