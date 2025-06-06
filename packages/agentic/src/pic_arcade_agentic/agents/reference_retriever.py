"""
Reference Retrieval Agent for Phase 2

Uses Bing Search API to find reference images for celebrities, artworks, styles, and brands.
Includes CLIP-based ranking to select the most relevant images.
"""

import logging
import os
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import httpx
import json
from urllib.parse import quote_plus

from ..types import (
    PromptReference,
    PromptReferenceType,
    SearchConfig
)
from ..utils.decision_logger import decision_logger, DecisionType

logger = logging.getLogger(__name__)


class ReferenceRetrievalAgent:
    """
    Agent responsible for retrieving reference images using search APIs.
    
    Searches for images based on extracted references and ranks them
    for relevance using CLIP or similar scoring methods.
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize the reference retrieval agent."""
        self.config = config or SearchConfig(
            base_url="https://api.bing.microsoft.com/v7.0/images/search",
            api_key=os.getenv("BING_API_KEY"),
            provider="bing",
            max_results=10,
            timeout=30
        )
        
        if not self.config.api_key:
            raise ValueError("Bing API key is required")
            
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.config.api_key,
            "User-Agent": "PicArcade/1.0"
        }
    
    async def retrieve_references(
        self, 
        references: List[PromptReference]
    ) -> List[PromptReference]:
        """
        Retrieve reference images for all provided references.
        
        Args:
            references: List of PromptReference objects from prompt parsing
            
        Returns:
            Updated list of PromptReference objects with image URLs
        """
        # Generate request ID for decision tracking
        request_id = f"retrieve_{int(time.time() * 1000)}"
        
        # Start decision tracking
        decision_logger.start_decision(
            request_id=request_id,
            agent_name="ReferenceRetrievalAgent",
            initial_context={
                "references_count": len(references),
                "reference_types": [ref.type.value for ref in references],
                "search_provider": self.config.provider,
                "max_results_per_reference": self.config.max_results
            }
        )
        
        start_time = time.time()
        
        # Log retrieval strategy decision
        decision_logger.log_decision_step(
            request_id=request_id,
            decision_type=DecisionType.REFERENCE_RETRIEVAL,
            input_data={
                "references_to_process": [{"text": ref.text, "type": ref.type.value, "query": ref.search_query} for ref in references],
                "total_references": len(references)
            },
            decision_reasoning=f"Processing {len(references)} references using {self.config.provider} search API with parallel retrieval strategy",
            output_data={
                "processing_strategy": "sequential_per_reference",
                "search_provider": self.config.provider,
                "max_results_per_search": self.config.max_results
            },
            confidence_score=0.9,
            metadata={
                "api_provider": self.config.provider,
                "parallel_processing": False  # Currently sequential
            }
        )
        
        updated_references = []
        successful_retrievals = 0
        failed_retrievals = 0
        
        # Process each reference
        for i, reference in enumerate(references):
            ref_start_time = time.time()
            
            try:
                # Log individual reference processing decision
                decision_logger.log_decision_step(
                    request_id=request_id,
                    decision_type=DecisionType.REFERENCE_RETRIEVAL,
                    input_data={
                        "reference_index": i,
                        "reference_text": reference.text,
                        "reference_type": reference.type.value,
                        "search_query": reference.search_query
                    },
                    decision_reasoning=f"Starting image search for {reference.type.value} reference: '{reference.text}' using optimized query",
                    output_data={"search_initiated": True},
                    confidence_score=reference.confidence,
                    metadata={
                        "reference_processing_step": "search_start",
                        "original_confidence": reference.confidence
                    }
                )
                
                logger.info(f"Retrieving images for reference: {reference.text}")
                
                # Search for images
                image_urls = await self._search_images(reference.search_query, reference.type, request_id, i)
                
                # Rank and select top images
                top_images = await self._rank_images(image_urls, reference, request_id, i)
                
                ref_execution_time = (time.time() - ref_start_time) * 1000
                
                # Update reference with found images
                updated_reference = reference.model_copy()
                updated_reference.image_urls = top_images
                updated_references.append(updated_reference)
                
                if top_images:
                    successful_retrievals += 1
                    
                    # Log successful retrieval
                    decision_logger.log_decision_step(
                        request_id=request_id,
                        decision_type=DecisionType.REFERENCE_RETRIEVAL,
                        input_data={"reference_processed": reference.text},
                        decision_reasoning=f"Successfully retrieved {len(top_images)} images for '{reference.text}' with quality ranking applied",
                        output_data={
                            "images_found": len(top_images),
                            "image_urls": top_images,
                            "retrieval_success": True
                        },
                        confidence_score=reference.confidence,
                        execution_time_ms=ref_execution_time,
                        metadata={
                            "reference_type": reference.type.value,
                            "search_success": True
                        }
                    )
                else:
                    failed_retrievals += 1
                    
                    # Log failed retrieval
                    decision_logger.log_decision_step(
                        request_id=request_id,
                        decision_type=DecisionType.REFERENCE_RETRIEVAL,
                        input_data={"reference_processed": reference.text},
                        decision_reasoning=f"No suitable images found for '{reference.text}' - search returned no results or all images failed quality filtering",
                        output_data={
                            "images_found": 0,
                            "retrieval_success": False
                        },
                        confidence_score=0.0,
                        execution_time_ms=ref_execution_time,
                        metadata={
                            "reference_type": reference.type.value,
                            "search_failure_reason": "no_quality_results"
                        }
                    )
                
                logger.info(f"Found {len(top_images)} images for '{reference.text}'")
                
            except Exception as e:
                ref_execution_time = (time.time() - ref_start_time) * 1000
                failed_retrievals += 1
                error_msg = str(e)
                
                # Log retrieval error
                decision_logger.log_decision_step(
                    request_id=request_id,
                    decision_type=DecisionType.ERROR_HANDLING,
                    input_data={"reference_error": reference.text, "error": error_msg},
                    decision_reasoning=f"Reference retrieval failed for '{reference.text}' due to error, adding reference without images to maintain workflow continuity",
                    output_data={
                        "error_type": type(e).__name__,
                        "recovery_action": "add_empty_reference"
                    },
                    confidence_score=0.0,
                    execution_time_ms=ref_execution_time,
                    error=error_msg,
                    metadata={
                        "reference_type": reference.type.value,
                        "error_handling": "graceful_degradation"
                    }
                )
                
                logger.error(f"Failed to retrieve images for '{reference.text}': {e}")
                # Add reference without images
                updated_references.append(reference)
        
        total_execution_time = (time.time() - start_time) * 1000
        success_rate = successful_retrievals / len(references) if references else 0
        
        # Log overall retrieval results
        decision_logger.log_decision_step(
            request_id=request_id,
            decision_type=DecisionType.REFERENCE_RETRIEVAL,
            input_data={"batch_processing_complete": True},
            decision_reasoning=f"Completed reference retrieval batch with {successful_retrievals}/{len(references)} successful retrievals ({success_rate:.1%} success rate)",
            output_data={
                "total_references_processed": len(references),
                "successful_retrievals": successful_retrievals,
                "failed_retrievals": failed_retrievals,
                "success_rate": success_rate,
                "total_images_retrieved": sum(len(ref.image_urls) if hasattr(ref, 'image_urls') and ref.image_urls else 0 for ref in updated_references)
            },
            confidence_score=success_rate,
            execution_time_ms=total_execution_time,
            metadata={
                "batch_processing_complete": True,
                "ready_for_generation": True
            }
        )
        
        # Complete decision tracking
        decision_logger.complete_decision(
            request_id=request_id,
            final_result={
                "references_processed": len(references),
                "successful_retrievals": successful_retrievals,
                "success_rate": success_rate,
                "execution_time_ms": total_execution_time
            },
            success=successful_retrievals > 0
        )
        
        return updated_references
    
    async def _search_images(
        self, 
        query: str, 
        reference_type: PromptReferenceType,
        request_id: str,
        reference_index: int
    ) -> List[Dict[str, Any]]:
        """
        Search for images using Bing Image Search API.
        
        Args:
            query: Search query string
            reference_type: Type of reference to optimize search
            request_id: Request ID for decision tracking
            reference_index: Index of reference being processed
            
        Returns:
            List of image metadata from search results
        """
        search_start_time = time.time()
        
        try:
            # Optimize query based on reference type
            optimized_query = self._optimize_search_query(query, reference_type)
            
            # Log query optimization decision
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.REFERENCE_RETRIEVAL,
                input_data={
                    "original_query": query,
                    "reference_type": reference_type.value,
                    "reference_index": reference_index
                },
                decision_reasoning=f"Optimized search query for {reference_type.value} type by adding relevant terms to improve search precision",
                output_data={
                    "optimized_query": optimized_query,
                    "optimization_applied": optimized_query != query
                },
                confidence_score=0.8,
                metadata={
                    "query_optimization": True,
                    "search_strategy": f"{reference_type.value}_optimized"
                }
            )
            
            # Prepare search parameters
            params = {
                "q": optimized_query,
                "count": min(self.config.max_results, 50),  # Bing max is 50
                "imageType": "Photo",
                "size": "Medium",
                "color": "ColorOnly",
                "freshness": "Month",
                "safeSearch": "Strict"
            }
            
            # Add type-specific filters
            if reference_type == PromptReferenceType.CELEBRITY:
                params["imageType"] = "Photo"
                params["aspect"] = "Square"
            elif reference_type == PromptReferenceType.ARTWORK:
                params["imageType"] = "Photo"
                params["size"] = "Large"
            
            # Log search parameters decision
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.REFERENCE_RETRIEVAL,
                input_data={"search_params": params},
                decision_reasoning=f"Configured Bing search parameters with {reference_type.value}-specific filters for optimal image quality and relevance",
                output_data={
                    "api_endpoint": self.config.base_url,
                    "search_filters_applied": True,
                    "expected_max_results": params["count"]
                },
                confidence_score=0.9,
                metadata={
                    "api_configuration": "type_specific_filters",
                    "search_provider": "bing"
                }
            )
            
            logger.debug(f"Bing search params: {params}")
            
            # Make API request
            api_start_time = time.time()
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    self.config.base_url,
                    params=params,
                    headers=self.headers
                )
                response.raise_for_status()
                
                data = response.json()
                
                api_time = (time.time() - api_start_time) * 1000
                
                # Log API response
                decision_logger.log_decision_step(
                    request_id=request_id,
                    decision_type=DecisionType.REFERENCE_RETRIEVAL,
                    input_data={"api_call_complete": True},
                    decision_reasoning=f"Received search results from Bing API with {len(data.get('value', []))} raw images for processing",
                    output_data={
                        "raw_results_count": len(data.get('value', [])),
                        "api_response_time_ms": api_time,
                        "api_status": "success"
                    },
                    confidence_score=0.9,
                    execution_time_ms=api_time,
                    metadata={
                        "api_success": True,
                        "search_provider_response": "valid"
                    }
                )
                
                # Extract image information
                images = []
                valid_images = 0
                invalid_images = 0
                
                for item in data.get("value", []):
                    image_info = {
                        "url": item.get("contentUrl"),
                        "thumbnail_url": item.get("thumbnailUrl"),
                        "width": item.get("width", 0),
                        "height": item.get("height", 0),
                        "size": item.get("contentSize"),
                        "name": item.get("name", ""),
                        "host_page_url": item.get("hostPageUrl"),
                        "encoding_format": item.get("encodingFormat", ""),
                        "accent_color": item.get("accentColor")
                    }
                    
                    # Only include images with valid URLs
                    if image_info["url"]:
                        images.append(image_info)
                        valid_images += 1
                    else:
                        invalid_images += 1
                
                search_time = (time.time() - search_start_time) * 1000
                
                # Log image extraction results
                decision_logger.log_decision_step(
                    request_id=request_id,
                    decision_type=DecisionType.FILTERING,
                    input_data={
                        "raw_results": len(data.get("value", [])),
                        "filtering_criteria": "valid_url_required"
                    },
                    decision_reasoning=f"Filtered search results to {valid_images} valid images out of {len(data.get('value', []))} total results by removing items without valid URLs",
                    output_data={
                        "valid_images": valid_images,
                        "invalid_images": invalid_images,
                        "filter_success_rate": valid_images / len(data.get("value", [])) if data.get("value") else 0
                    },
                    confidence_score=0.8 if valid_images > 0 else 0.3,
                    execution_time_ms=search_time,
                    metadata={
                        "filtering_stage": "url_validation",
                        "ready_for_ranking": valid_images > 0
                    }
                )
                
                logger.info(f"Found {len(images)} images for query: {optimized_query}")
                return images
                
        except httpx.HTTPError as e:
            search_time = (time.time() - search_start_time) * 1000
            error_msg = f"HTTP error during image search: {e}"
            
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"http_error": str(e), "query": query},
                decision_reasoning="HTTP error encountered during Bing API call, returning empty results to prevent workflow failure",
                output_data={"fallback_images": []},
                confidence_score=0.0,
                execution_time_ms=search_time,
                error=error_msg,
                metadata={
                    "error_type": "http_error",
                    "api_provider": "bing"
                }
            )
            
            logger.error(error_msg)
            return []
        except Exception as e:
            search_time = (time.time() - search_start_time) * 1000
            error_msg = f"Error during image search: {e}"
            
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.ERROR_HANDLING,
                input_data={"general_error": str(e), "query": query},
                decision_reasoning="Unexpected error during image search, returning empty results to maintain system stability",
                output_data={"fallback_images": []},
                confidence_score=0.0,
                execution_time_ms=search_time,
                error=error_msg,
                metadata={
                    "error_type": "general_error",
                    "recovery_strategy": "empty_results"
                }
            )
            
            logger.error(error_msg)
            return []
    
    def _optimize_search_query(
        self, 
        query: str, 
        reference_type: PromptReferenceType
    ) -> str:
        """
        Optimize search query based on reference type.
        
        Args:
            query: Original search query
            reference_type: Type of reference
            
        Returns:
            Optimized search query
        """
        query = query.strip()
        
        if reference_type == PromptReferenceType.CELEBRITY:
            # Add portrait/headshot terms for celebrities
            return f"{query} portrait headshot photo"
            
        elif reference_type == PromptReferenceType.ARTWORK:
            # Add art-specific terms
            return f"{query} artwork painting art museum"
            
        elif reference_type == PromptReferenceType.STYLE:
            # Add style-specific terms  
            return f"{query} style aesthetic visual design"
            
        elif reference_type == PromptReferenceType.BRAND:
            # Add brand/logo terms
            return f"{query} logo brand design official"
            
        return query
    
    async def _rank_images(
        self, 
        images: List[Dict[str, Any]], 
        reference: PromptReference,
        request_id: str,
        reference_index: int
    ) -> List[str]:
        """
        Rank images by relevance and return top URLs.
        
        For Phase 2, this is a simple ranking based on image metadata.
        In Phase 3+, we'll add CLIP-based semantic ranking.
        
        Args:
            images: List of image metadata from search
            reference: Original reference object
            request_id: Request ID for decision tracking
            reference_index: Index of reference being processed
            
        Returns:
            List of top 3 image URLs
        """
        ranking_start_time = time.time()
        
        if not images:
            # Log no images to rank
            decision_logger.log_decision_step(
                request_id=request_id,
                decision_type=DecisionType.RANKING,
                input_data={"images_to_rank": 0},
                decision_reasoning="No images available for ranking, returning empty results",
                output_data={"ranked_images": []},
                confidence_score=0.0,
                metadata={
                    "ranking_stage": "no_input_images",
                    "reference_index": reference_index
                }
            )
            return []
        
        # Log ranking start
        decision_logger.log_decision_step(
            request_id=request_id,
            decision_type=DecisionType.RANKING,
            input_data={
                "images_to_rank": len(images),
                "reference_text": reference.text,
                "ranking_method": "metadata_based"
            },
            decision_reasoning=f"Starting Phase 2 metadata-based ranking for {len(images)} images using quality indicators and relevance scoring",
            output_data={
                "ranking_algorithm": "metadata_scoring",
                "ranking_factors": ["size", "format", "name_relevance", "host_quality"]
            },
            confidence_score=0.7,
            metadata={
                "ranking_phase": "2",
                "ranking_type": "metadata_heuristic"
            }
        )
        
        # Simple ranking based on image quality indicators
        ranked_images = []
        
        for image in images:
            score = 0.0
            
            # Size scoring (prefer larger images)
            width = image.get("width", 0)
            height = image.get("height", 0)
            if width > 500 and height > 500:
                score += 0.3
            if width > 1000 and height > 1000:
                score += 0.2
                
            # Format scoring (prefer common formats)
            format_str = image.get("encoding_format", "").lower()
            if format_str in ["jpeg", "jpg", "png"]:
                score += 0.2
                
            # Name relevance (basic keyword matching)
            name = image.get("name", "").lower()
            query_words = reference.search_query.lower().split()
            matching_words = sum(1 for word in query_words if word in name)
            if matching_words > 0:
                score += 0.3 * (matching_words / len(query_words))
            
            # Host page relevance
            host_url = image.get("host_page_url", "").lower()
            if any(domain in host_url for domain in ["wikipedia", "imdb", "museum", "gallery"]):
                score += 0.1
                
            ranked_images.append((image["url"], score))
        
        # Sort by score descending and return top 3 URLs
        ranked_images.sort(key=lambda x: x[1], reverse=True)
        top_urls = [url for url, score in ranked_images[:3]]
        
        ranking_time = (time.time() - ranking_start_time) * 1000
        
        # Log ranking results
        decision_logger.log_decision_step(
            request_id=request_id,
            decision_type=DecisionType.RANKING,
            input_data={"ranking_complete": True},
            decision_reasoning=f"Completed metadata-based ranking, selected top {len(top_urls)} images out of {len(images)} candidates based on quality scores",
            output_data={
                "total_images_ranked": len(images),
                "top_images_selected": len(top_urls),
                "selection_rate": len(top_urls) / len(images) if images else 0,
                "top_scores": [score for url, score in ranked_images[:3]] if ranked_images else []
            },
            confidence_score=0.8 if top_urls else 0.2,
            execution_time_ms=ranking_time,
            metadata={
                "ranking_algorithm_used": "metadata_heuristic",
                "future_improvement": "clip_semantic_ranking"
            }
        )
        
        logger.debug(f"Ranked {len(images)} images, returning top {len(top_urls)}")
        return top_urls
    
    async def retrieve_celebrity_references(
        self, 
        celebrity_prompts: List[str]
    ) -> List[Tuple[str, List[str]]]:
        """
        Retrieve reference images for celebrity prompts (for testing).
        
        Args:
            celebrity_prompts: List of celebrity names/descriptions
            
        Returns:
            List of tuples (prompt, image_urls)
        """
        results = []
        
        for prompt in celebrity_prompts:
            try:
                # Create a temporary reference for testing
                reference = PromptReference(
                    text=prompt,
                    type=PromptReferenceType.CELEBRITY,
                    search_query=prompt,
                    confidence=1.0
                )
                
                # Retrieve images
                updated_references = await self.retrieve_references([reference])
                
                if updated_references:
                    image_urls = updated_references[0].image_urls
                    results.append((prompt, image_urls))
                else:
                    results.append((prompt, []))
                    
            except Exception as e:
                logger.error(f"Failed to retrieve celebrity reference for '{prompt}': {e}")
                results.append((prompt, []))
                
        return results 

    def _validate_config(self) -> None:
        """Validate API configuration."""
        # Use Perplexity API instead of Bing
        api_key = self.config.get("api_key") or os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            logger.warning("Perplexity API key not found - reference retrieval may not work")
        else:
            self.config["api_key"] = api_key 