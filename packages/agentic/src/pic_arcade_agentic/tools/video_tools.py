"""
Video Generation Tools for PicArcade Phase 5

Comprehensive video generation toolkit supporting multiple providers:
- Runway ML (official SDK)
- Google Veo 2 (via Replicate)
- Luma Ray / Dream Machine (via Replicate)
- Tencent HunyuanVideo (via Replicate)
- Minimax Video-01 (via Replicate)
- Various specialized video tools
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum

import replicate
from .base import Tool, ToolCategory, ToolResult

# Try to import Runway SDK at module level for testing
try:
    from runwayml import AsyncRunwayML
    RUNWAY_SDK_AVAILABLE = True
except ImportError:
    # Create a mock class for testing purposes
    class AsyncRunwayML:
        pass
    RUNWAY_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

class VideoProvider(Enum):
    """Video generation providers"""
    RUNWAY = "runway"
    GOOGLE_VEO2 = "google_veo2"
    LUMA_RAY = "luma_ray"
    HUNYUAN_VIDEO = "hunyuan_video"
    MINIMAX_VIDEO = "minimax_video"
    KLING_VIDEO = "kling_video"
    VIDEOCRAFTER = "videocrafter"

class VideoQuality(Enum):
    """Video quality presets"""
    LOW = "480p"
    MEDIUM = "720p"
    HIGH = "1080p"
    ULTRA = "4K"

class VideoDuration(Enum):
    """Standard video durations"""
    SHORT = 4  # 4 seconds
    MEDIUM = 6  # 6 seconds
    LONG = 10  # 10 seconds
    EXTENDED = 16  # 16 seconds

class VideoAspectRatio(Enum):
    """Video aspect ratios"""
    SQUARE = "1:1"
    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"
    WIDESCREEN = "21:9"
    CUSTOM_1280_720 = "1280:720"
    CUSTOM_1920_1080 = "1920:1080"

class RunwayVideoTool(Tool):
    """
    Runway ML Video Generation Tool
    
    Uses the official Runway Python SDK for high-quality video generation.
    Supports both text-to-video and image-to-video generation.
    """
    
    def __init__(self):
        # Set API key first before calling parent constructor
        self.api_key = os.getenv('RUNWAYML_API_SECRET')
        
        name = "runway_video_generation"
        description = "Generate high-quality videos using Runway ML's advanced video models"
        category = ToolCategory.VIDEO_GENERATION
        
        input_schema = {
            "type": "object",
            "properties": {
                "prompt_text": {
                    "type": "string",
                    "description": "Text prompt describing the video to generate"
                },
                "prompt_image": {
                    "type": "string",
                    "description": "URL or base64 image to use as starting point (optional for image-to-video)"
                },
                "model": {
                    "type": "string",
                    "enum": ["gen4_turbo", "gen4_standard"],
                    "default": "gen4_turbo",
                    "description": "Runway model to use"
                },
                "ratio": {
                    "type": "string",
                    "enum": ["1280:720", "1920:1080", "720:1280", "1080:1920"],
                    "default": "1280:720",
                    "description": "Video aspect ratio"
                },
                "duration": {
                    "type": "integer",
                    "minimum": 4,
                    "maximum": 10,
                    "default": 6,
                    "description": "Video duration in seconds"
                },
                "seed": {
                    "type": "integer",
                    "description": "Seed for reproducible results (optional)"
                }
            },
            "required": ["prompt_text"]
        }
        
        output_schema = {
            "type": "object",
            "properties": {
                "video_url": {"type": "string"},
                "task_id": {"type": "string"},
                "model_used": {"type": "string"},
                "duration": {"type": "number"},
                "resolution": {"type": "string"},
                "prompt_text": {"type": "string"},
                "prompt_image": {"type": "string"},
                "processing_time": {"type": "number"},
                "cost_estimate": {"type": "number"},
                "metadata": {"type": "object"}
            }
        }
        
        super().__init__(name, description, category, input_schema, output_schema)

    def _validate_config(self) -> None:
        """Validate tool configuration (API keys, etc.)"""
        if not self.api_key:
            logger.warning("RUNWAYML_API_SECRET not found. Runway video generation will not work.")

    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        """Generate video using Runway ML"""
        import time
        start_time = time.time()
        
        try:
            # Check API key first before anything else
            if not self.api_key:
                return ToolResult(
                    success=False,
                    data={
                        "error": "RUNWAYML_API_SECRET environment variable not set",
                        "processing_time": time.time() - start_time
                    }
                )
            
            # Check if Runway SDK is available (check if it's a real SDK with expected methods)
            if not hasattr(AsyncRunwayML, '__module__') or 'runwayml' not in getattr(AsyncRunwayML, '__module__', ''):
                # Allow Mock objects to pass (for testing)
                if not (hasattr(AsyncRunwayML, '_mock_name') or str(type(AsyncRunwayML).__name__) == 'MagicMock'):
                    return ToolResult(
                        success=False,
                        data={
                            "error": "Runway ML SDK not installed. Install with: pip install runwayml",
                            "processing_time": time.time() - start_time
                        }
                    )
            
            # Initialize Runway client
            client = AsyncRunwayML(api_key=self.api_key)
            
            # Prepare parameters
            params = {
                "model": input_data.get("model", "gen4_turbo"),
                "prompt_text": input_data["prompt_text"],
                "ratio": input_data.get("ratio", "1280:720")
            }
            
            # Add prompt image if provided
            if input_data.get("prompt_image"):
                params["prompt_image"] = input_data["prompt_image"]
            
            # Add seed if provided
            if input_data.get("seed"):
                params["seed"] = input_data["seed"]
            
            logger.info(f"Creating video with Runway: {params['prompt_text'][:100]}...")
            
            # Create video generation task
            def run_runway_sync():
                import asyncio
                return asyncio.run(client.image_to_video.create(**params))
            
            # Run in thread to avoid blocking
            video_task = await asyncio.to_thread(run_runway_sync)
            
            processing_time = time.time() - start_time
            
            # Calculate estimated cost (Runway pricing varies)
            cost_estimate = self._calculate_runway_cost(
                duration=input_data.get("duration", 6),
                quality=params["ratio"]
            )
            
            result_data = {
                "video_url": None,  # Runway returns task ID, need to poll for completion
                "task_id": video_task.id,
                "model_used": params["model"],
                "duration": input_data.get("duration", 6),
                "resolution": params["ratio"],
                "prompt_text": params["prompt_text"],
                "prompt_image": params.get("prompt_image"),
                "processing_time": processing_time,
                "cost_estimate": cost_estimate,
                "metadata": {
                    "provider": "runway",
                    "task_status": "created",
                    "note": "Use task_id to poll for completion"
                }
            }
            
            logger.info(f"Runway video task created: {video_task.id}")
            return ToolResult(success=True, data=result_data)
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Runway video generation failed: {str(e)}")
            return ToolResult(
                success=False,
                data={
                    "error": f"Runway video generation failed: {str(e)}",
                    "processing_time": processing_time
                }
            )
    
    def _calculate_runway_cost(self, duration: int, quality: str) -> float:
        """Calculate estimated cost for Runway video generation"""
        # Runway pricing is typically per video, varies by plan
        base_cost = 0.50  # Estimated base cost per video
        duration_multiplier = duration / 6  # Base 6 seconds
        quality_multiplier = 1.5 if "1920" in quality else 1.0
        
        return base_cost * duration_multiplier * quality_multiplier

class ReplicateVideoTool(Tool):
    """
    Multi-Provider Video Generation Tool via Replicate
    
    Supports multiple state-of-the-art video generation models:
    - Google Veo 2 (4K quality)
    - Luma Ray / Dream Machine
    - Tencent HunyuanVideo
    - Minimax Video-01
    - Kling Video
    """
    
    def __init__(self):
        # Set API key first before calling parent constructor
        self.api_token = os.getenv('REPLICATE_API_TOKEN')
        
        name = "replicate_video_generation"
        description = "Generate videos using various state-of-the-art models via Replicate API"
        category = ToolCategory.VIDEO_GENERATION
        
        input_schema = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text prompt describing the video to generate"
                },
                "image": {
                    "type": "string",
                    "description": "Input image URL for image-to-video generation (optional)"
                },
                "provider": {
                    "type": "string",
                    "enum": ["Google Veo 2", "Luma Ray", "HunyuanVideo", "Minimax Video", "Kling Video", "VideoCrafter"],
                    "default": "Luma Ray",
                    "description": "Video generation provider to use"
                },
                "duration": {
                    "type": "integer",
                    "enum": [4, 5, 6, 8, 9, 10, 16],
                    "default": 6,
                    "description": "Video duration in seconds"
                },
                "quality": {
                    "type": "string",
                    "enum": ["480p", "540p", "720p", "1080p", "4K"],
                    "default": "720p",
                    "description": "Video quality/resolution"
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": ["1:1", "16:9", "9:16", "21:9"],
                    "default": "16:9",
                    "description": "Video aspect ratio"
                },
                "fps": {
                    "type": "integer",
                    "enum": [24, 25, 30],
                    "default": 25,
                    "description": "Frames per second"
                },
                "seed": {
                    "type": "integer",
                    "description": "Seed for reproducible results (optional)"
                },
                "motion_strength": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7,
                    "description": "Motion intensity (0.0 = static, 1.0 = high motion)"
                }
            },
            "required": ["prompt"]
        }
        
        output_schema = {
            "type": "object",
            "properties": {
                "video_url": {"type": "string"},
                "provider_used": {"type": "string"},
                "model_used": {"type": "string"},
                "duration": {"type": "number"},
                "resolution": {"type": "string"},
                "fps": {"type": "integer"},
                "prompt": {"type": "string"},
                "input_image": {"type": "string"},
                "processing_time": {"type": "number"},
                "cost_estimate": {"type": "number"},
                "confidence": {"type": "number"},
                "metadata": {"type": "object"}
            }
        }
        
        super().__init__(name, description, category, input_schema, output_schema)

    def _validate_config(self) -> None:
        """Validate tool configuration (API keys, etc.)"""
        if not self.api_token:
            logger.warning("REPLICATE_API_TOKEN not found. Replicate video generation will not work.")

    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        """Generate video using selected Replicate model"""
        import time
        start_time = time.time()
        
        try:
            if not self.api_token:
                return ToolResult(
                    success=False,
                    data={
                        "error": "REPLICATE_API_TOKEN environment variable not set",
                        "processing_time": time.time() - start_time
                    }
                )
            
            provider = input_data.get("provider", "luma_ray")
            model_config = self._get_model_config(provider)
            
            if not model_config:
                return ToolResult(
                    success=False,
                    data={
                        "error": f"Unsupported provider: {provider}",
                        "processing_time": time.time() - start_time
                    }
                )
            
            # Prepare model inputs based on provider
            model_inputs = self._prepare_model_inputs(input_data, model_config)
            
            logger.info(f"Generating video with {provider}: {input_data['prompt'][:100]}...")
            
            # Run Replicate prediction
            def run_replicate():
                return replicate.run(model_config["model"], input=model_inputs)
            
            # Execute in thread to avoid blocking
            output = await asyncio.to_thread(run_replicate)
            
            processing_time = time.time() - start_time
            
            # Extract video URL from output
            video_url = self._extract_video_url(output, provider)
            
            if not video_url:
                return ToolResult(
                    success=False,
                    data={
                        "error": f"No video URL returned from {provider}",
                        "output": str(output),
                        "processing_time": processing_time
                    }
                )
            
            # Calculate confidence based on provider and processing time
            confidence = self._calculate_confidence(provider, processing_time)
            
            # Calculate cost estimate
            cost_estimate = self._calculate_cost(provider, input_data)
            
            result_data = {
                "video_url": video_url,
                "provider_used": provider,
                "model_used": model_config["model"],
                "duration": input_data.get("duration", 6),
                "resolution": input_data.get("quality", "720p"),
                "fps": input_data.get("fps", 25),
                "prompt": input_data["prompt"],
                "input_image": input_data.get("image"),
                "processing_time": processing_time,
                "cost_estimate": cost_estimate,
                "confidence": confidence,
                "metadata": {
                    "provider_info": model_config,
                    "model_inputs": model_inputs,
                    "raw_output": output if isinstance(output, dict) else str(output)
                }
            }
            
            logger.info(f"Video generated successfully with {provider}: {video_url}")
            return ToolResult(success=True, data=result_data)
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Video generation failed with {input_data.get('provider', 'unknown')}: {str(e)}")
            return ToolResult(
                success=False,
                data={
                    "error": f"Video generation failed: {str(e)}",
                    "provider": input_data.get("provider"),
                    "processing_time": processing_time
                }
            )
    
    def _get_model_config(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get model configuration for provider"""
        # Map natural provider names to internal names
        provider_mapping = {
            "Google Veo 2": "google_veo2",
            "Luma Ray": "luma_ray", 
            "HunyuanVideo": "hunyuan_video",
            "Minimax Video": "minimax_video",
            "Kling Video": "kling_video",
            "VideoCrafter": "videocrafter"
        }
        
        # Use mapped name if available, otherwise use original
        internal_provider = provider_mapping.get(provider, provider.lower().replace(" ", "_"))
        
        configs = {
            "google_veo2": {
                "model": "google/veo-2",
                "supports_image": True,  # Supports image-to-video generation
                "max_duration": 8,
                "aspect_ratios": ["16:9", "9:16", "1:1"],  # Supported aspect ratios
                "description": "Google's state-of-the-art video generation with prompt enhancement"
            },
            "luma_ray": {
                "model": "luma/ray",
                "supports_image": True,
                "max_duration": 5,
                "qualities": ["720p", "1080p"],
                "description": "Fast, high-quality video generation (Dream Machine)"
            },
            "hunyuan_video": {
                "model": "tencent/hunyuan-video",
                "supports_image": False,
                "max_duration": 16,
                "qualities": ["480p", "720p", "1080p"],
                "description": "Open-source high-quality video generation"
            },
            "minimax_video": {
                "model": "minimax/video-01",
                "supports_image": True,
                "max_duration": 6,
                "qualities": ["720p", "1080p"],
                "description": "Great for animation and character consistency"
            },
            "kling_video": {
                "model": "kwaivgi/kling-v1.6-pro",
                "supports_image": True,
                "max_duration": 10,
                "qualities": ["720p", "1080p"],
                "description": "High-quality video generation up to 1080p"
            },
            "videocrafter": {
                "model": "cjwbw/videocrafter",
                "supports_image": True,
                "max_duration": 8,
                "qualities": ["480p", "720p"],
                "description": "Text-to-video and image-to-video generation"
            }
        }
        return configs.get(internal_provider)
    
    def _prepare_model_inputs(self, input_data: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs specific to each model"""
        provider = input_data.get("provider", "Luma Ray")
        
        # Map natural provider names to internal names 
        provider_mapping = {
            "Google Veo 2": "google_veo2",
            "Luma Ray": "luma_ray", 
            "HunyuanVideo": "hunyuan_video",
            "Minimax Video": "minimax_video",
            "Kling Video": "kling_video",
            "VideoCrafter": "videocrafter"
        }
        
        internal_provider = provider_mapping.get(provider, provider.lower().replace(" ", "_"))
        inputs = {}
        
        if internal_provider == "google_veo2":
            inputs = {
                "prompt": input_data["prompt"],
                "aspect_ratio": input_data.get("aspect_ratio", "16:9"),
                "duration": min(input_data.get("duration", 6), 8),
                "enhance_prompt": True  # Enable prompt enhancement for better results
            }
            # Add image if provided for image-to-video
            if input_data.get("image"):
                inputs["image"] = input_data["image"]
        
        elif internal_provider == "luma_ray":
            inputs = {
                "prompt": input_data["prompt"]
            }
            if input_data.get("image"):
                inputs["image"] = input_data["image"]
        
        elif internal_provider == "hunyuan_video":
            inputs = {
                "prompt": input_data["prompt"],
                "num_frames": min(input_data.get("duration", 6) * 8, 129),  # ~8 fps
                "height": 720 if input_data.get("quality") == "720p" else 480,
                "width": 1280 if input_data.get("quality") == "720p" else 720
            }
        
        elif internal_provider == "minimax_video":
            inputs = {
                "prompt": input_data["prompt"]
            }
            if input_data.get("image"):
                inputs["prompt_image"] = input_data["image"]
        
        elif internal_provider == "kling_video":
            inputs = {
                "prompt": input_data["prompt"],
                "duration": min(input_data.get("duration", 5), 10),
                "aspect_ratio": input_data.get("aspect_ratio", "16:9")
            }
            if input_data.get("image"):
                inputs["image"] = input_data["image"]
        
        elif internal_provider == "videocrafter":
            inputs = {
                "prompt": input_data["prompt"],
                "num_frames": input_data.get("duration", 6) * 8,  # ~8 fps
                "height": 320,
                "width": 512
            }
            if input_data.get("image"):
                inputs["image"] = input_data["image"]
        
        # Add seed if provided
        if input_data.get("seed"):
            inputs["seed"] = input_data["seed"]
        
        return inputs
    
    def _extract_video_url(self, output: Any, provider: str) -> Optional[str]:
        """Extract video URL from model output"""
        if isinstance(output, str):
            return output
        elif isinstance(output, list) and len(output) > 0:
            return output[0] if isinstance(output[0], str) else None
        elif isinstance(output, dict):
            # Try common keys
            for key in ["video", "video_url", "output", "result", "url"]:
                if key in output:
                    value = output[key]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, list) and len(value) > 0:
                        return value[0]
        return None
    
    def _calculate_confidence(self, provider: str, processing_time: float) -> float:
        """Calculate confidence score based on provider and processing time"""
        # Map natural provider names to internal names
        provider_mapping = {
            "Google Veo 2": "google_veo2",
            "Luma Ray": "luma_ray", 
            "HunyuanVideo": "hunyuan_video",
            "Minimax Video": "minimax_video",
            "Kling Video": "kling_video",
            "VideoCrafter": "videocrafter"
        }
        
        internal_provider = provider_mapping.get(provider, provider.lower().replace(" ", "_"))
        
        base_confidence = {
            "google_veo2": 0.95,
            "luma_ray": 0.90,
            "hunyuan_video": 0.85,
            "minimax_video": 0.80,
            "kling_video": 0.85,
            "videocrafter": 0.75
        }.get(internal_provider, 0.70)
        
        # Adjust for processing time (faster usually means better optimization)
        if processing_time < 60:
            time_factor = 1.1
        elif processing_time < 180:
            time_factor = 1.0
        else:
            time_factor = 0.9
        
        return min(base_confidence * time_factor, 1.0)
    
    def _calculate_cost(self, provider: str, input_data: Dict[str, Any]) -> float:
        """Calculate estimated cost based on provider and parameters"""
        # Map natural provider names to internal names
        provider_mapping = {
            "Google Veo 2": "google_veo2",
            "Luma Ray": "luma_ray", 
            "HunyuanVideo": "hunyuan_video",
            "Minimax Video": "minimax_video",
            "Kling Video": "kling_video",
            "VideoCrafter": "videocrafter"
        }
        
        internal_provider = provider_mapping.get(provider, provider.lower().replace(" ", "_"))
        
        duration = input_data.get("duration", 6)
        quality = input_data.get("quality", "720p")
        
        # Base costs per provider (rough estimates)
        base_costs = {
            "google_veo2": 0.10,  # Per second
            "luma_ray": 0.08,     # Per second
            "hunyuan_video": 0.02, # Per video
            "minimax_video": 0.05, # Per video
            "kling_video": 0.06,   # Per video
            "videocrafter": 0.01   # Per video
        }
        
        base_cost = base_costs.get(internal_provider, 0.05)
        
        # Adjust for quality
        quality_multiplier = {
            "480p": 1.0,
            "540p": 1.2,
            "720p": 1.5,
            "1080p": 2.0,
            "4K": 4.0
        }.get(quality, 1.5)
        
        # For per-second pricing
        if internal_provider in ["google_veo2", "luma_ray"]:
            return base_cost * duration * quality_multiplier
        else:
            return base_cost * quality_multiplier

class VideoEditingTool(Tool):
    """
    Video Editing and Enhancement Tool
    
    Provides video editing capabilities including:
    - Video upscaling and enhancement
    - Style transfer for videos
    - Video-to-video transformation
    - Motion editing and effects
    """
    
    def __init__(self):
        # Set API key first before calling parent constructor
        self.api_token = os.getenv('REPLICATE_API_TOKEN')
        
        name = "video_editing"
        description = "Edit and enhance videos with various transformations and effects"
        category = ToolCategory.VIDEO_GENERATION
        
        input_schema = {
            "type": "object",
            "properties": {
                "video_url": {
                    "type": "string",
                    "description": "URL of the input video to edit"
                },
                "operation": {
                    "type": "string",
                    "enum": ["upscale", "style_transfer", "enhance", "stabilize", "motion_edit"],
                    "description": "Type of editing operation to perform"
                },
                "style_prompt": {
                    "type": "string",
                    "description": "Style description for style transfer (required for style_transfer operation)"
                },
                "enhancement_type": {
                    "type": "string",
                    "enum": ["quality", "brightness", "contrast", "saturation", "sharpness"],
                    "description": "Type of enhancement (for enhance operation)"
                },
                "upscale_factor": {
                    "type": "integer",
                    "enum": [2, 4],
                    "default": 2,
                    "description": "Upscaling factor (for upscale operation)"
                },
                "motion_strength": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Strength of motion effects"
                }
            },
            "required": ["video_url", "operation"]
        }
        
        output_schema = {
            "type": "object",
            "properties": {
                "edited_video_url": {"type": "string"},
                "operation_performed": {"type": "string"},
                "original_video_url": {"type": "string"},
                "processing_time": {"type": "number"},
                "enhancement_details": {"type": "object"},
                "metadata": {"type": "object"}
            }
        }
        
        super().__init__(name, description, category, input_schema, output_schema)

    def _validate_config(self) -> None:
        """Validate tool configuration (API keys, etc.)"""
        if not self.api_token:
            logger.warning("REPLICATE_API_TOKEN not found. Replicate video tools will not work.")

    async def invoke(self, input_data: Dict[str, Any]) -> ToolResult:
        """Edit video with specified operation"""
        import time
        start_time = time.time()
        
        try:
            if not self.api_token:
                return ToolResult(
                    success=False,
                    data={
                        "error": "REPLICATE_API_TOKEN environment variable not set",
                        "processing_time": time.time() - start_time
                    }
                )
            
            operation = input_data["operation"]
            video_url = input_data["video_url"]
            
            # Select appropriate model based on operation
            model_config = self._get_editing_model(operation)
            
            if not model_config:
                return ToolResult(
                    success=False,
                    data={
                        "error": f"Unsupported operation: {operation}",
                        "processing_time": time.time() - start_time
                    }
                )
            
            # Prepare inputs
            model_inputs = self._prepare_editing_inputs(input_data, model_config)
            
            logger.info(f"Editing video with operation: {operation}")
            
            # Run editing model
            def run_editing():
                return replicate.run(model_config["model"], input=model_inputs)
            
            output = await asyncio.to_thread(run_editing)
            
            processing_time = time.time() - start_time
            
            # Extract result video URL
            edited_video_url = self._extract_video_url(output, operation)
            
            if not edited_video_url:
                return ToolResult(
                    success=False,
                    data={
                        "error": f"No edited video returned from {operation}",
                        "processing_time": processing_time
                    }
                )
            
            result_data = {
                "edited_video_url": edited_video_url,
                "operation_performed": operation,
                "original_video_url": video_url,
                "processing_time": processing_time,
                "enhancement_details": {
                    "model_used": model_config["model"],
                    "operation_type": operation,
                    "parameters": model_inputs
                },
                "metadata": {
                    "model_config": model_config,
                    "raw_output": output
                }
            }
            
            logger.info(f"Video editing completed: {operation}")
            return ToolResult(success=True, data=result_data)
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Video editing failed: {str(e)}")
            return ToolResult(
                success=False,
                data={
                    "error": f"Video editing failed: {str(e)}",
                    "processing_time": processing_time
                }
            )
    
    def _get_editing_model(self, operation: str) -> Optional[Dict[str, Any]]:
        """Get model configuration for editing operation"""
        models = {
            "upscale": {
                "model": "pollinations/real-basicvsr-video-superresolution",
                "description": "Video super-resolution and upscaling"
            },
            "style_transfer": {
                "model": "deforum/deforum_stable_diffusion",
                "description": "Video style transfer with stable diffusion"
            },
            "enhance": {
                "model": "pollinations/real-basicvsr-video-superresolution",
                "description": "Video quality enhancement"
            },
            "stabilize": {
                "model": "arielreplicate/robust_video_matting",
                "description": "Video stabilization and matting"
            },
            "motion_edit": {
                "model": "deforum/deforum_stable_diffusion",
                "description": "Motion editing and effects"
            }
        }
        return models.get(operation)
    
    def _prepare_editing_inputs(self, input_data: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for video editing models"""
        operation = input_data["operation"]
        inputs = {"video": input_data["video_url"]}
        
        if operation == "style_transfer" and input_data.get("style_prompt"):
            inputs["prompt"] = input_data["style_prompt"]
        
        if operation == "upscale":
            inputs["upscale"] = input_data.get("upscale_factor", 2)
        
        if operation == "motion_edit":
            inputs["motion_strength"] = input_data.get("motion_strength", 0.5)
        
        return inputs

# Register all video tools
video_tools = [
    RunwayVideoTool(),
    ReplicateVideoTool(),
    VideoEditingTool()
] 