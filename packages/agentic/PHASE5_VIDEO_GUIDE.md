# Phase 5: Video Generation Guide

Welcome to PicArcade's comprehensive video generation platform! Phase 5 introduces state-of-the-art video generation capabilities with multiple providers and advanced editing tools.

## 🎬 Overview

Phase 5 transforms PicArcade from an image-focused platform to a complete **image and video generation** solution. With support for multiple cutting-edge models and providers, you can now create professional videos for any use case.

### ✨ Key Features

- **🚀 Runway ML Integration**: Premium video generation with Gen-4 Turbo
- **🌍 Multiple Providers**: Google Veo 2, Luma Ray, HunyuanVideo, Minimax, Kling Video
- **📝 Text-to-Video**: Generate videos from text descriptions
- **🖼️ Image-to-Video**: Animate still images into dynamic videos
- **🎨 Video Editing**: Upscaling, style transfer, stabilization, effects
- **📐 Flexible Formats**: Support for multiple resolutions and aspect ratios
- **💰 Cost Optimization**: Intelligent provider selection and pricing

## 🛠️ Available Tools

### 1. RunwayVideoTool
**Premium video generation with Runway ML**

```python
from pic_arcade_agentic.tools import RunwayVideoTool

runway_tool = RunwayVideoTool()

# Text-to-video generation
result = await runway_tool.invoke({
    "prompt_text": "A cinematic shot of a cat walking through a neon-lit city at night",
    "model": "gen4_turbo",
    "ratio": "1280:720",
    "duration": 6
})
```

**Features:**
- Gen-4 Turbo and Standard models
- Text-to-video and image-to-video
- Up to 10 seconds duration
- Multiple aspect ratios (16:9, 9:16, etc.)
- Professional quality output

### 2. ReplicateVideoTool
**Multi-provider video generation via Replicate**

```python
from pic_arcade_agentic.tools import ReplicateVideoTool

replicate_tool = ReplicateVideoTool()

# Google Veo 2 - State-of-the-art quality
result = await replicate_tool.invoke({
    "prompt": "A majestic eagle soaring over mountains at golden hour",
    "provider": "google_veo2",
    "duration": 8,
    "quality": "4K"
})

# Luma Ray - Fast and high-quality
result = await replicate_tool.invoke({
    "prompt": "A coffee cup with steam rising in a cozy cafe",
    "provider": "luma_ray",
    "duration": 5,
    "quality": "1080p"
})

# Image-to-video with Minimax
result = await replicate_tool.invoke({
    "prompt": "The person starts walking forward",
    "image": "https://example.com/portrait.jpg",
    "provider": "minimax_video",
    "duration": 6
})
```

**Supported Providers:**
- **Google Veo 2**: State-of-the-art, up to 4K resolution
- **Luma Ray**: Fast, high-quality (Dream Machine)
- **HunyuanVideo**: Open-source, excellent quality
- **Minimax Video**: Great for animation and character consistency
- **Kling Video**: High-quality up to 1080p
- **VideoCrafter**: Text-to-video and image-to-video

### 3. VideoEditingTool
**Video enhancement and editing**

```python
from pic_arcade_agentic.tools import VideoEditingTool

editing_tool = VideoEditingTool()

# Upscale video
result = await editing_tool.invoke({
    "video_url": "https://example.com/video.mp4",
    "operation": "upscale",
    "upscale_factor": 2
})

# Apply style transfer
result = await editing_tool.invoke({
    "video_url": "https://example.com/video.mp4",
    "operation": "style_transfer",
    "style_prompt": "Van Gogh painting style with swirling brushstrokes"
})
```

**Editing Operations:**
- **Upscale**: 2x or 4x resolution enhancement
- **Style Transfer**: Apply artistic styles to videos
- **Enhance**: Quality improvement and artifact removal
- **Stabilize**: Video stabilization and matting
- **Motion Edit**: Add motion effects and transitions

## 🎯 Use Cases & Examples

### 📱 Social Media Content

```python
# Instagram/TikTok vertical video
result = await replicate_tool.invoke({
    "prompt": "A trendy lifestyle shot of someone enjoying coffee in a modern apartment",
    "provider": "luma_ray",
    "duration": 5,
    "quality": "1080p",
    "aspect_ratio": "9:16"  # Vertical for mobile
})
```

### 🎬 Marketing Videos

```python
# Professional product video
result = await runway_tool.invoke({
    "prompt_text": "A luxury watch rotating on a marble surface with dramatic lighting and reflections",
    "model": "gen4_turbo",
    "ratio": "1920:1080",
    "duration": 8
})
```

### 🎨 Creative Content

```python
# Artistic animation
result = await replicate_tool.invoke({
    "prompt": "A magical forest scene with floating particles and ethereal lighting",
    "provider": "hunyuan_video",
    "duration": 10,
    "quality": "720p"
})
```

### 📺 Documentary Style

```python
# Nature documentary
result = await replicate_tool.invoke({
    "prompt": "Aerial view of a pristine lake surrounded by mountains, cinematic camera movement",
    "provider": "google_veo2",
    "duration": 8,
    "quality": "4K"
})
```

## 🔄 Workflow Integration

### Chaining Operations

```python
# 1. Generate base video
video_result = await replicate_tool.invoke({
    "prompt": "A person walking through a city street",
    "provider": "luma_ray",
    "duration": 5
})

# 2. Enhance the video
enhanced_result = await editing_tool.invoke({
    "video_url": video_result.data["video_url"],
    "operation": "enhance"
})

# 3. Apply style transfer
final_result = await editing_tool.invoke({
    "video_url": enhanced_result.data["edited_video_url"],
    "operation": "style_transfer",
    "style_prompt": "Cyberpunk neon aesthetic"
})
```

### Image-to-Video Pipeline

```python
# Start with image generation (existing Flux tools)
image_result = await flux_tool.invoke({
    "prompt": "A serene mountain landscape at sunset",
    "style": "photorealistic"
})

# Animate the image
video_result = await replicate_tool.invoke({
    "prompt": "Gentle camera movement revealing the landscape",
    "image": image_result.data["image_url"],
    "provider": "minimax_video",
    "duration": 6
})
```

## 📊 Provider Comparison

| Provider | Max Duration | Quality Options | Strengths | Best For |
|----------|-------------|----------------|-----------|----------|
| **Google Veo 2** | 8s | 720p, 1080p, 4K | State-of-the-art quality | High-end productions |
| **Luma Ray** | 5s | 720p, 1080p | Speed + Quality | General use, social media |
| **Runway ML** | 10s | 720p, 1080p | Professional grade | Commercial projects |
| **HunyuanVideo** | 16s | 480p, 720p, 1080p | Open source, long videos | Creative projects |
| **Minimax Video** | 6s | 720p, 1080p | Animation, consistency | Character-focused content |
| **Kling Video** | 10s | 720p, 1080p | Balanced quality/cost | General purpose |

## 💰 Cost Optimization

### Provider Selection Strategy

```python
def choose_provider(use_case, budget, quality_priority):
    if quality_priority == "ultra_high" and budget == "high":
        return "google_veo2"
    elif quality_priority == "high" and budget == "medium":
        return "luma_ray"
    elif use_case == "animation":
        return "minimax_video"
    elif budget == "low":
        return "hunyuan_video"
    else:
        return "luma_ray"  # Good default
```

### Cost Estimates (per video)

| Provider | 5s Video | 720p | 1080p | 4K |
|----------|----------|------|-------|-----|
| Google Veo 2 | $0.40 | $0.60 | $1.20 | $4.80 |
| Luma Ray | $0.25 | $0.40 | $0.80 | N/A |
| Runway ML | $0.30 | $0.50 | $0.75 | N/A |
| HunyuanVideo | $0.02 | $0.03 | $0.06 | N/A |
| Minimax Video | $0.05 | $0.08 | $0.15 | N/A |

## 🚀 Getting Started

### 1. Install Dependencies

```bash
# Install Runway SDK
pip install runwayml>=1.0.0

# Ensure Replicate is installed
pip install replicate>=0.25.1
```

### 2. Set API Keys

```bash
# Required for Replicate models
export REPLICATE_API_TOKEN="your_token_here"

# Optional for Runway ML
export RUNWAYML_API_SECRET="your_secret_here"
```

### 3. Run Demo

```bash
# Comprehensive Phase 5 demo
npm run demo:phase5

# Or directly with Python
python examples/phase5_video_demo.py
```

### 4. Run Tests

```bash
# Test video generation tools
npm run test:phase5

# Test specific tools
npm run test:video
```

## 🎛️ Advanced Configuration

### Custom Video Parameters

```python
# Fine-tuned configuration
result = await replicate_tool.invoke({
    "prompt": "Your detailed prompt here",
    "provider": "luma_ray",
    "duration": 5,
    "quality": "1080p",
    "aspect_ratio": "16:9",
    "fps": 30,
    "seed": 42,  # For reproducible results
    "motion_strength": 0.8  # High motion
})
```

### Error Handling

```python
try:
    result = await replicate_tool.invoke(input_data)
    if result.success:
        video_url = result.data["video_url"]
        print(f"Generated: {video_url}")
    else:
        print(f"Error: {result.data['error']}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 📈 Performance Tips

### 1. **Choose the Right Provider**
- Use Google Veo 2 for maximum quality
- Use Luma Ray for balance of speed/quality
- Use HunyuanVideo for cost-effective generation

### 2. **Optimize Prompts**
- Be specific about camera movements
- Include style descriptors (cinematic, professional, etc.)
- Mention lighting and atmosphere

### 3. **Quality vs. Speed Trade-offs**
- 720p generates faster than 1080p
- Shorter videos (4-5s) complete quicker
- Some providers are inherently faster

### 4. **Batch Processing**
- Generate multiple videos concurrently
- Use different providers to distribute load
- Implement retry logic for failures

## 🔮 Future Enhancements

Phase 5 sets the foundation for future video capabilities:

- **Multi-shot Videos**: Longer, scene-based content
- **Video-to-Video**: Transform existing videos
- **Advanced Editing**: Professional-grade effects
- **Audio Integration**: Synchronized audio generation
- **Real-time Preview**: Interactive video creation

## 📚 Additional Resources

- **[Phase 5 Demo](examples/phase5_video_demo.py)**: Complete demonstration
- **[Test Suite](tests/test_phase5_video.py)**: Comprehensive tests
- **[API Documentation](video_tools.py)**: Detailed tool documentation
- **[Runway Documentation](https://docs.runwayml.com/)**: Official Runway guides
- **[Replicate Models](https://replicate.com/collections/text-to-video)**: Available video models

---

**🎉 Phase 5 Complete!** You now have access to a comprehensive video generation platform that rivals any commercial solution. Start creating amazing videos today! 