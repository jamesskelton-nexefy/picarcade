# Flux Kontext Max Usage Guide

## Overview

The **black-forest-labs/flux-kontext-max** model integration provides Pic Arcade with cutting-edge image generation and editing capabilities. This guide covers all available tools and their professional use cases.

## Model Capabilities

Based on the documentation in `docs/flux.md`, Flux Kontext Max excels at:

- **Style Transfer**: Converting photos to different art styles
- **Object/Clothing Changes**: Modifying specific elements naturally  
- **Text Editing**: Replacing text in signs, posters, labels
- **Background Swapping**: Changing environments while preserving subjects
- **Character Consistency**: Maintaining identity across multiple edits

## Available Tools

### 1. FluxKontextMaxTool (Unified Tool)

The main tool that provides access to all Flux Kontext Max capabilities through different operation types.

```python
from pic_arcade_agentic.tools import FluxKontextMaxTool

# Initialize with Replicate API configuration
flux_tool = FluxKontextMaxTool({
    "api_key": "your_replicate_token"
})

# Generation
result = await flux_tool.invoke({
    "prompt": "professional portrait of a confident business woman",
    "operation_type": "generation",
    "aspect_ratio": "3:4",
    "guidance": 7.5,
    "steps": 28
})

# Style Transfer
result = await flux_tool.invoke({
    "prompt": "transform this portrait to watercolor style",
    "image": "base64_encoded_image_or_url",
    "operation_type": "style_transfer", 
    "style": "watercolor",
    "strength": 0.7,
    "preserve_details": True
})
```

### 2. StyleTransferTool

Specialized tool for converting photos to different art styles.

```python
from pic_arcade_agentic.tools import StyleTransferTool

style_tool = StyleTransferTool(config)

result = await style_tool.invoke({
    "image": "portrait.jpg",
    "style": "oil_painting",  # watercolor, sketch, digital_art, vintage_photo, impressionist, abstract
    "prompt": "professional portrait",
    "strength": 0.8,
    "preserve_details": True
})

# Output: styled_image, style_applied, processing_time
```

### 3. ObjectChangeTool

Modify specific objects and clothing while maintaining natural integration.

```python
from pic_arcade_agentic.tools import ObjectChangeTool

object_tool = ObjectChangeTool(config)

result = await object_tool.invoke({
    "image": "person.jpg",
    "target_object": "hair",
    "modification": "blonde curly hair",
    "strength": 0.8,
    "preserve_lighting": True
})

# Output: modified_image, object_modified, modification_applied, processing_time
```

### 4. TextEditingTool

Replace text in signs, posters, and labels while maintaining typography.

```python
from pic_arcade_agentic.tools import TextEditingTool

text_tool = TextEditingTool(config)

result = await text_tool.invoke({
    "image": "storefront.jpg",
    "original_text": "Old Store Name",  # Optional
    "new_text": "PIC ARCADE",
    "text_location": "on the storefront sign",
    "maintain_style": True
})

# Output: edited_image, new_text, text_location, processing_time
```

### 5. BackgroundSwapTool

Change environments while preserving subject integrity and lighting.

```python
from pic_arcade_agentic.tools import BackgroundSwapTool

bg_tool = BackgroundSwapTool(config)

result = await bg_tool.invoke({
    "image": "portrait.jpg",
    "new_background": "modern office with city view",
    "environment_type": "indoor",  # outdoor, studio, fantasy, abstract
    "lighting_match": True,
    "strength": 0.8
})

# Output: swapped_image, background_description, environment_type, processing_time
```

### 6. CharacterConsistencyTool

Maintain character identity across different poses, expressions, and edits.

```python
from pic_arcade_agentic.tools import CharacterConsistencyTool

character_tool = CharacterConsistencyTool(config)

result = await character_tool.invoke({
    "reference_image": "character_ref.jpg",
    "character_description": "young woman with brown hair and blue eyes",
    "new_scenario": "laughing with arms crossed",
    "maintain_features": ["facial_features", "hair", "body_proportions"],
    "variation_strength": 0.4
})

# Output: consistent_image, character_description, scenario_applied, features_maintained, processing_time
```

## Configuration

### Environment Setup

```bash
# Required environment variable
export REPLICATE_API_TOKEN="your_token_here"

# Optional: Configure logging
export PYTHONPATH="${PYTHONPATH}:./packages/agentic/src"
```

### Tool Configuration

```python
config = {
    "api_key": os.getenv("REPLICATE_API_TOKEN"),
    # Additional configuration options can be added here
}
```

## Use Cases by Industry

### Photography & Portrait Studios

```python
# Professional headshot enhancement
style_result = await style_tool.invoke({
    "image": "raw_headshot.jpg",
    "style": "digital_art",
    "strength": 0.3,  # Subtle enhancement
    "preserve_details": True
})

# Background replacement for consistent branding
bg_result = await bg_tool.invoke({
    "image": "portrait.jpg", 
    "new_background": "professional studio backdrop",
    "environment_type": "studio",
    "lighting_match": True
})
```

### Marketing & Advertising

```python
# Product placement with text updates
text_result = await text_tool.invoke({
    "image": "billboard.jpg",
    "new_text": "Your Brand Here",
    "text_location": "on the main billboard",
    "maintain_style": True
})

# Style variations for campaigns
style_variations = []
for style in ["digital_art", "vintage_photo", "impressionist"]:
    result = await style_tool.invoke({
        "image": "campaign_image.jpg",
        "style": style,
        "strength": 0.6
    })
    style_variations.append(result)
```

### Creative Content & Art

```python
# Artistic transformations
artistic_styles = ["watercolor", "oil_painting", "sketch", "abstract"]
for style in artistic_styles:
    result = await style_tool.invoke({
        "image": "photo.jpg",
        "style": style,
        "strength": 0.8,
        "preserve_details": False  # Allow more dramatic transformation
    })
```

### Character Design & Gaming

```python
# Consistent character across scenarios
scenarios = [
    "standing confidently",
    "running in action pose", 
    "sitting thoughtfully"
]

character_variations = []
for scenario in scenarios:
    result = await character_tool.invoke({
        "reference_image": "character_base.jpg",
        "character_description": "fantasy warrior with silver armor",
        "new_scenario": scenario,
        "maintain_features": ["facial_features", "clothing", "body_proportions"],
        "variation_strength": 0.3
    })
    character_variations.append(result)
```

## Advanced Workflows

### Multi-Step Editing Pipeline

```python
async def advanced_portrait_workflow(original_image: str) -> Dict[str, Any]:
    """Complete portrait enhancement workflow."""
    
    # Step 1: Object modification (hair color)
    hair_result = await object_tool.invoke({
        "image": original_image,
        "target_object": "hair", 
        "modification": "platinum blonde hair",
        "strength": 0.7
    })
    
    # Step 2: Background replacement
    bg_result = await bg_tool.invoke({
        "image": hair_result.data["modified_image"],
        "new_background": "luxury penthouse interior",
        "environment_type": "indoor",
        "lighting_match": True
    })
    
    # Step 3: Style enhancement
    final_result = await style_tool.invoke({
        "image": bg_result.data["swapped_image"],
        "style": "digital_art",
        "strength": 0.4,
        "preserve_details": True
    })
    
    return {
        "original": original_image,
        "hair_modified": hair_result.data["modified_image"],
        "background_swapped": bg_result.data["swapped_image"], 
        "final_styled": final_result.data["styled_image"],
        "total_processing_time": (
            hair_result.data["processing_time"] + 
            bg_result.data["processing_time"] + 
            final_result.data["processing_time"]
        )
    }
```

### Batch Processing

```python
async def batch_style_transfer(images: List[str], target_style: str) -> List[Dict]:
    """Apply same style to multiple images."""
    
    tasks = []
    for image in images:
        task = style_tool.invoke({
            "image": image,
            "style": target_style,
            "strength": 0.7,
            "preserve_details": True
        })
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return [r.data for r in results if r.success]
```

## Performance Optimization

### Recommended Settings by Use Case

```python
# High Quality (slow)
HIGH_QUALITY = {
    "steps": 50,
    "guidance": 7.5,
    "output_quality": 95,
    "preserve_details": True
}

# Balanced (default)
BALANCED = {
    "steps": 28,
    "guidance": 7.5, 
    "output_quality": 80,
    "preserve_details": True
}

# Fast Preview (quick)
FAST_PREVIEW = {
    "steps": 15,
    "guidance": 5.0,
    "output_quality": 70,
    "preserve_details": False
}
```

### Caching Strategy

```python
import hashlib
import json

def generate_cache_key(input_data: Dict[str, Any]) -> str:
    """Generate cache key for input parameters."""
    # Remove non-deterministic fields
    cacheable_data = {k: v for k, v in input_data.items() 
                     if k not in ["seed"]}
    
    return hashlib.md5(
        json.dumps(cacheable_data, sort_keys=True).encode()
    ).hexdigest()

# Use cache key to avoid regenerating identical requests
cache_key = generate_cache_key(input_params)
```

## Error Handling

```python
async def robust_flux_operation(tool, input_data: Dict[str, Any], max_retries: int = 3):
    """Robust operation with retry logic."""
    
    for attempt in range(max_retries):
        try:
            result = await tool.invoke(input_data)
            
            if result.success:
                return result
            else:
                print(f"Attempt {attempt + 1} failed: {result.error}")
                
        except Exception as e:
            print(f"Attempt {attempt + 1} error: {e}")
            
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    raise Exception("All retry attempts failed")
```

## Integration with Tool-First Architecture

```python
from pic_arcade_agentic.agents import ToolFirstAgent

# Initialize agent with Flux tools
agent = ToolFirstAgent({
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "replicate_api_token": os.getenv("REPLICATE_API_TOKEN")
})

# Dynamic tool selection for complex requests
result = await agent.process_request(
    "Transform this portrait to watercolor style, change the hair to blonde, and replace the background with a beach scene",
    context={"image_url": "portrait.jpg"}
)

# Agent will automatically:
# 1. Parse the request to identify needed operations
# 2. Plan the workflow (style transfer → object change → background swap)
# 3. Execute tools in optimal sequence
# 4. Return final result with metadata
```

## Best Practices

1. **Prompt Engineering**: Use detailed, specific prompts for better results
2. **Strength Tuning**: Start with 0.7 strength and adjust based on results
3. **Quality vs Speed**: Use appropriate settings for your use case
4. **Error Handling**: Implement retry logic for production systems
5. **Caching**: Cache results to avoid redundant API calls
6. **Batch Processing**: Process multiple images in parallel when possible

## Troubleshooting

### Common Issues

1. **API Token Missing**: Ensure `REPLICATE_API_TOKEN` is set
2. **Image Format**: Use supported formats (JPG, PNG, WebP)
3. **File Size**: Keep images under 10MB for best performance
4. **Timeout**: Large/complex operations may take up to 10 minutes
5. **Rate Limits**: Implement proper backoff for high-volume usage

### Performance Monitoring

```python
import time

async def monitored_operation(tool, input_data):
    """Monitor operation performance."""
    start_time = time.time()
    
    result = await tool.invoke(input_data)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Operation completed in {total_time:.2f}s")
    print(f"API processing time: {result.data.get('processing_time', 0):.2f}s") 
    print(f"Overhead: {total_time - result.data.get('processing_time', 0):.2f}s")
    
    return result
``` 