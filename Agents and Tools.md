# Agents and Tools

This document provides a comprehensive overview of all agents and tools available in the PicArcade system, including their purposes and the AI models they utilize.

## Agents

### ToolFirstAgent
- **Purpose**: Main agent using tool-first architecture for dynamic capability selection. Analyzes user requests, discovers relevant tools, plans multi-step workflows, and executes tool chains dynamically while maintaining conversation context for multi-turn interactions.
- **Location**: `packages/agentic/src/pic_arcade_agentic/agents/tool_agent.py`

### Mem0ToolFirstAgent  
- **Purpose**: Enhanced tool-first agent with Mem0 persistent memory for multi-session context. Provides persistent conversation context across API requests, intelligent memory management, multi-turn image editing with proper context, user preference learning and adaptation.
- **Location**: `packages/agentic/src/pic_arcade_agentic/agents/mem0_tool_agent.py`

### PromptParsingAgent
- **Purpose**: Agent responsible for parsing user prompts using GPT-4. Extracts structured information for downstream processing in the agentic workflow pipeline.
- **Location**: `packages/agentic/src/pic_arcade_agentic/agents/prompt_parser.py`

### ReferenceRetrievalAgent
- **Purpose**: Agent for retrieving reference images and content based on parsed prompt entities.
- **Location**: `packages/agentic/src/pic_arcade_agentic/agents/reference_retriever.py`

### WorkflowOrchestrator
- **Purpose**: LangGraph-based orchestrator for the agentic pipeline. Manages workflow state and coordinates between different agents for prompt parsing and reference retrieval.
- **Location**: `packages/agentic/src/pic_arcade_agentic/workflow/orchestrator.py`

## Tools

### Prompt Processing Tools

#### PromptParsingTool
- **Model**: GPT-4o (OpenAI)
- **Purpose**: Parse user prompts to extract intent, entities, modifiers, and references using structured JSON output
- **Category**: prompt_processing

#### PromptOptimizationTool
- **Model**: GPT-4o (OpenAI)
- **Purpose**: Optimize user prompts for better AI image/video generation results by adding technical terms and improving structure
- **Category**: prompt_processing

### Search Tools

#### PerplexitySearchTool
- **Model**: llama-3.1-sonar-large-128k-online (Perplexity)
- **Purpose**: Search for images and information using Perplexity API with intelligent analysis and query optimization
- **Category**: image_search

#### WebSearchTool
- **Model**: llama-3.1-sonar-large-128k-online (Perplexity)
- **Purpose**: Alias for PerplexitySearchTool to maintain compatibility for web searches
- **Category**: image_search

#### BingImageSearchTool (Legacy)
- **Model**: llama-3.1-sonar-large-128k-online (Perplexity)
- **Purpose**: Legacy compatibility tool that redirects to Perplexity API
- **Category**: image_search

#### GoogleImageSearchTool (Legacy)
- **Model**: llama-3.1-sonar-large-128k-online (Perplexity)
- **Purpose**: Legacy compatibility tool that redirects to Perplexity API
- **Category**: image_search

### Image Generation Tools

#### FluxImageManipulationTool
- **Model**: black-forest-labs/flux-kontext-max (Replicate)
- **Purpose**: Advanced image editing and manipulation - ONLY uses flux-kontext-max for ALL editing operations like style transfer, object changes, text editing, background swapping, and character consistency
- **Category**: image_editing

#### FluxImageGenerationTool
- **Model**: black-forest-labs/flux-1.1-pro-ultra (Replicate)
- **Purpose**: Dedicated tool for pure image generation from text prompts - ONLY uses flux-1.1-pro-ultra for creating new images
- **Category**: image_generation

#### StableDiffusionImageTool
- **Model**: stability-ai/stable-diffusion-xl-base-1.0 (Replicate)
- **Purpose**: High-quality image generation using Stable Diffusion XL with advanced parameter control
- **Category**: image_generation

#### DALLEImageGenerationTool
- **Model**: dall-e-3 (OpenAI)
- **Purpose**: Create high-quality images using OpenAI's DALL-E 3 model with natural language prompts
- **Category**: image_generation

### Image Editing Tools

#### StyleTransferTool
- **Model**: black-forest-labs/flux-kontext-max (Replicate)
- **Purpose**: Convert photos to different art styles (watercolor, oil painting, sketches, etc.) while preserving subject and composition
- **Category**: image_editing

#### ObjectChangeTool
- **Model**: black-forest-labs/flux-kontext-max (Replicate)
- **Purpose**: Modify specific objects in images (hair, clothing, accessories, colors) while maintaining natural integration
- **Category**: image_editing

#### TextEditingTool
- **Model**: black-forest-labs/flux-kontext-max (Replicate)
- **Purpose**: Replace text in signs, posters, labels, and other text elements while maintaining typography style
- **Category**: image_editing

#### BackgroundSwapTool
- **Model**: black-forest-labs/flux-kontext-max (Replicate)
- **Purpose**: Change image backgrounds while preserving subjects and maintaining proper lighting integration
- **Category**: image_editing

#### CharacterConsistencyTool
- **Model**: black-forest-labs/flux-kontext-max (Replicate)
- **Purpose**: Maintain character identity across different poses, expressions, and edits for consistent visual storytelling
- **Category**: image_editing

#### ImageEditingTool
- **Model**: black-forest-labs/flux-1-fill-pro (Replicate)
- **Purpose**: Advanced image editing with inpainting, outpainting, background removal, and object replacement
- **Category**: image_editing

### Face Manipulation Tools

#### FaceSwapTool
- **Model**: Multiple models via Replicate (omniedgeio, instantid, become-image)
- **Purpose**: High-quality face swapping with face enhancement and identity preservation
- **Category**: face_manipulation

### Quality Assessment Tools

#### QualityAssessmentTool
- **Model**: Internal quality metrics + OpenAI CLIP models
- **Purpose**: Comprehensive image quality analysis including aesthetic scores, technical quality, artifact detection, and improvement recommendations
- **Category**: quality_assessment

### Video Generation Tools

#### RunwayVideoTool
- **Model**: runwayml/gen-3-alpha-turbo (Replicate)
- **Purpose**: Generate high-quality videos from text prompts using Runway's Gen-3 model with advanced motion control
- **Category**: video_generation

#### ReplicateVideoTool
- **Model**: Multiple video models via Replicate
- **Purpose**: Flexible video generation using various Replicate video models with dynamic model selection
- **Category**: video_generation

#### VideoEditingTool
- **Model**: Multiple video editing models via Replicate
- **Purpose**: Comprehensive video editing including upscaling, style transfer, enhancement, stabilization, and motion editing
- **Category**: video_generation

### Workflow Tools

#### WorkflowPlanningTool
- **Model**: GPT-4o (OpenAI)
- **Purpose**: Plan multi-step workflows by selecting and sequencing appropriate tools with conversation context awareness
- **Category**: workflow_planning

#### WorkflowExecutorTool
- **Model**: N/A (Orchestration)
- **Purpose**: Execute planned workflows by coordinating tool execution, managing state, and handling errors. Features intelligent entity combination to construct complete prompts from parsed data (e.g., "cat at party" instead of just "cat")
- **Category**: workflow_planning

## Tool Categories

- **prompt_processing**: Tools for analyzing and optimizing user prompts
- **image_search**: Tools for finding reference images and web content
- **image_generation**: Tools for creating new images from prompts
- **image_editing**: Tools for modifying existing images
- **video_generation**: Tools for creating and editing videos
- **face_manipulation**: Tools for face-related modifications
- **quality_assessment**: Tools for evaluating content quality
- **workflow_planning**: Tools for orchestrating multi-step operations

## Architecture Overview

The PicArcade system follows a **tool-first architecture** where:

1. **Agents** act as coordinators that analyze requests and select appropriate tools
2. **Tools** wrap external AI models (OpenAI, Replicate, Perplexity) with standardized interfaces
3. **Workflow tools** enable complex multi-step operations by chaining multiple tools together
4. **Memory systems** (Mem0) provide persistent context across sessions for enhanced user experience
5. **Dynamic model selection** optimizes performance by using the best model for each task type

### Model Selection Strategy

The system intelligently selects models based on operation type:
- **Pure generation** (creating images from scratch): Uses `flux-1.1-pro-ultra` for optimal generation quality
- **Image editing** (style transfer, object changes, etc.): Uses `flux-kontext-max` for superior editing capabilities

This modular design allows for easy extension with new tools and models while maintaining consistent interfaces throughout the system. 