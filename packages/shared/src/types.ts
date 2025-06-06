// Shared types for Pic Arcade platform

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  environment: string;
  services: Record<string, string>;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

// User and Authentication Types
export interface User {
  id: string;
  email: string;
  username: string;
  fullName?: string;
  avatar?: string;
  plan: SubscriptionPlan;
  createdAt: string;
  updatedAt: string;
}

export enum SubscriptionPlan {
  FREE = 'free',
  PLUS = 'plus',
  PRO = 'pro'
}

// Generation Request Types
export interface GenerationRequest {
  id: string;
  userId: string;
  prompt: string;
  type: GenerationType;
  parameters: GenerationParameters;
  status: GenerationStatus;
  createdAt: string;
  updatedAt: string;
}

export enum GenerationType {
  IMAGE = 'image',
  VIDEO = 'video',
  FACE_SWAP = 'face_swap',
  STYLE_TRANSFER = 'style_transfer',
  IMAGE_EDIT = 'image_edit',
  VIDEO_EDIT = 'video_edit'
}

export enum GenerationStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export interface GenerationParameters {
  model?: string;
  style?: string;
  quality?: 'low' | 'medium' | 'high' | 'ultra';
  aspectRatio?: string;
  seed?: number;
  steps?: number;
  guidance?: number;
  negativePrompt?: string;
  referenceImages?: string[];
}

// Prompt Parsing Types (Phase 2)
export interface ParsedPrompt {
  intent: string;
  entities: PromptEntity[];
  modifiers: PromptModifier[];
  references: PromptReference[];
  confidence: number;
}

export interface PromptEntity {
  text: string;
  type: 'person' | 'object' | 'style' | 'action' | 'setting';
  confidence: number;
}

export interface PromptModifier {
  text: string;
  type: 'quality' | 'style' | 'lighting' | 'mood' | 'technical';
  value?: string;
}

export interface PromptReference {
  text: string;
  type: 'celebrity' | 'artwork' | 'style' | 'brand';
  searchQuery: string;
  imageUrls?: string[];
}

// Image and Video Generation Results
export interface GenerationResult {
  id: string;
  requestId: string;
  type: GenerationType;
  status: GenerationStatus;
  outputs: GenerationOutput[];
  metadata: GenerationMetadata;
  createdAt: string;
}

export interface GenerationOutput {
  id: string;
  type: 'image' | 'video';
  url: string;
  thumbnailUrl?: string;
  width: number;
  height: number;
  duration?: number; // for videos
  fileSize: number;
  format: string;
  qualityScore?: number;
}

export interface GenerationMetadata {
  model: string;
  parameters: GenerationParameters;
  processingTime: number;
  cost: number;
  qualityMetrics?: QualityMetrics;
}

export interface QualityMetrics {
  clipScore?: number;
  aestheticScore?: number;
  artifactScore?: number;
  faceQuality?: number;
  overallScore: number;
}

// Agentic Workflow Types
export interface WorkflowState {
  requestId: string;
  currentStep: WorkflowStep;
  steps: WorkflowStep[];
  context: WorkflowContext;
  status: GenerationStatus;
}

export enum WorkflowStep {
  PARSE_PROMPT = 'parse_prompt',
  RETRIEVE_REFERENCES = 'retrieve_references',
  GENERATE_CONTENT = 'generate_content',
  EDIT_CONTENT = 'edit_content',
  QUALITY_CHECK = 'quality_check',
  FINALIZE = 'finalize'
}

export interface WorkflowContext {
  prompt: ParsedPrompt;
  references: PromptReference[];
  generatedOutputs: GenerationOutput[];
  qualityScores: QualityMetrics[];
  retryCount: number;
}

// API Configuration
export interface ApiConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
  apiKey?: string;
} 