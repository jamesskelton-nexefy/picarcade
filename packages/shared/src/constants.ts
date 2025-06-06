// Shared constants for Pic Arcade platform

export const API_VERSION = '0.1.0';

export const DEFAULT_TIMEOUTS = {
  API_REQUEST: 30000,
  IMAGE_GENERATION: 120000,
  VIDEO_GENERATION: 300000,
  FACE_SWAP: 60000,
} as const;

export const GENERATION_LIMITS = {
  FREE: {
    DAILY_IMAGES: 10,
    DAILY_VIDEOS: 2,
    MAX_RESOLUTION: '1024x1024',
  },
  PLUS: {
    DAILY_IMAGES: 100,
    DAILY_VIDEOS: 20,
    MAX_RESOLUTION: '2048x2048',
  },
  PRO: {
    DAILY_IMAGES: 1000,
    DAILY_VIDEOS: 100,
    MAX_RESOLUTION: '4096x4096',
  },
} as const;

export const SUPPORTED_FORMATS = {
  IMAGE: ['png', 'jpg', 'jpeg', 'webp'],
  VIDEO: ['mp4', 'webm', 'mov'],
} as const;

export const QUALITY_THRESHOLDS = {
  MIN_CLIP_SCORE: 0.25,
  MIN_AESTHETIC_SCORE: 0.5,
  MAX_ARTIFACT_SCORE: 0.3,
  MIN_OVERALL_SCORE: 0.6,
} as const; 