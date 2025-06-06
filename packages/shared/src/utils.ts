// Shared utilities for Pic Arcade platform

import { ApiResponse, SubscriptionPlan } from './types';
import { GENERATION_LIMITS } from './constants';

/**
 * Create a standardized API response
 */
export function createApiResponse<T>(
  success: boolean,
  data?: T,
  error?: string
): ApiResponse<T> {
  return {
    success,
    data,
    error,
    timestamp: new Date().toISOString(),
  };
}

/**
 * Generate a unique ID
 */
export function generateId(): string {
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
}

/**
 * Check if user has reached generation limits
 */
export function checkGenerationLimits(
  plan: SubscriptionPlan,
  currentUsage: { images: number; videos: number }
): { canGenerateImage: boolean; canGenerateVideo: boolean } {
  const limits = GENERATION_LIMITS[plan.toUpperCase() as keyof typeof GENERATION_LIMITS];
  
  return {
    canGenerateImage: currentUsage.images < limits.DAILY_IMAGES,
    canGenerateVideo: currentUsage.videos < limits.DAILY_VIDEOS,
  };
}

/**
 * Validate image resolution for plan
 */
export function validateResolution(
  plan: SubscriptionPlan,
  requestedResolution: string
): boolean {
  const limits = GENERATION_LIMITS[plan.toUpperCase() as keyof typeof GENERATION_LIMITS];
  const [maxWidth, maxHeight] = limits.MAX_RESOLUTION.split('x').map(Number);
  const [reqWidth, reqHeight] = requestedResolution.split('x').map(Number);
  
  return reqWidth <= maxWidth && reqHeight <= maxHeight;
}

/**
 * Format file size in human readable format
 */
export function formatFileSize(bytes: number): string {
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 Bytes';
  
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Validate email format
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Sleep utility for async operations
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
} 