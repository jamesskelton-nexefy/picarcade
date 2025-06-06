'use client'

import React, { useState, useEffect } from 'react'
import HistorySlider from '../components/HistorySlider'
import { saveAsset, Asset } from '../lib/supabase'

interface WorkflowStep {
  step: number;
  tool_name: string;
  description: string;
  inputs: Record<string, any>;
  expected_output: string;
  dependencies: string[];
}

interface WorkflowPlan {
  workflow_plan: WorkflowStep[];
  reasoning: string;
  confidence: number;
  estimated_time?: number;
  estimated_cost?: number;
}

interface ExecutionResult {
  step: number;
  tool_name: string;
  success: boolean;
  data?: Record<string, any>;
  error?: string;
  execution_time?: number;
  resolved_inputs?: Record<string, any>;
}

interface ExecutionData {
  execution_results: ExecutionResult[];
  final_outputs: Record<string, any>;
  execution_status: string;
  total_time: number;
  errors: string[];
}

interface WorkflowMetadata {
  tools_used: string[];
  total_time: number;
  execution_status: string;
}

interface ToolFirstResponse {
  success: boolean;
  request_id: string;
  user_request: string;
  workflow_plan?: WorkflowPlan;
  execution_results?: ExecutionData;
  metadata?: WorkflowMetadata;
  error?: string;
}

export default function HomePage() {
  const [prompt, setPrompt] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [workflowResult, setWorkflowResult] = useState<ToolFirstResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showHistory, setShowHistory] = useState(false)
  
  // Current image state for secondary prompts
  const [currentImages, setCurrentImages] = useState<{type: string; url: string; title: string; source: string; metadata?: any}[]>([])
  const [isSecondaryPrompt, setIsSecondaryPrompt] = useState(false)
  
  // Generate persistent user session for Mem0 memory (client-side only to avoid hydration issues)
  const [userId, setUserId] = useState<string>('')
  const [isClientReady, setIsClientReady] = useState(false)
  
  useEffect(() => {
    // Only run on client side to avoid hydration mismatch
    const initializeUserId = () => {
      // Try to get existing session from localStorage
      const existingUserId = localStorage.getItem('picarcade_user_id')
      
      if (existingUserId) {
        setUserId(existingUserId)
      } else {
        // Generate new session ID
        const timestamp = Date.now()
        const random = Math.random().toString(36).substr(2, 9)
        const newUserId = `user_${timestamp}_${random}`
        
        // Store in localStorage for persistence
        localStorage.setItem('picarcade_user_id', newUserId)
        setUserId(newUserId)
      }
      
      setIsClientReady(true)
    }
    
    initializeUserId()
  }, [])

  // Save generated assets to Supabase when workflow completes
  useEffect(() => {
    if (workflowResult?.success && workflowResult.execution_results) {
      saveGeneratedAssets()
      
      // Update current images for future secondary prompts
      const outputs = getGeneratedOutputs()
      const imageOutputs = outputs
        .filter(output => output.type === 'image')
        .map(output => ({
          type: output.type,
          url: output.url,
          title: output.title,
          source: output.source,
          metadata: output.metadata
        }))
      
      if (imageOutputs.length > 0) {
        setCurrentImages(imageOutputs)
      }
    }
  }, [workflowResult])

  const saveGeneratedAssets = async () => {
    if (!workflowResult?.execution_results) return
    
    const outputs = getGeneratedOutputs()
    
    for (const output of outputs) {
      try {
        await saveAsset({
          title: output.title,
          description: `Generated via ${output.source}`,
          type: output.type as 'image' | 'video',
          url: output.url,
          thumbnail_url: output.url, // For now, use the same URL as thumbnail
          source_tool: output.source,
          prompt: workflowResult.user_request,
          metadata: output.metadata || {}
        })
      } catch (err) {
        console.error('Error saving asset:', err)
      }
    }
  }

  const handleAssetSelect = (asset: Asset) => {
    // When user selects an asset from history, show it in a new tab
    window.open(asset.url, '_blank')
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!prompt.trim() || !isClientReady) return
    
    // Detect if this might be a secondary prompt (we have current images and prompt suggests editing)
    const isLikelySecondaryPrompt = currentImages.length > 0 && (
      /\b(edit|modify|change|add|remove|replace|adjust|update|improve|enhance|fix)\b/i.test(prompt) ||
      /\b(make it|turn it|convert|transform)\b/i.test(prompt) ||
      /\b(with|without|instead of|but|however)\b/i.test(prompt)
    )
    
    setIsProcessing(true)
    setError(null)
    setIsSecondaryPrompt(isLikelySecondaryPrompt)
    
    // Only clear results if this is not a secondary prompt
    if (!isLikelySecondaryPrompt) {
      setWorkflowResult(null)
      setCurrentImages([])
    }
    
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/workflow/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt.trim(),
          user_id: userId || 'anonymous'
        })
      })
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`)
      }
      
      const result: ToolFirstResponse = await response.json()
      setWorkflowResult(result)
      setPrompt('') // Clear the input after successful submission
      
    } catch (err) {
      console.error('Workflow error:', err)
      setError(err instanceof Error ? err.message : 'Failed to process prompt')
    } finally {
      setIsProcessing(false)
      setIsSecondaryPrompt(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as any)
    }
  }

  const formatTime = (seconds?: number) => {
    if (!seconds) return 'N/A'
    return seconds < 1 ? `${Math.round(seconds * 1000)}ms` : `${seconds.toFixed(1)}s`
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-900 text-green-300'
      case 'completed_with_errors': return 'bg-yellow-900 text-yellow-300'
      case 'failed': return 'bg-red-900 text-red-300'
      case 'processing': return 'bg-blue-900 text-blue-300'
      default: return 'bg-gray-900 text-gray-300'
    }
  }

  const renderOutputValue = (value: any): string => {
    if (typeof value === 'string') return value
    if (typeof value === 'object' && value !== null) {
      return JSON.stringify(value, null, 2)
    }
    return String(value)
  }

  // Extract generated images from workflow results
  const getGeneratedOutputs = () => {
    if (!workflowResult?.execution_results) return []
    
    const outputs: { type: string; url: string; title: string; source: string; metadata?: any }[] = []
    
    workflowResult.execution_results.execution_results.forEach((result) => {
      if (result.success && result.data) {
        // Check for images in various formats
        if (result.data.images && Array.isArray(result.data.images)) {
          result.data.images.forEach((img: any, idx: number) => {
            if (typeof img === 'string') {
              outputs.push({
                type: 'image',
                url: img,
                title: `Generated Image ${idx + 1}`,
                source: result.tool_name
              })
            } else if (img.url) {
              outputs.push({
                type: 'image',
                url: img.url,
                title: img.title || `Generated Image ${idx + 1}`,
                source: result.tool_name,
                metadata: img
              })
            }
          })
        }
        
        // Check for single image outputs
        if (result.data.styled_image) {
          outputs.push({
            type: 'image',
            url: result.data.styled_image,
            title: 'Styled Image',
            source: result.tool_name
          })
        }
        
        if (result.data.modified_image) {
          outputs.push({
            type: 'image',
            url: result.data.modified_image,
            title: 'Modified Image',
            source: result.tool_name
          })
        }
        
        if (result.data.edited_image) {
          outputs.push({
            type: 'image',
            url: result.data.edited_image,
            title: 'Edited Image',
            source: result.tool_name
          })
        }
        
        if (result.data.swapped_image) {
          outputs.push({
            type: 'image',
            url: result.data.swapped_image,
            title: 'Background Swapped Image',
            source: result.tool_name
          })
        }
        
        if (result.data.consistent_image) {
          outputs.push({
            type: 'image',
            url: result.data.consistent_image,
            title: 'Character Consistent Image',
            source: result.tool_name
          })
        }
        
        // Check for video outputs
        if (result.data.video_url) {
          outputs.push({
            type: 'video',
            url: result.data.video_url,
            title: 'Generated Video',
            source: result.tool_name,
            metadata: {
              provider: result.data.provider_used,
              model: result.data.model_used,
              duration: result.data.duration,
              resolution: result.data.resolution,
              fps: result.data.fps
            }
          })
        }
        
        // Check for videos array (if multiple videos are returned)
        if (result.data.videos && Array.isArray(result.data.videos)) {
          result.data.videos.forEach((video: any, idx: number) => {
            if (typeof video === 'string') {
              outputs.push({
                type: 'video',
                url: video,
                title: `Generated Video ${idx + 1}`,
                source: result.tool_name
              })
            } else if (video.url) {
              outputs.push({
                type: 'video',
                url: video.url,
                title: video.title || `Generated Video ${idx + 1}`,
                source: result.tool_name,
                metadata: video
              })
            }
          })
        }
      }
    })
    
    return outputs
  }

  const generatedOutputs = getGeneratedOutputs()

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Main Content Area */}
      <div className="flex flex-col h-screen">
        {/* Header/Content Area */}
        <div className="flex-1 p-6 max-w-6xl mx-auto w-full">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-green-400 to-green-600 rounded-lg flex items-center justify-center">
                <span className="text-2xl">🎮</span>
              </div>
              <div>
                <h1 className="text-2xl font-semibold">Picarcade</h1>
                <p className="text-sm text-gray-400">Tool-First AI Agent</p>
              </div>
              
              {/* Session Indicator */}
              <div className="ml-auto flex items-center gap-2 text-xs text-gray-500">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Memory Active</span>
                <span className="font-mono text-gray-600">
                  {isClientReady && userId ? userId.slice(-8) : '•••••••'}
                </span>
              </div>
            </div>
          </div>

          {/* Main Output Area */}
          {(generatedOutputs.length > 0 || (isSecondaryPrompt && currentImages.length > 0)) ? (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 mb-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-green-400">
                  {isSecondaryPrompt ? 'Current Image' : 'Generated Content'}
                  {isProcessing && isSecondaryPrompt && (
                    <span className="ml-2 text-sm text-yellow-400 animate-pulse">
                      • Processing edit...
                    </span>
                  )}
                </h2>
                <div className="flex items-center gap-2 text-sm text-gray-400">
                  {isSecondaryPrompt ? (
                    <>
                      <span>{currentImages.length} current image{currentImages.length !== 1 ? 's' : ''}</span>
                      {isProcessing && (
                        <>
                          <span>•</span>
                          <span className="text-yellow-400">Editing...</span>
                        </>
                      )}
                    </>
                  ) : (
                    <>
                      <span>{generatedOutputs.length} output{generatedOutputs.length !== 1 ? 's' : ''}</span>
                      <span>•</span>
                      <span>{workflowResult?.metadata?.tools_used.join(', ')}</span>
                    </>
                  )}
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {(isSecondaryPrompt && generatedOutputs.length === 0 ? currentImages : generatedOutputs).map((output, idx) => (
                  <div key={idx} className="group relative">
                    <div className="aspect-square bg-gray-750 rounded-lg overflow-hidden border border-gray-600 hover:border-gray-500 transition-colors">
                      {/* Processing overlay for secondary prompts */}
                      {isSecondaryPrompt && isProcessing && generatedOutputs.length === 0 && (
                        <div className="absolute inset-0 bg-black bg-opacity-60 backdrop-blur-sm flex items-center justify-center z-10">
                          <div className="text-center">
                            <svg className="w-8 h-8 text-yellow-400 animate-spin mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                            <p className="text-xs text-yellow-400">Editing...</p>
                          </div>
                        </div>
                      )}
                      {output.type === 'image' ? (
                        <>
                          <img 
                            src={output.url} 
                            alt={output.title}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                            onError={(e) => {
                              e.currentTarget.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBzdHJva2U9IiM2Mzc1OEIiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo='
                            }}
                          />
                          {/* Image overlay with download button */}
                          <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-50 transition-opacity duration-300 flex items-center justify-center opacity-0 group-hover:opacity-100">
                            <div className="flex gap-2">
                              <button 
                                onClick={() => window.open(output.url, '_blank')}
                                className="bg-white bg-opacity-20 hover:bg-opacity-30 backdrop-blur-sm rounded-lg p-2 transition-colors"
                                title="View full size"
                              >
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                </svg>
                              </button>
                              <a 
                                href={output.url}
                                download={`${output.title.replace(/\s+/g, '_')}.jpg`}
                                className="bg-white bg-opacity-20 hover:bg-opacity-30 backdrop-blur-sm rounded-lg p-2 transition-colors"
                                title="Download image"
                              >
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                              </a>
                            </div>
                          </div>
                        </>
                      ) : output.type === 'video' ? (
                        <>
                          <video 
                            src={output.url}
                            className="w-full h-full object-cover"
                            controls
                            muted
                            preload="metadata"
                            onError={(e) => {
                              console.error('Video failed to load:', output.url)
                            }}
                          />
                          {/* Video overlay with download button */}
                          <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-50 transition-opacity duration-300 flex items-center justify-center opacity-0 group-hover:opacity-100 pointer-events-none">
                            <div className="flex gap-2 pointer-events-auto">
                              <button 
                                onClick={() => window.open(output.url, '_blank')}
                                className="bg-white bg-opacity-20 hover:bg-opacity-30 backdrop-blur-sm rounded-lg p-2 transition-colors"
                                title="View full size"
                              >
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                </svg>
                              </button>
                              <a 
                                href={output.url}
                                download={`${output.title.replace(/\s+/g, '_')}.mp4`}
                                className="bg-white bg-opacity-20 hover:bg-opacity-30 backdrop-blur-sm rounded-lg p-2 transition-colors"
                                title="Download video"
                              >
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                              </a>
                            </div>
                          </div>
                        </>
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-gray-400">
                          <span>Output: {output.type}</span>
                        </div>
                      )}
                    </div>
                    
                    <div className="mt-3">
                      <h3 className="text-sm font-medium text-gray-200 truncate">{output.title}</h3>
                      <p className="text-xs text-gray-400 mt-1">via {output.source}</p>
                      {output.metadata && (
                        <div className="text-xs text-gray-500 mt-1">
                          {output.type === 'image' ? (
                            output.metadata.width && output.metadata.height && (
                              <span>{output.metadata.width}×{output.metadata.height}</span>
                            )
                          ) : output.type === 'video' ? (
                            <div className="space-y-1">
                              {output.metadata.duration && (
                                <span>{output.metadata.duration}s</span>
                              )}
                              {output.metadata.resolution && (
                                <span className="ml-2">{output.metadata.resolution}</span>
                              )}
                              {output.metadata.fps && (
                                <span className="ml-2">{output.metadata.fps}fps</span>
                              )}
                              {output.metadata.provider && (
                                <div className="text-purple-400">via {output.metadata.provider}</div>
                              )}
                            </div>
                          ) : null}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              
              {/* Quick Actions */}
              <div className="mt-6 pt-6 border-t border-gray-700">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-gray-400">
                    {isSecondaryPrompt && generatedOutputs.length === 0 ? (
                      isProcessing ? 'Processing edit...' : 'Ready for editing'
                    ) : (
                      `Generated in ${formatTime(workflowResult?.metadata?.total_time)}`
                    )}
                  </p>
                  <div className="flex gap-3">
                    <button 
                      onClick={() => {
                        setWorkflowResult(null)
                        setCurrentImages([])
                        setIsSecondaryPrompt(false)
                      }}
                      className="text-sm text-gray-400 hover:text-white transition-colors"
                    >
                      Clear Results
                    </button>
                    <button 
                      onClick={() => {
                        const currentDisplayedImages = isSecondaryPrompt && generatedOutputs.length === 0 ? currentImages : generatedOutputs
                        const urls = currentDisplayedImages.map(o => o.url).join('\n')
                        navigator.clipboard.writeText(urls)
                      }}
                      className="text-sm bg-purple-600 hover:bg-purple-700 px-3 py-1 rounded transition-colors"
                    >
                      Copy URLs
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Upload Card - shown when no results */
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 mb-6">
              <div className="flex flex-col items-center justify-center text-center">
                <button className="w-24 h-24 border-2 border-dashed border-gray-600 rounded-lg flex items-center justify-center hover:border-gray-500 transition-colors mb-4">
                  <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                </button>
                <p className="text-sm text-gray-400 mb-2">Upload files or generate with AI</p>
                <p className="text-xs text-gray-500">Try: "Generate a portrait of Emma Stone" or "Create a cyberpunk cityscape"</p>
              </div>
            </div>
          )}

          {/* Tool-First Workflow Results */}
          {workflowResult && (
            <div className="space-y-6 mb-6">
              {/* Overview Card */}
              <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-medium text-green-400">AI Agent Workflow</h3>
                  <span className={`px-2 py-1 rounded text-xs ${workflowResult.success ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
                    {workflowResult.success ? 'Success' : 'Failed'}
                  </span>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Request ID:</span>
                    <p className="text-purple-300 font-mono text-xs">{workflowResult.request_id}</p>
                  </div>
                  <div>
                    <span className="text-gray-400">Tools Used:</span>
                    <p className="text-blue-300">{workflowResult.metadata?.tools_used.length || 0}</p>
                  </div>
                  <div>
                    <span className="text-gray-400">Total Time:</span>
                    <p className="text-yellow-300">{formatTime(workflowResult.metadata?.total_time)}</p>
                  </div>
                  <div>
                    <span className="text-gray-400">Status:</span>
                    <span className={`ml-2 px-2 py-1 rounded text-xs ${getStatusColor(workflowResult.metadata?.execution_status || 'unknown')}`}>
                      {workflowResult.metadata?.execution_status || 'Unknown'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Workflow Plan */}
              {workflowResult.workflow_plan && (
                <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
                  <h4 className="text-md font-medium mb-4 text-blue-400">Workflow Plan</h4>
                  
                  <div className="space-y-4 mb-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-gray-400">Confidence:</span>
                        <span className="ml-2">{(workflowResult.workflow_plan.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Steps:</span>
                        <span className="ml-2">{workflowResult.workflow_plan.workflow_plan.length}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Est. Time:</span>
                        <span className="ml-2">{formatTime(workflowResult.workflow_plan.estimated_time)}</span>
                      </div>
                    </div>
                    
                    {workflowResult.workflow_plan.reasoning && (
                      <div>
                        <span className="text-gray-400">Reasoning:</span>
                        <p className="text-sm mt-1 text-gray-300">{workflowResult.workflow_plan.reasoning}</p>
                      </div>
                    )}
                  </div>

                  <div className="space-y-3">
                    {workflowResult.workflow_plan.workflow_plan.map((step, idx) => (
                      <div key={idx} className="bg-gray-750 p-4 rounded border-l-4 border-blue-500">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-blue-300">
                            Step {step.step}: {step.tool_name}
                          </span>
                          <span className="text-xs text-gray-400">
                            {step.dependencies.length > 0 && `Depends on: ${step.dependencies.join(', ')}`}
                          </span>
                        </div>
                        <p className="text-sm text-gray-300 mb-2">{step.description}</p>
                        <div className="text-xs text-gray-400">
                          Expected: {step.expected_output}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Execution Results */}
              {workflowResult.execution_results && (
                <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-md font-medium text-yellow-400">Execution Results</h4>
                    <div className="flex gap-4 text-sm">
                      <span className="text-gray-400">
                        Success Rate: <span className="text-green-300">
                          {Math.round((workflowResult.execution_results.execution_results.filter(r => r.success).length / 
                            workflowResult.execution_results.execution_results.length) * 100)}%
                        </span>
                      </span>
                      <span className="text-gray-400">
                        Errors: <span className="text-red-300">{workflowResult.execution_results.errors.length}</span>
                      </span>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {workflowResult.execution_results.execution_results.map((result, idx) => (
                      <div key={idx} className={`bg-gray-750 p-4 rounded border-l-4 ${
                        result.success ? 'border-green-500' : 'border-red-500'
                      }`}>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">
                            Step {result.step}: {result.tool_name}
                          </span>
                          <div className="flex items-center gap-2">
                            <span className={`px-2 py-1 rounded text-xs ${
                              result.success ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                            }`}>
                              {result.success ? 'Success' : 'Failed'}
                            </span>
                            {result.execution_time && (
                              <span className="text-xs text-gray-400">{formatTime(result.execution_time)}</span>
                            )}
                          </div>
                        </div>
                        
                        {result.error && (
                          <div className="text-red-300 text-sm mb-2">
                            Error: {result.error}
                          </div>
                        )}
                        
                        {result.data && (
                          <div className="text-sm">
                            <div className="text-gray-400 mb-1">Output:</div>
                            <div className="bg-gray-900 p-2 rounded font-mono text-xs overflow-x-auto max-h-32">
                              {Object.entries(result.data).map(([key, value]) => (
                                <div key={key} className="mb-1">
                                  <span className="text-blue-300">{key}:</span>{' '}
                                  <span className="text-gray-300">{renderOutputValue(value)}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>

                  {workflowResult.execution_results.errors.length > 0 && (
                    <div className="mt-4 p-4 bg-red-900/20 border border-red-700 rounded">
                      <h5 className="text-red-300 font-medium mb-2">Workflow Errors:</h5>
                      <ul className="text-sm text-red-200 space-y-1">
                        {workflowResult.execution_results.errors.map((error, idx) => (
                          <li key={idx}>• {error}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="bg-red-900 border border-red-700 rounded-lg p-4 mb-6">
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-red-300 text-sm">{error}</span>
              </div>
            </div>
          )}
        </div>

        {/* Chat Input Area */}
        <div className="border-t border-gray-700 p-4">
          <div className="max-w-6xl mx-auto">
            <form onSubmit={handleSubmit} className="flex items-center gap-3">
              {/* Attachment Icon */}
              <button type="button" className="text-gray-400 hover:text-gray-300">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                </svg>
              </button>
              
              {/* History Icon */}
              <button 
                type="button" 
                onClick={() => setShowHistory(true)}
                className="text-gray-400 hover:text-purple-400 transition-colors"
                title="View creation history"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </button>
              
              {/* Message Input */}
              <div className="flex-1 relative">
                {/* Current images indicator */}
                {currentImages.length > 0 && !isProcessing && (
                  <div className="absolute left-3 top-1/2 transform -translate-y-1/2 flex items-center gap-1 text-xs text-purple-400 z-10">
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span>{currentImages.length}</span>
                  </div>
                )}
                <input
                  type="text"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={
                    !isClientReady ? "Initializing..." : 
                    isProcessing ? "Processing..." : 
                    currentImages.length > 0 ? "Describe your edit (e.g., 'add a hat', 'change background to forest')..." :
                    "Ask the AI agent anything..."
                  }
                  disabled={isProcessing || !isClientReady}
                  className={`w-full bg-gray-800 border border-gray-600 rounded-lg py-3 pr-12 text-white placeholder-gray-400 focus:outline-none focus:border-purple-500 disabled:opacity-50 ${
                    currentImages.length > 0 ? 'pl-12' : 'px-4'
                  }`}
                />
                
                {/* Send Button */}
                <button 
                  type="submit"
                  disabled={isProcessing || !prompt.trim() || !isClientReady}
                  className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isProcessing ? (
                    <svg className="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>

      {/* History Slider Modal */}
      <HistorySlider 
        isVisible={showHistory}
        onClose={() => setShowHistory(false)}
        onAssetSelect={handleAssetSelect}
      />
    </div>
  )
} 