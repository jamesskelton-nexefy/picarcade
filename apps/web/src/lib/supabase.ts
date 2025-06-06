// Mock Supabase implementation for development
// This will be replaced with actual Supabase once dependencies are resolved

export interface Asset {
  id: string
  title: string
  description?: string
  type: 'image' | 'video'
  url: string
  thumbnail_url?: string
  source_tool?: string
  prompt?: string
  metadata?: Record<string, any>
  created_at: string
  updated_at: string
}

// Mock data for development
let mockAssets: Asset[] = [
  {
    id: '1',
    title: 'Sample Generated Image',
    description: 'Generated via replicate',
    type: 'image',
    url: 'https://picsum.photos/400/400?random=1',
    thumbnail_url: 'https://picsum.photos/200/200?random=1',
    source_tool: 'flux-dev',
    prompt: 'A beautiful landscape with mountains',
    metadata: { width: 400, height: 400 },
    created_at: new Date(Date.now() - 1000 * 60 * 30).toISOString(), // 30 minutes ago
    updated_at: new Date(Date.now() - 1000 * 60 * 30).toISOString()
  },
  {
    id: '2',
    title: 'AI Portrait',
    description: 'Generated via runwayml',
    type: 'image',
    url: 'https://picsum.photos/400/400?random=2',
    thumbnail_url: 'https://picsum.photos/200/200?random=2',
    source_tool: 'runwayml',
    prompt: 'Portrait of a person with artistic lighting',
    metadata: { width: 400, height: 400 },
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(), // 2 hours ago
    updated_at: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString()
  },
  {
    id: '3',
    title: 'Abstract Art',
    description: 'Generated via stable-diffusion',
    type: 'image',
    url: 'https://picsum.photos/400/400?random=3',
    thumbnail_url: 'https://picsum.photos/200/200?random=3',
    source_tool: 'stable-diffusion',
    prompt: 'Abstract colorful geometric patterns',
    metadata: { width: 400, height: 400 },
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(), // 1 day ago
    updated_at: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString()
  }
]

// Check if we're in a real Supabase environment
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
const isSupabaseConfigured = supabaseUrl && supabaseAnonKey

// Mock supabase object for development
export const supabase = {
  from: () => ({
    insert: () => ({ select: () => ({ single: () => Promise.resolve({ data: null, error: null }) }) }),
    select: () => ({ order: () => ({ limit: () => Promise.resolve({ data: mockAssets, error: null }) }) }),
    delete: () => ({ eq: () => Promise.resolve({ error: null }) })
  })
}

// Save a new asset to the database (mock implementation)
export async function saveAsset(asset: Omit<Asset, 'id' | 'created_at' | 'updated_at'>) {
  console.log('Mock: Saving asset:', asset.title)
  
  const newAsset: Asset = {
    ...asset,
    id: Date.now().toString(),
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  }
  
  // Add to mock data
  mockAssets.unshift(newAsset)
  
  // Simulate async operation
  await new Promise(resolve => setTimeout(resolve, 100))
  
  return newAsset
}

// Get recent assets for the history slider (mock implementation)
export async function getRecentAssets(limit: number = 20): Promise<Asset[]> {
  console.log('Mock: Fetching recent assets, limit:', limit)
  
  // Simulate async operation
  await new Promise(resolve => setTimeout(resolve, 200))
  
  return mockAssets.slice(0, limit)
}

// Delete an asset (mock implementation)
export async function deleteAsset(id: string) {
  console.log('Mock: Deleting asset:', id)
  
  // Remove from mock data
  mockAssets = mockAssets.filter(asset => asset.id !== id)
  
  // Simulate async operation
  await new Promise(resolve => setTimeout(resolve, 100))
} 