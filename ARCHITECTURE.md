# PicArcade Architecture

PicArcade is a **mobile-first** professional picture and video editor with advanced AI/ML capabilities. It's built as a modern monorepo with a hybrid tech stack.

## 🏗️ Project Structure

```
pic-arcade/
├── apps/                          # Frontend Applications
│   ├── web/                       # Next.js Web App
│   └── mobile/                    # React Native Mobile App
├── packages/                      # Backend & Shared Packages
│   ├── api/                       # Python FastAPI Backend
│   ├── agentic/                   # Python AI/ML Processing
│   └── shared/                    # TypeScript Shared Utilities
└── docs/                          # Documentation
```

## 📱 Frontend Applications

### Web App (`apps/web`)
- **Framework**: Next.js 14 with App Router
- **UI**: React 18 + TypeScript + Tailwind CSS
- **Database**: Supabase (PostgreSQL + Auth + Storage)
- **Purpose**: Professional web interface for desktop users

### Mobile App (`apps/mobile`)
- **Framework**: React Native 0.73
- **State Management**: Redux Toolkit
- **UI Components**: Native Base
- **Purpose**: Primary mobile-first interface

## 🖥️ Backend Services

### API Service (`packages/api`)
- **Framework**: Python FastAPI
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Caching**: Redis
- **Background Jobs**: Celery
- **Purpose**: Main REST API for both web and mobile apps

### AI/ML Service (`packages/agentic`)
- **Language**: Python 3.11+
- **Capabilities**: 
  - Advanced image editing and processing
  - Video processing and manipulation
  - AI-powered tools and filters
  - Face swap functionality
  - Quality assessment
- **Purpose**: Core AI/ML processing engine

## 🔧 Shared Components

### Shared Package (`packages/shared`)
- **Language**: TypeScript
- **Purpose**: Common types, utilities, and logic shared between web and mobile apps

## 🚀 Development Commands

### Frontend Development
```bash
npm run dev:web          # Start Next.js web app
npm run dev:mobile       # Start React Native mobile app
npm run build:web        # Build web app for production
npm run build:mobile     # Build mobile app
```

### Backend Development
```bash
npm run dev:api          # Start FastAPI server
npm run install:python   # Install all Python dependencies
npm run api:test         # Run API tests
```

### AI/ML Development
```bash
npm run agentic:test              # Run all AI/ML tests
npm run agentic:test:quick        # Run quick test suite
npm run agentic:demo:tool-first   # Demo tool-first architecture
npm run agentic:demo:flux         # Demo Flux capabilities
npm run agentic:demo:video        # Demo video processing
```

### Environment Setup
```bash
npm run install:all      # Install all dependencies (Node.js + Python)
npm run setup:test-env   # Setup test environment
npm run validate:test-env # Validate test environment
```

## 🌐 Technology Stack Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Frontend** | Next.js 14 + React 18 + TypeScript | Desktop web interface |
| **Mobile Frontend** | React Native 0.73 + Redux Toolkit | Mobile interface |
| **API Backend** | Python FastAPI + SQLAlchemy | REST API server |
| **AI/ML Engine** | Python + Advanced Libraries | Image/video processing |
| **Database** | PostgreSQL (via Supabase) | Data persistence |
| **Authentication** | Supabase Auth | User management |
| **File Storage** | Supabase Storage | Media file storage |
| **Caching** | Redis | Performance optimization |
| **Styling** | Tailwind CSS + Native Base | UI styling |

## 🔄 Data Flow

1. **Mobile/Web App** → Makes API calls to FastAPI backend
2. **FastAPI Backend** → Handles business logic, auth, and database operations
3. **AI/ML Service** → Processes images/videos when requested by API
4. **Supabase** → Provides database, auth, and file storage
5. **Redis** → Caches frequently accessed data

## 📦 Package Dependencies

- **Node.js**: >=18.17.0
- **Python**: >=3.11.0
- **npm**: >=9.0.0

## 🎯 Key Features

- **Mobile-First Design**: Optimized for mobile users
- **Professional Editing**: Advanced image and video editing capabilities
- **AI-Powered Tools**: Cutting-edge AI/ML processing
- **Cross-Platform**: Web and mobile applications
- **Real-time Processing**: Fast, responsive editing experience
- **Cloud Storage**: Secure file storage and sharing

## 🔧 Development Setup

1. **Clone the repository**
2. **Install all dependencies**: `npm run install:all`
3. **Setup environment variables** (see `.env.example` files)
4. **Start development servers**:
   - API: `npm run dev:api`
   - Web: `npm run dev:web`
   - Mobile: `npm run dev:mobile`

## 📚 Additional Documentation

- See individual package READMEs for detailed setup instructions
- Check `packages/agentic/` for AI/ML specific documentation
- Refer to Supabase documentation for database and auth setup 