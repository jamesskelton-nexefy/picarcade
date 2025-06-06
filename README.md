# 🎮 PicArcade

A **mobile-first** professional picture and video editor with cutting-edge AI/ML capabilities. Built as a modern monorepo with intelligent AI agents and a beautiful cross-platform interface.

[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2014-black)](apps/web)
[![React Native](https://img.shields.io/badge/Mobile-React%20Native%200.73-blue)](apps/mobile)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-red)](packages/api)
[![Python AI](https://img.shields.io/badge/AI-Python%203.11-green)](packages/agentic)

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18.17+
- **Python** 3.11+
- **API Keys**: OpenAI, Replicate (see [Environment Setup](#environment-setup))

### Installation
```bash
git clone <repo-url>
cd picarcade
npm run install:all     # Installs both Node.js and Python dependencies
```

### Environment Setup
```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=sk-...
# REPLICATE_API_TOKEN=r8_...
```

### Start Development
```bash
# Terminal 1: Start API server
npm run dev:api          # http://localhost:8000

# Terminal 2: Start web app  
npm run dev:web          # http://localhost:3000

# Terminal 3: Start mobile app (optional)
npm run dev:mobile       # React Native development
```

## 🎯 What is PicArcade?

PicArcade is a **professional-grade image and video editor** that combines:

- **🤖 AI-Powered Tools**: Advanced image generation, style transfer, object manipulation
- **📱 Mobile-First Design**: Optimized for mobile users with responsive web interface
- **🎬 Video Processing**: AI video generation and editing capabilities
- **⚡ Real-time Processing**: Fast, responsive editing experience
- **☁️ Cloud Integration**: Supabase for auth, storage, and database

## 🌟 Key Features

### **Image Editing & Generation**
- **Advanced AI Generation**: Powered by Flux 1.1 Pro Ultra
- **Style Transfer**: Convert photos to artistic styles
- **Object Manipulation**: Change clothing, accessories, backgrounds
- **Text Editing**: Replace text within images
- **Face Swap**: Professional face replacement
- **Quality Enhancement**: AI-powered image improvement

### **Video Processing**
- **AI Video Generation**: Runway ML, Google Veo 2, Luma Ray
- **Video Enhancement**: Upscaling, stabilization, style transfer
- **Multi-format Support**: Professional video editing tools

### **Cross-Platform Experience**
- **Web App**: Next.js 14 with beautiful UI and real-time visualization
- **Mobile App**: React Native with native performance
- **Unified Backend**: FastAPI serving both platforms

## 🏗️ Architecture

This project uses a **hybrid monorepo architecture**:

```
pic-arcade/
├── apps/
│   ├── web/                 # Next.js 14 + React 18 + TypeScript
│   └── mobile/             # React Native 0.73 + Redux Toolkit
├── packages/
│   ├── api/                # Python FastAPI + SQLAlchemy
│   ├── agentic/           # Python AI/ML Processing Engine
│   └── shared/            # TypeScript Shared Utilities
```

**📖 For detailed architecture information, see [ARCHITECTURE.md](./ARCHITECTURE.md)**

## 🛠️ Development Commands

### **Frontend Development**
```bash
npm run dev:web          # Next.js web app (http://localhost:3000)
npm run dev:mobile       # React Native mobile app
npm run build:web        # Build web app for production
npm run build:mobile     # Build mobile app
```

### **Backend Development**
```bash
npm run dev:api          # FastAPI server (http://localhost:8000)
npm run install:python   # Install all Python dependencies
npm run api:test         # Run API tests
```

### **AI/ML Development**
```bash
npm run agentic:test              # Run AI/ML tests
npm run agentic:test:quick        # Quick test suite
npm run agentic:demo:tool-first   # Demo AI agent
npm run agentic:demo:flux         # Demo image generation
npm run agentic:demo:video        # Demo video processing
```

### **Environment & Setup**
```bash
npm run install:all      # Install all dependencies
npm run setup:test-env   # Setup test environment
npm run validate:test-env # Validate environment
```

## 🌐 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Frontend** | Next.js 14 + React 18 + TypeScript | Desktop interface |
| **Mobile Frontend** | React Native 0.73 + Redux Toolkit | Mobile interface |
| **API Backend** | Python FastAPI + SQLAlchemy | REST API server |
| **AI/ML Engine** | Python + Advanced Libraries | Image/video processing |
| **Database** | PostgreSQL (via Supabase) | Data persistence |
| **Authentication** | Supabase Auth | User management |
| **File Storage** | Supabase Storage | Media storage |
| **Styling** | Tailwind CSS + Native Base | UI components |

## 🔑 Environment Configuration

### **Required API Keys**

#### **OpenAI (Required)**
- **Purpose**: GPT-4o for AI workflow planning
- **Get Key**: [OpenAI API Keys](https://platform.openai.com/api-keys)
- **Variable**: `OPENAI_API_KEY=sk-...`

#### **Replicate (Required)**
- **Purpose**: Flux 1.1 Pro Ultra for image generation
- **Get Key**: [Replicate API Tokens](https://replicate.com/account/api-tokens)
- **Variable**: `REPLICATE_API_TOKEN=r8_...`

#### **Optional Services**
- **Perplexity**: Enhanced search capabilities (`PERPLEXITY_API_KEY`)
- **Runway ML**: Premium video generation (`RUNWAYML_API_SECRET`)
- **Supabase**: Database and auth (`SUPABASE_SERVICE_ROLE_KEY`)

### **Environment Files**
```bash
# Root .env (main configuration)
OPENAI_API_KEY=sk-your-key-here
REPLICATE_API_TOKEN=r8_your-token-here

# Package-specific .env files are auto-synced from root
```

## 📊 Usage Examples

### **Simple Image Generation**
```
"Generate a portrait of Emma Stone in Renaissance style"
→ AI selects optimal tools → Beautiful portrait generated
```

### **Advanced Style Transfer**
```
"Make this photo look like Van Gogh painted it"
→ Style analysis → Artistic transformation → Van Gogh style result
```

### **Complex Multi-Step Workflow**
```
"Find a reference of the Mona Lisa and create a cyberpunk version"
→ Reference search → Style analysis → Cyberpunk transformation
```

## 🧪 Testing

The project includes comprehensive testing with **real API calls**:

```bash
# Quick validation (2-3 tests, ~3 minutes)
npm run agentic:test:quick

# Performance benchmarks
npm run agentic:test:performance

# Full test suite (20-30 minutes)
npm run agentic:test
```

**All tests use actual API calls** to ensure real-world reliability.

## 📈 Performance

- **Overall Success Rate**: ≥90% across all workflows
- **Average Response Time**: 30-60 seconds for complex workflows
- **Cost per Operation**: $0.02-0.08 for most image operations
- **Frontend Load Time**: <2 seconds

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Install dependencies**: `npm run install:all`
4. **Set up environment**: Copy `.env.example` to `.env` and add API keys
5. **Test your changes**: `npm run agentic:test:quick`
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

## 📚 Documentation

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Detailed system architecture
- **[packages/agentic/README.md](./packages/agentic/README.md)** - AI/ML engine documentation
- **[packages/agentic/TESTING.md](./packages/agentic/TESTING.md)** - Testing guide
- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs (when running)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/picarcade/issues)
- **Documentation**: [Architecture Guide](./ARCHITECTURE.md)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) (when running)

---

**Built with ❤️ for professional creators who demand cutting-edge AI tools**