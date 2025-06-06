## Pic Arcade: Technical Execution Plan (Real API + TDD)

### 🧭 Architecture Overview
Agentic backend with LangGraph orchestrates AI APIs for text-to-image, face swap, video generation, and editing workflows. React Native frontend. Cursor IDE for test-driven development using real data and APIs.

---

## ✅ PHASE 1: SYSTEM FOUNDATION (Weeks 1–3)

### 1. Project Bootstrapping
- Monorepo: `/frontend`, `/backend`, `/agentic`, `/tests`
- CI setup (GitHub Actions)
- Environments: dev, staging, prod

**Tests:**
- Health check hits `/api/health` and returns version metadata
- CI runs all tests using real API calls

---

## 📥 PHASE 2: PROMPT HANDLING + AGENTIC BACKEND (Weeks 4–7)

### 2. Prompt Parsing Agent
- GPT-4o API: extract entities, intent, modifiers, references

**Tests:**
- 50+ prompts
- Output includes: `intent`, `entities`, `modifiers`, `references`

### 3. Reference Retrieval Agent
- Bing or Google Search API
- CLIP ranker for image selection

**Tests:**
- 10 celebrity prompts
- Top 3 image URLs include correct result

---

## 🖼️ PHASE 3: IMAGE GENERATION & EDITING (Weeks 8–11)

### 4. Image Generation Agent
- Flux API (via Replicate)
- Fallback: OpenAI DALL·E

**Tests:**
- 25 real prompts
- Perceptual hash for diversity
- CLIP similarity to references

### 5. Image Editing Agent
- Inpainting/outpainting using Flux-fill or OpenAI Image Edit

**Tests:**
- Background removal, object editing
- Regions match instructions via diff mask

---

## 🧑‍🎤 PHASE 4: FACE SWAP & STYLE TRANSFER (Weeks 12–14)

### 6. Face Swap & Virtual Try-On Agent
- APIs: PiAPI, Segmind

**Tests:**
- 5 real source/target pairs
- Face position, tone, alignment validated

### 7. Style Transfer Agent
- OpenArt API
- Histogram diff to confirm visual style change

**Tests:**
- Real celebrity-to-user style prompts

---

## 🎥 PHASE 5: VIDEO GENERATION & ANIMATION (Weeks 15–18)

### 8. Video Generation Agent
- Runway Gen-4 + Fal.ai

**Tests:**
- 10 real prompts
- Validate frame count, animation, file format

### 9. Video Editing Agent
- Banuba, Runway, HeyGen

**Tests:**
- Validate face motion, background effects
- Face landmarks pre/post edit

---

## 🔍 PHASE 6: QUALITY ASSURANCE + WORKFLOW (Weeks 19–21)

### 10. Quality Assurance Agent
- CLIP scoring, artifact detection

**Tests:**
- 100 samples
- Re-run failed items (<0.8 score)

### 11. LangGraph Orchestration
- State: parse → retrieve → generate → edit → QA → finalize

**Tests:**
- Trace execution path
- Ensure no skipped states

---

## 📱 PHASE 7: FRONTEND & MOBILE UX (Weeks 22–24)

### 12. React Native App
- RTK Query, Redux, NativeBase UI

**Tests:**
- Simulate prompt entry to render
- iOS/Android end-to-end runs

---

## 💵 PHASE 8: PAYMENTS & RATE LIMITING (Weeks 25–26)

### 13. Stripe + Quotas
- Free, Plus, Pro tiers

**Tests:**
- Real subscription flow
- Generation caps per plan

---

## TDD PRINCIPLES FOR CURSOR
- Real API invocations only
- Assertions based on perceptual + semantic output
- Test failure paths, retries, and fallback routing

---

## 🚀 LAUNCH METRICS
| Metric                         | Target            |
|-------------------------------|-------------------|
| Beta users                    | 10,000+           |
| Avg latency                   | < 5s              |
| QA pass rate                  | > 85%             |
| Registered users (6 mo)       | 100,000+          |
| MRR                           | $500,000+         |

---

## 🗂️ Folder Structure Overview

```
pic-arcade/
├── apps/
│   ├── mobile/              # React Native app
│   └── web/                 # Optional web front (e.g., Next.js previewer)
├── packages/
│   ├── agentic/             # LangGraph backend logic
│   ├── api/                 # REST API layer (FastAPI or Next API routes)
│   ├── shared/              # Type definitions, constants, and utilities
│   └── tests/               # Real API integration tests
├── .github/workflows/       # CI pipeline (GitHub Actions)
├── .env.example             # Env file with real API keys (masked)
└── README.md
```

### 📦 `packages/agentic/`
Contains all LangGraph logic using Python + LangChain.

### 🧪 `packages/tests/`
All tests use **real data** and call external services. Structured by phase.

### 📱 `apps/mobile/`
React Native app using Redux Toolkit, RTK Query, NativeBase.

### 🌐 `packages/api/`
FastAPI or Next.js route handlers. Interfaces with LangGraph orchestrator.

### 🔁 CI: `.github/workflows/test.yml`
Runs all tests with `pytest`, React Native lint/type check, and enforces real API use.

### 🧪 Real API Key Setup
`.env.example` with real keys (masked): OPENAI_API_KEY, BING_API_KEY, etc.
