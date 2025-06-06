# Real-World Testing Guide for Pic Arcade

This guide covers comprehensive testing of Pic Arcade's agentic AI backend using **real images and actual API calls**. These tests validate the complete tool-first architecture with Flux 1.1 Pro Ultra integration.

## 🎯 Test Overview

Our test suite includes:
- **Real image processing** using high-quality Unsplash photos
- **Actual API calls** to OpenAI, Replicate, and Perplexity
- **Professional workflows** for marketing, social media, e-commerce, and creative content
- **Performance benchmarks** with real-world timing expectations
- **Error handling** with edge cases and API failures

## 🚀 Quick Start

### 1. Environment Setup

Configure your API keys using our interactive setup script:

```bash
npm run setup:test-env
```

This will guide you through configuring:
- **Replicate API** (required) - For Flux 1.1 Pro Ultra image generation
- **OpenAI API** (required) - For GPT-4o prompt processing
- **Perplexity API** (optional) - For enhanced search functionality

### 2. Validate Setup

Check that your environment is properly configured:

```bash
npm run validate:test-env
```

### 3. Run Quick Tests

Start with a fast validation test:

```bash
npm run test:quick
```

## 📋 Available Test Suites

### Core Integration Tests

| Command | Description | Duration | API Calls |
|---------|-------------|----------|-----------|
| `npm run test:quick` | Fast validation (2-3 tests) | 2-3 min | ~5-10 |
| `npm run test:flux` | All Flux image editing tools | 10-15 min | ~15-25 |
| `npm run test:tool-first` | Complete tool-first workflow | 5-10 min | ~10-15 |
| `npm run test:real-world` | All real-world tests | 20-30 min | ~50-80 |

### Professional Workflow Tests

| Command | Description | Scenarios |
|---------|-------------|-----------|
| `npm run test:demo` | Full professional workflow demo | Marketing campaigns, social media, e-commerce, creative content |
| `npm run test:performance` | Performance benchmarks | Speed and efficiency testing across all operations |

### Legacy Tests (Phase 2)

| Command | Description |
|---------|-------------|
| `npm run test:phase2` | Original Phase 2 tests (prompt parsing, reference retrieval) |
| `npm run test:prompt-parser` | GPT-4o prompt parsing validation |
| `npm run test:reference-retriever` | Bing Search API integration |

## 🎨 Test Categories

### 1. Flux Tools Tests (`test:flux`)

Tests all Flux 1.1 Pro Ultra capabilities:

- **Image Generation**: Creating images from text prompts
- **Style Transfer**: Watercolor, oil painting, sketch, digital art styles
- **Object Modification**: Hair changes, clothing swaps, color adjustments
- **Background Swapping**: Environment changes while preserving subjects
- **Text Editing**: Replacing text in signs, posters, labels
- **Character Consistency**: Maintaining identity across variations
- **Parameter Validation**: Aspect ratios, formats, normalization

**Real Images Used**:
- Professional portraits from Unsplash
- Business headshots and lifestyle photos
- Storefronts and signage for text editing
- Full-body shots for background changes

### 2. Tool-First Integration Tests (`test:tool-first`)

Tests the complete tool-first architecture:

- **Dynamic Workflow Planning**: AI-powered tool selection
- **Multi-step Processing**: Complex editing sequences
- **Error Recovery**: Graceful handling of failures
- **Performance Optimization**: Efficient tool chaining
- **Real Image Processing**: End-to-end workflows

**Test Scenarios**:
- Style transfer with real portraits
- Object changes on business photos
- Background swapping for lifestyle images
- Complex multi-step transformations
- Batch processing workflows

### 3. Real-World Demo Tests (`test:demo`)

Professional workflow demonstrations:

#### Marketing Campaign Workflow
- Executive portrait enhancement (oil painting style, office background)
- Product rebranding with luxury aesthetics
- Billboard advertisement creation with neon effects

#### Social Media Content Creation
- Instagram-ready lifestyle content (1:1 ratio, vibrant colors)
- Pinterest travel pins (2:3 ratio, vintage style)
- TikTok food content (9:16 ratio, dynamic filters)

#### E-commerce Product Enhancement
- Fashion photography (outfit changes, studio lighting)
- Lifestyle product showcases (warm environments)
- Retail space presentation (branding, atmosphere)

#### Creative Artistic Transformations
- Fine art portraits (Renaissance style, chiaroscuro lighting)
- Architectural visualizations (blueprint overlays)
- Nature art (impressionist painting style)

## 🔧 Configuration

### Required API Keys

#### Replicate API (Required)
- **Purpose**: Flux 1.1 Pro Ultra image generation
- **Get Key**: https://replicate.com/account/api-tokens
- **Format**: `r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
- **Cost**: ~$0.003-0.01 per image generation

#### OpenAI API (Required)
- **Purpose**: GPT-4o prompt processing and workflow planning
- **Get Key**: https://platform.openai.com/api-keys
- **Format**: `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
- **Cost**: ~$0.01-0.03 per request

#### Perplexity API (Optional)
- **Purpose**: Enhanced search and reference retrieval
- **Get Key**: https://www.perplexity.ai/settings/api
- **Format**: `pplx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Environment Variables

Create a `.env` file in the project root:

```env
# Replicate API (Flux Image Generation)
REPLICATE_API_TOKEN=r8_your_token_here

# OpenAI API (GPT-4o)
OPENAI_API_KEY=sk-your_key_here

# Perplexity API (Search) - Optional
PERPLEXITY_API_KEY=pplx-your_key_here
```

## 📊 Performance Expectations

### Typical Processing Times

| Operation | Expected Time | API Calls | Cost Estimate |
|-----------|--------------|-----------|---------------|
| Style Transfer | 30-45s | 2-3 | $0.01-0.05 |
| Object Change | 25-40s | 2-3 | $0.01-0.05 |
| Background Swap | 35-50s | 2-4 | $0.02-0.06 |
| Text Editing | 30-45s | 2-3 | $0.01-0.05 |
| Character Consistency | 40-60s | 3-4 | $0.02-0.07 |
| Complex Workflow | 60-120s | 5-8 | $0.05-0.15 |

### Success Rate Targets

- **Individual Operations**: ≥90% success rate
- **Workflow Tests**: ≥85% success rate  
- **Complex Multi-step**: ≥80% success rate
- **Batch Processing**: ≥85% success rate

## 🚨 Important Notes

### Cost Management
- Tests make **real API calls** and incur actual costs
- Estimated total cost for full test suite: **$2-5**
- Rate limiting is built-in to avoid overwhelming APIs
- Failed tests don't incur generation costs

### Test Images
- All test images are high-quality photos from Unsplash
- Images are publicly accessible and license-free
- URLs include optimization parameters for consistent testing
- No sensitive or copyrighted content is used

### Rate Limiting
- Built-in delays between API calls (2-5 seconds)
- Respects API rate limits and fair usage
- Longer delays for complex operations
- Automatic retry logic for transient failures

## 🐛 Troubleshooting

### Common Issues

#### 1. Missing API Keys
```bash
❌ Missing REPLICATE_API_TOKEN: Not configured (REQUIRED)
```
**Solution**: Run `npm run setup:test-env` to configure keys

#### 2. Invalid API Key Format
```bash
⚠️ Invalid format OPENAI_API_KEY: sk-abc... (should be 40+ chars)
```
**Solution**: Check your API key format and regenerate if needed

#### 3. Network/API Errors
```bash
ReplicateError: Model not found or unavailable
```
**Solution**: Check API status and retry. Some models may be temporarily unavailable.

#### 4. Rate Limiting
```bash
HTTP 429: Too Many Requests
```
**Solution**: Tests include automatic backoff. Wait and retry.

### Debug Mode

Run tests with verbose output:

```bash
# Detailed test output
npm run test:flux -- -v -s

# Show API calls and timing
npm run test:quick -- -v -s --tb=short
```

### Test Individual Components

```bash
# Test specific Flux capability
python -m pytest tests/test_flux_tools.py::TestFluxTools::test_style_transfer_variations -v

# Test specific workflow
python -m pytest tests/test_real_world_demo.py::TestRealWorldDemo::test_marketing_campaign_workflow -v
```

## 📈 Performance Monitoring

### Benchmark Results

After running performance tests, you'll see:

```
📊 Performance Summary:
   Success Rate: 4/4 (100%)
   Average Time: 42.3s
   Max Time: 58.1s

✅ generation: 35.2s
✅ style_transfer: 41.8s  
✅ object_change: 38.9s
✅ background_swap: 53.4s
```

### Monitoring Commands

```bash
# Check environment status
npm run validate:test-env

# Show all available test commands
npm run show:test-commands

# Monitor test progress
npm run test:performance -- -v
```

## 🎯 Best Practices

### Before Running Tests
1. **Verify API keys** are properly configured
2. **Check account balances** to ensure sufficient credits
3. **Run quick test first** to validate setup
4. **Monitor costs** during test execution

### During Testing
1. **Don't interrupt** tests mid-execution (may leave partial charges)
2. **Monitor output** for errors or unexpected behavior
3. **Note performance metrics** for optimization
4. **Check rate limiting** if tests slow down

### After Testing
1. **Review test results** and success rates
2. **Check API usage** in provider dashboards
3. **Report issues** with specific test names and errors
4. **Clean up** any temporary files or resources

## 🔄 Continuous Integration

For CI/CD environments, use environment-based configuration:

```bash
# Validate environment in CI
npm run validate:test-env

# Run subset for CI (faster, cheaper)
npm run test:quick

# Full test suite for releases
npm run test:real-world
```

---

## 📞 Support

If you encounter issues:

1. **Check this guide** for common solutions
2. **Validate your setup** with `npm run validate:test-env`
3. **Try quick tests** first: `npm run test:quick`
4. **Review API provider status** pages
5. **Open an issue** with test output and environment details

Happy testing! 🚀 