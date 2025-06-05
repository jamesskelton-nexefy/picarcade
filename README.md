# PicArcade

A modern web-based image generator with a key UX feature: **the current image remains visible while generating a new one**.

## Features

- 🖼️ **Persistent Image Display**: When generating a new image, the previous image stays visible with a loading overlay on top
- 🎨 **Beautiful Modern UI**: Clean, responsive design with smooth animations
- ⚡ **Fast Feedback**: Instant visual feedback during generation
- 🔄 **Smart Error Handling**: Preserves the current image even if generation fails

## How It Works

1. Enter a prompt describing the image you want to generate
2. Click "Generate Image" or press Enter
3. The current image (if any) remains visible while the new one is being generated
4. A semi-transparent loading overlay appears on top of the current image
5. Once the new image is ready, it seamlessly replaces the old one

## Usage

Simply open `index.html` in a web browser. No server or installation required!

## Technical Details

The application uses:
- Pure HTML/CSS/JavaScript (no frameworks required)
- Simulated image generation using Unsplash API for demonstration
- Responsive design that works on all devices
- Graceful error handling and loading states

## Key Implementation

The core feature is implemented in `script.js`:
- The current image URL is stored in state
- During generation, a loading overlay is shown on top of the existing image
- The image element is only updated when the new image is fully loaded
- If generation fails, the previous image remains visible