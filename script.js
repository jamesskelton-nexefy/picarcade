// Elements
const promptInput = document.getElementById('prompt');
const generateBtn = document.getElementById('generateBtn');
const displayImage = document.getElementById('displayImage');
const placeholder = document.getElementById('placeholder');
const loadingOverlay = document.getElementById('loadingOverlay');
const statusMessage = document.getElementById('statusMessage');
const btnText = generateBtn.querySelector('.btn-text');
const btnSpinner = generateBtn.querySelector('.spinner');

// State
let isGenerating = false;
let currentImageUrl = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    generateBtn.addEventListener('click', generateImage);
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            generateImage();
        }
    });
});

async function generateImage() {
    // Don't generate if already generating or no prompt
    if (isGenerating || !promptInput.value.trim()) {
        if (!promptInput.value.trim()) {
            showStatus('Please enter a prompt first', 'error');
        }
        return;
    }

    isGenerating = true;
    const prompt = promptInput.value.trim();
    
    // Update UI state
    generateBtn.disabled = true;
    btnText.style.display = 'none';
    btnSpinner.style.display = 'block';
    showStatus('');
    
    // If there's a current image, show loading overlay on top of it
    if (currentImageUrl) {
        loadingOverlay.style.display = 'flex';
    } else {
        // If no current image, hide placeholder and show loading
        placeholder.style.display = 'none';
        loadingOverlay.style.display = 'flex';
    }

    try {
        // Simulate image generation with a delay
        // In a real application, this would be an API call
        const imageUrl = await simulateImageGeneration(prompt);
        
        // Only update the image when the new one is ready
        displayImage.src = imageUrl;
        displayImage.style.display = 'block';
        placeholder.style.display = 'none';
        currentImageUrl = imageUrl;
        
        showStatus('Image generated successfully!', 'success');
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
        
        // If there was no previous image, show placeholder again
        if (!currentImageUrl) {
            placeholder.style.display = 'flex';
        }
    } finally {
        // Reset UI state
        isGenerating = false;
        generateBtn.disabled = false;
        btnText.style.display = 'block';
        btnSpinner.style.display = 'none';
        loadingOverlay.style.display = 'none';
    }
}

async function simulateImageGeneration(prompt) {
    // Simulate network delay (2-4 seconds)
    const delay = 2000 + Math.random() * 2000;
    await new Promise(resolve => setTimeout(resolve, delay));
    
    // Simulate occasional errors
    if (Math.random() < 0.1) {
        throw new Error('Failed to generate image. Please try again.');
    }
    
    // Use placeholder images from Unsplash based on prompt keywords
    const keywords = prompt.toLowerCase().split(' ').slice(0, 3).join(',');
    const randomId = Math.floor(Math.random() * 1000);
    
    // Return a random image URL based on the prompt
    return `https://source.unsplash.com/800x800/?${keywords}&sig=${randomId}`;
}

function showStatus(message, type = '') {
    statusMessage.textContent = message;
    statusMessage.className = 'status-message';
    
    if (type) {
        statusMessage.classList.add(type);
    }
    
    if (message && type === 'success') {
        // Auto-hide success messages after 3 seconds
        setTimeout(() => {
            if (statusMessage.textContent === message) {
                statusMessage.textContent = '';
                statusMessage.className = 'status-message';
            }
        }, 3000);
    }
}

// Image loading error handler
displayImage.addEventListener('error', () => {
    showStatus('Failed to load image', 'error');
    displayImage.style.display = 'none';
    
    if (!currentImageUrl) {
        placeholder.style.display = 'flex';
    } else {
        // Try to keep showing the previous image
        displayImage.src = currentImageUrl;
    }
});