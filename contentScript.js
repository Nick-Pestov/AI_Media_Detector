// contentScript.js
// Injected into every web page specified by "matches" in manifest.json

/********************************************************************
 * 1) CREATE / INJECT A SIDE PANEL FOR SHOWING WARNINGS OR RESULTS
 ********************************************************************/
const panelId = 'ai-detector-panel';
let panel = document.getElementById(panelId);
if (!panel) {
  panel = document.createElement('div');
  panel.id = panelId;
  // Basic styling. The rest is in overlay.css, but we can do some inline too:
  panel.style.cssText = `
    position: fixed;
    top: 60px;
    right: 0;
    width: 320px;
    max-height: 70%;
    overflow-y: auto;
    background: #fff;
    border-left: 2px solid #ccc;
    padding: 10px;
    z-index: 999999;
    font-family: Arial, sans-serif;
  `;
  document.body.appendChild(panel);
}

/** Helper to append messages or logs to the side panel */
function appendToPanel(message) {
  const el = document.createElement('p');
  el.innerText = message;
  panel.appendChild(el);
}

/** Highlight an image in red outline and log reason in side panel */
function highlightImage(img, reason) {
  img.style.outline = '3px solid red';
  appendToPanel(reason);
}

/********************************************************************
 * 2) FIND & PROCESS IMAGES
 ********************************************************************/
const images = document.querySelectorAll('img');

images.forEach(async (img, index) => {
  try {
    // A) Convert the image to base64 DataURL (if same-origin or CORS enabled)
    const dataURL = await fetchImageAsDataURL(img.src);
    if (!dataURL) return;

    // B) OCR: Extract text from the image
    const extractedText = await runOCR(dataURL);
    if (extractedText) {
      // C) Check for misinformation / clickbait
      const { isClickbait, isFalseClaim, reason } = await analyzeTextForMisinformation(extractedText);
      if (isClickbait || isFalseClaim) {
        highlightImage(img, `Image #${index} - Text flagged: ${reason}`);
      }
    }

    // D) AI Manipulation or AI-Generated check
    const { isAIGenerated, confidence, suspectRegions } = await checkAIManipulation(dataURL);
    if (isAIGenerated) {
      highlightImage(img, `Image #${index} - Possibly AI-generated (confidence: ${confidence}%)`);

      // Optionally draw bounding boxes for suspect regions
      if (Array.isArray(suspectRegions)) {
        suspectRegions.forEach(region => {
          drawOverlayBox(img, region);
        });
      }
    }
  } catch (error) {
    console.warn(`Error analyzing image #${index}:`, error);
  }
});

/********************************************************************
 * 3) SUPPORTING FUNCTIONS
 ********************************************************************/

/** 
 * Convert an image URL to base64 DataURL 
 * Returns null if fetch fails or is blocked by CORS
 */
async function fetchImageAsDataURL(url) {
  try {
    const res = await fetch(url, { mode: 'cors' });
    if (!res.ok) throw new Error(`Fetch failed with status: ${res.status}`);
    const blob = await res.blob();
    return await blobToBase64(blob);
  } catch (e) {
    console.warn('Could not fetch image:', url, e);
    return null;
  }
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/** 
 * Run OCR to extract text from the base64 image
 * In a real app, you might:
 *   1) Use Tesseract.js locally
 *   2) Or send to a server OCR API
 * 
 * Below is a placeholder that returns "" for now.
 */
async function runOCR(base64Image) {
  // Example using Tesseract.js (CDN script) if you included it in your extension:
  // return Tesseract.recognize(base64Image, 'eng').then(({ data: { text } }) => text);

  // For demonstration, we just return an empty string or some dummy text
  // Replace with real OCR logic
  return "";
}

/** 
 * Analyze the extracted text for misinformation or clickbait 
 * This is a placeholder:
 *   - You might call a fact-checking API
 *   - Or run your own ML model
 */
async function analyzeTextForMisinformation(text) {
  // Simple naive example: Check if text has "miracle cure" or "shocking secret" for clickbait
  const clickbaitPhrases = ["miracle cure", "shocking secret", "you won't believe"];
  const isClickbait = clickbaitPhrases.some(phrase => text.toLowerCase().includes(phrase));

  // Check a "fake" fact-check
  let isFalseClaim = false;
  let reason = "";
  if (isClickbait) {
    isFalseClaim = true;
    reason = `Detected clickbait phrase in text: "${text}"`;
  }

  return { isClickbait, isFalseClaim, reason };
}

/** 
 * Check if the image is AI-generated or manipulated. 
 * In reality, you’d call specialized models or APIs (e.g., stable diffusion detection, GAN fingerprint).
 * 
 * For now, we return a dummy response.
 */
async function checkAIManipulation(base64Image) {
    const response = await fetch('http://localhost:5000/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Image })
    });
  
    return await response.json();
  }

/**
 * Draw bounding box overlays on suspicious regions
 * region = { x, y, width, height } relative to the image’s actual pixel size
 */
function drawOverlayBox(img, region) {
  const { x, y, width, height } = region;

  // We must position it correctly on the screen relative to the image's position
  const imgRect = img.getBoundingClientRect();

  // If the image is scaled, we need to find the ratio
  const naturalWidth = img.naturalWidth || imgRect.width;
  const naturalHeight = img.naturalHeight || imgRect.height;
  
  // Scale factor
  const scaleX = imgRect.width / naturalWidth;
  const scaleY = imgRect.height / naturalHeight;

  const overlay = document.createElement('div');
  overlay.style.position = 'absolute';
  overlay.style.border = '2px solid red';
  overlay.style.pointerEvents = 'none';
  overlay.style.left = `${imgRect.left + window.scrollX + (x * scaleX)}px`;
  overlay.style.top = `${imgRect.top + window.scrollY + (y * scaleY)}px`;
  overlay.style.width = `${width * scaleX}px`;
  overlay.style.height = `${height * scaleY}px`;
  overlay.style.zIndex = '999999';

  document.body.appendChild(overlay);
}
