function findAllImagesAndAnalyze() {
  const images = document.getElementsByTagName('img');
  const imageUrls = [];

  for (let img of images) {
    if (img.src) {
      imageUrls.push(img.src);
    }
  }

  try {
    chrome.runtime.sendMessage({ type: 'CHECK_IMAGES', images: imageUrls });
  } catch (e) {
    console.warn("Extension context invalidated, stopping interval.");
    clearInterval(imageCheckInterval);
  }
}

// DOMContentLoaded and load listeners clearly set up
window.addEventListener('load', findAllImagesAndAnalyze);
window.addEventListener('DOMContentLoaded', findAllImagesAndAnalyze);

// Clearly store interval reference to disable it on context invalidation
const imageCheckInterval = setInterval(findAllImagesAndAnalyze, 5000);

// Highlight and handle flagged images clearly
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'FLAG_IMAGE') {
      const flaggedImages = document.querySelectorAll(`img[src="${message.imageUrl}"]`);

      flaggedImages.forEach(img => {
          img.style.outline = '3px solid red';
          addInfoIcon(img, message);
      });
  }
});

function addInfoIcon(img, message) {
  try {
    const icon = document.createElement('div');
    icon.textContent = 'ℹ️';
    icon.style.cssText = `
        cursor: pointer;
        background-color: white;
        border-radius: 50%;
        box-shadow: 0 0 5px rgba(0,0,0,0.3);
        padding: 5px;
        font-size: 14px;
        position: absolute;
        z-index: 10000;
    `;

    icon.onclick = () => showExplanationOverlay(
        message.reasons,
        message.visuals,
        message.explanations,
        message.text_analysis
    );

    const rect = img.getBoundingClientRect();
    icon.style.top = `${rect.top + window.scrollY + 5}px`;
    icon.style.left = `${rect.left + window.scrollX + 5}px`;

    document.body.appendChild(icon);
  } catch (e) {
    console.warn("Cannot add icon: context invalidated or DOM detached.");
  }
}
