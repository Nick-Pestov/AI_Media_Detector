// contentScript.js
function findAllImages() {
  const images = document.getElementsByTagName('img');
  const imageUrls = [];

  for (let img of images) {
      if (img.src) {
          imageUrls.push(img.src);
      }
  }

  chrome.runtime.sendMessage({ type: 'CHECK_IMAGES', images: imageUrls });
}

// Highlight AI-flagged images
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'FLAG_IMAGE') {
      const flaggedImages = document.querySelectorAll(`img[src="${message.imageUrl}"]`);

      flaggedImages.forEach(img => {
          img.style.outline = '3px solid red';
          addInfoIcon(img, message.reason);
      });
  }
});

function addInfoIcon(img, reason) {
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

  icon.title = reason;
  icon.onclick = () => alert(`🚩 AI-Generated Image Detected:\n\nReason: ${reason}`);

  const rect = img.getBoundingClientRect();
  icon.style.top = `${rect.top + window.scrollY + 5}px`;
  icon.style.left = `${rect.left + window.scrollX + 5}px`;

  document.body.appendChild(icon);
}

// Initial trigger
findAllImages();
