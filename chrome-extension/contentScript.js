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
    const imgs = document.querySelectorAll(`img[src="${message.imageUrl}"]`);
    const flags = message.flags || [];
    const reason = message.reason || 'Flagged content';

    imgs.forEach(img => {
      // Handle violence or clickbait with blur
      if (flags.includes('blur_violent') || flags.includes('blur_clickbait')) {
        img.style.filter = 'blur(12px)';
        img.style.transition = 'filter 0.3s';

        const revealBtn = document.createElement('button');
        revealBtn.textContent = '🔓 Reveal';
        revealBtn.style.cssText = `
          position: absolute;
          background: rgba(255,255,255,0.95);
          border: 1px solid #ccc;
          padding: 4px 8px;
          font-size: 12px;
          cursor: pointer;
          z-index: 9999;
        `;
        const rect = img.getBoundingClientRect();
        revealBtn.style.top = `${rect.top + window.scrollY + 5}px`;
        revealBtn.style.left = `${rect.left + window.scrollX + 5}px`;
        revealBtn.onclick = () => {
          img.style.filter = 'none';
          revealBtn.remove();
        };
        document.body.appendChild(revealBtn);
      }

      // Show the info icon regardless
      addInfoIcon(img, reason);
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
