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
    imgs.forEach(img => {
      switch (message.category) {
        case 'ai':
        case 'clickbait':
          img.style.outline = '3px solid red';
          addInfoIcon(img, message.reason);
          break;
        case 'violent':
          blurImageWithReveal(img, message.reason);
          break;
        case 'misinformation':
          addInfoIcon(img, message.reason, true);
          break;
      }
    });
  }
});

function addInfoIcon(img, reason, showOnHoverOnly = false) {
  const icon = document.createElement('div');
  icon.textContent = 'ℹ️';
  icon.title = reason;
  icon.style.cssText = `
    position: absolute;
    font-size: 16px;
    background: white;
    padding: 3px;
    border-radius: 50%;
    box-shadow: 0 0 5px rgba(0,0,0,0.3);
    cursor: pointer;
    z-index: 9999;
    ${showOnHoverOnly ? 'opacity: 0; transition: opacity 0.3s;' : ''}
  `;

  const rect = img.getBoundingClientRect();
  icon.style.top = `${rect.top + window.scrollY + 5}px`;
  icon.style.left = `${rect.left + window.scrollX + 5}px`;

  icon.onclick = () => alert(`ℹ️ Explanation:\n${reason}`);
  document.body.appendChild(icon);

  if (showOnHoverOnly) {
    img.addEventListener('mouseenter', () => icon.style.opacity = 1);
    img.addEventListener('mouseleave', () => icon.style.opacity = 0);
  }
}

function blurImageWithReveal(img, reason) {
  img.style.filter = 'blur(12px)';
  img.style.transition = 'filter 0.3s';

  const revealBtn = document.createElement('button');
  revealBtn.textContent = '🔓 Reveal';
  revealBtn.style.cssText = `
    position: absolute;
    z-index: 9999;
    background: rgba(255,255,255,0.9);
    border: 1px solid #ccc;
    padding: 4px 8px;
    font-size: 12px;
    cursor: pointer;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
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


// Initial trigger
findAllImages();
