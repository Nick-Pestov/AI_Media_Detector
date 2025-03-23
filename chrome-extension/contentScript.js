function findAllImages() {
  const images = document.getElementsByTagName('img');
  const imageUrls = Array.from(images).map(img => img.src).filter(Boolean);
  chrome.runtime.sendMessage({ type: 'CHECK_IMAGES', images: imageUrls });
}

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'FLAG_IMAGE') {
    const imgs = document.querySelectorAll(`img[src="${message.imageUrl}"]`);
    imgs.forEach(img => {
      if ((message.flags || []).some(f => f.startsWith("blur_"))) {
        img.style.filter = 'blur(12px)';
        img.style.transition = 'filter 0.3s ease-in-out';
      }
      addInfoIcon(img, message.reason, message.visuals, message.verdict);
    });
  }
});

function addInfoIcon(img, reason, visuals, verdict) {
  const icon = document.createElement('div');
  icon.textContent = '⚠️';
  icon.title = 'Click for details';
  icon.style.cssText = `
    position: absolute;
    top: ${img.getBoundingClientRect().top + window.scrollY + 5}px;
    left: ${img.getBoundingClientRect().left + window.scrollX + 5}px;
    z-index: 99999;
    font-size: 20px;
    background: #fff;
    border-radius: 50%;
    padding: 2px;
    cursor: pointer;
    box-shadow: 0 0 6px rgba(0,0,0,0.5);
  `;
  icon.onclick = () => showAlertWithVisuals({ verdict, reasons: reason.split('; '), visuals });

  document.body.appendChild(icon);
}

function showAlertWithVisuals({ verdict, reasons, visuals }) {
  const reasonText = reasons.join('<br>');
  const modalId = "aiVisualsModal-" + Date.now();
  const alertId = "alertBox-" + Date.now();

  const modalHTML = `
    <div id="${modalId}" style="display:none; position:fixed; top:10%; left:50%; transform:translateX(-50%);
        background:#1e1e1e; padding:20px; border-radius:10px; z-index:99999; max-height:80%; overflow:auto; color:white;">
        <h3 style="margin-top:0;">🔍 AI Detection Visuals</h3>
        <p>${reasonText}</p>
        ${Object.entries(visuals).map(([title, base64]) => `
            <div style="margin-bottom:15px;">
                <h4 style="margin-bottom:5px;">${title.replace(/_/g, ' ')}</h4>
                <img src="data:image/png;base64,${base64}" style="max-width:100%; border:1px solid #333; border-radius:6px;" />
            </div>
        `).join('')}
        <button onclick="document.getElementById('${modalId}').remove()" 
            style="margin-top:10px; padding:6px 14px; background:#d9534f; color:white; border:none; border-radius:6px;">
            Close
        </button>
    </div>
  `;

  const alertBox = document.createElement('div');
  alertBox.id = alertId;
  alertBox.style = `
    position: fixed;
    bottom: 10px;
    left: 10px;
    background: #2c2c2c;
    padding: 16px;
    border-radius: 10px;
    border: 2px solid red;
    color: white;
    z-index: 99999;
    min-width: 300px;
    max-width: 400px;
    box-shadow: 0 0 10px rgba(0,0,0,0.5);
    position: relative;
  `;

  alertBox.innerHTML = `
    <button onclick="document.getElementById('${alertId}').remove()" 
      style="position: absolute; top: 6px; right: 8px; background: transparent; color: white;
      border: none; font-size: 18px; cursor: pointer; font-weight: bold;">✕</button>

    <div>
      <strong>🚩 AI-Generated Image Detected</strong><br>
      <div style="margin-top: 5px;">${reasonText}</div>
      ${Object.keys(visuals).length > 0
        ? `<button id="showVisualsBtn-${alertId}" style="margin-top:10px; padding:6px 14px; background:#3a3a3a; color:white; border:none; border-radius:6px;">See More</button>`
        : ''}
      ${modalHTML}
    </div>
  `;

  document.body.appendChild(alertBox);

  const showBtn = document.getElementById(`showVisualsBtn-${alertId}`);
  if (showBtn) {
    showBtn.onclick = () => {
      document.getElementById(modalId).style.display = "block";
    };
  }
}

findAllImages();
