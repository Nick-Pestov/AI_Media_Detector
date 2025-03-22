// background.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'CHECK_IMAGES') {
      request.images.forEach(imageUrl => {
          fetch('http://localhost:5000/analyze', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ image_url: imageUrl })
          })
          .then(res => res.json())
          .then(data => {
              if (data.verdict === 'AI_GENERATED') {
                  chrome.tabs.sendMessage(sender.tab.id, {
                      type: 'FLAG_IMAGE',
                      imageUrl: imageUrl,
                      reason: data.reason
                  });
              }
          })
          .catch(console.error);
      });
  }
});
