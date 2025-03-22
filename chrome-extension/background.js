// background.js
chrome.runtime.onMessage.addListener((request, sender) => {
    if (request.type === 'CHECK_IMAGES') {
      request.images.forEach(imageUrl => {
        fetch('http://127.0.0.1:5000/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_url: imageUrl })
        })
        .then(res => res.json())
        .then(data => {
            if (data.verdict === 'AI_GENERATED' || (data.text_analysis && (data.text_analysis.accuracy !== 'accurate' || data.text_analysis.clickbait || data.text_analysis.harmful))) {
                chrome.tabs.sendMessage(sender.tab.id, {
                    type: 'FLAG_IMAGE',
                    imageUrl: imageUrl,
                    reasons: data.reasons,
                    visuals: data.visuals,
                    explanations: data.explanations,
                    text_analysis: data.text_analysis
                });
            }
        })
        .catch(console.error);
      });
    }
  });
  