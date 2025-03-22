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
            const reasons = data.reasons || [];
            const openai = data.text_analysis || {};
  
            // Flag for AI/deepfake
            if (reasons.some(r => r.toLowerCase().includes('face forgery') || r.includes('geometry'))) {
              chrome.tabs.sendMessage(sender.tab.id, {
                type: 'FLAG_IMAGE',
                category: 'ai',
                imageUrl,
                reason: reasons.join('; ')
              });
            }
  
            // Flag for clickbait
            if (reasons.some(r => r.toLowerCase().includes('clickbait'))) {
              chrome.tabs.sendMessage(sender.tab.id, {
                type: 'FLAG_IMAGE',
                category: 'clickbait',
                imageUrl,
                reason: 'Clickbait content'
              });
            }
  
            // Flag for violence
            if (data.clip_label === 'violent') {
              chrome.tabs.sendMessage(sender.tab.id, {
                type: 'FLAG_IMAGE',
                category: 'violent',
                imageUrl,
                reason: 'Violent content detected'
              });
            }
  
            // Flag for misinformation
            if (openai.accuracy === 'inaccurate' || openai.harmful) {
              chrome.tabs.sendMessage(sender.tab.id, {
                type: 'FLAG_IMAGE',
                category: 'misinformation',
                imageUrl,
                reason: openai.reasoning || 'OpenAI flagged as inaccurate'
              });
            }
          })
          .catch(console.error);
      });
    }
  });
  