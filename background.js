// background.js (service worker in Manifest V3)

// Optional: You can handle external messages, do authentication, or call external APIs here.
// For now, we'll keep it minimal.

chrome.runtime.onInstalled.addListener(() => {
    console.log('AI & Misinformation Detector extension installed.');
  });
  