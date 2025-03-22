// popup.js

document.addEventListener('DOMContentLoaded', () => {
    const toggleScanBtn = document.getElementById('toggleScanBtn');
    const statusEl = document.getElementById('status');
  
    // Example: store scanning state in chrome.storage
    chrome.storage.local.get(['scanningEnabled'], (result) => {
      const scanningEnabled = result.scanningEnabled ?? true; // default true
      statusEl.textContent = `Scanning is ${scanningEnabled ? 'ON' : 'OFF'}`;
    });
  
    toggleScanBtn.addEventListener('click', () => {
      chrome.storage.local.get(['scanningEnabled'], (res) => {
        const scanningEnabled = !res.scanningEnabled; // flip state
        chrome.storage.local.set({ scanningEnabled });
        statusEl.textContent = `Scanning is ${scanningEnabled ? 'ON' : 'OFF'}`;
      });
    });
  });
  