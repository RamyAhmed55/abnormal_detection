/**
 * upload.js — Upload & Test page logic
 * Handles: drag-drop, file upload, SocketIO live detection, VLM result display
 */

'use strict';

// ─── State ────────────────────────────────────────────────────────────────────
let socket = null;
let currentSessionId = null;
let uploadedFile = null;
let isDetecting = false;

// ─── DOM Ready ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initDropZone();
  initSocket();
});

// ─── SocketIO Init ────────────────────────────────────────────────────────────
function initSocket() {
  socket = io();

  socket.on('connect', () => {
    console.log('[Socket] Connected:', socket.id);
  });

  socket.on('detection_started', (data) => {
    console.log('[Socket] Detection started:', data);
    setDetecting(true);
  });

  socket.on('frame', (data) => {
    // Update video feed
    const img = document.getElementById('videoCanvas');
    const placeholder = document.getElementById('videoPlaceholder');
    if (data.frame) {
      img.src = 'data:image/jpeg;base64,' + data.frame;
      img.style.display = 'block';
      placeholder.style.display = 'none';
    }

    // Update HUD
    document.getElementById('statusFps').textContent = `FPS: ${data.fps}`;
    document.getElementById('statusVlm').textContent = `VLM: ${data.vlm_status || 'Idle'}`;
    const detBar = document.getElementById('detectionBar');
    if (detBar) detBar.style.display = 'flex';

    // Update detections list
    updateDetectionBar(data.detections || []);
  });

  socket.on('vlm_analyzing', (data) => {
    console.log('[Socket] VLM analyzing:', data);
    document.getElementById('statusVlm').textContent = `VLM: ${data.message}`;
    showAnalyzingStatus(data);
  });

  socket.on('vlm_result', (result) => {
    console.log('[Socket] VLM result:', result);
    showVLMResult(result);
    addSessionAlert(result);
    // Update sidebar alert count immediately in real-time
    if (result.is_abnormal) {
      const badge = document.getElementById('alertBadge');
      if (badge) {
        let count = parseInt(badge.textContent || '0', 10);
        count = isNaN(count) ? 1 : count + 1;
        badge.textContent = count;
        badge.style.display = 'inline-block';
      }
    }
  });

  socket.on('detection_complete', (data) => {
    console.log('[Socket] Detection complete:', data);
    setDetecting(false);
    if (data.final_result) {
      showVLMResult(data.final_result);
    }
    showToast('Detection complete!', 'success');
  });

  socket.on('error', (data) => {
    console.error('[Socket] Error:', data.message);
    setDetecting(false);
    showToast('Error: ' + data.message, 'error');
  });

  socket.on('disconnect', () => {
    console.log('[Socket] Disconnected');
    if (isDetecting) {
      setDetecting(false);
      showToast('Connection lost.', 'error');
    }
  });
}

// ─── Drop Zone ────────────────────────────────────────────────────────────────
function initDropZone() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');

  dropZone.addEventListener('click', (e) => {
    if (!e.target.closest('button')) {
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files[0]) handleFile(files[0]);
  });
}

// ─── File Handling ────────────────────────────────────────────────────────────
async function handleFile(file) {
  const allowed = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm', 'video/x-msvideo'];
  if (!allowed.some(t => file.type.startsWith('video/')) && !file.name.match(/\.(mp4|avi|mov|mkv|webm)$/i)) {
    showToast('Please upload a video file (MP4, AVI, MOV, MKV, WebM)', 'error');
    return;
  }

  uploadedFile = file;
  await uploadFile(file);
}

async function uploadFile(file) {
  // Show progress
  document.getElementById('dropContent').style.display = 'none';
  document.getElementById('uploadProgress').style.display = 'flex';
  document.getElementById('progressFilename').textContent = file.name;

  const formData = new FormData();
  formData.append('video', file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        document.getElementById('progressFill').style.width = pct + '%';
        document.getElementById('progressPct').textContent = pct + '%';
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status === 200) {
        const data = JSON.parse(xhr.responseText);
        currentSessionId = data.session_id;
        document.getElementById('progressFill').style.width = '100%';
        document.getElementById('progressPct').textContent = '100% — Ready!';
        document.getElementById('uploadActions').style.display = 'flex';
        resolve(data);
      } else {
        const err = JSON.parse(xhr.responseText);
        showToast('Upload failed: ' + (err.error || xhr.status), 'error');
        clearUpload();
        reject(err);
      }
    });

    xhr.addEventListener('error', () => {
      showToast('Upload failed: network error', 'error');
      clearUpload();
      reject(new Error('Network error'));
    });

    xhr.open('POST', '/api/upload');
    xhr.send(formData);
  });
}

// ─── Detection Control ────────────────────────────────────────────────────────
function startDetection() {
  if (!currentSessionId || !socket) return;

  console.log('[Upload] Starting detection for session:', currentSessionId);

  // Reset UI
  document.getElementById('resultCard').style.display = 'none';
  document.getElementById('sessionAlerts').style.display = 'none';
  document.getElementById('sessionAlertsList').innerHTML = '';
  document.getElementById('statusCard').style.display = 'flex';

  socket.emit('start_detection', { session_id: currentSessionId });
}

function stopDetection() {
  if (socket) {
    socket.emit('stop_detection', {});
  }
  setDetecting(false);
}

function setDetecting(active) {
  isDetecting = active;
  document.getElementById('startBtn').disabled = active;
  document.getElementById('stopBtn').style.display = active ? 'flex' : 'none';

  if (!active) {
    document.getElementById('statusCard').style.display = 'none';
  }
}

// ─── UI Updates ───────────────────────────────────────────────────────────────
function updateDetectionBar(detections) {
  const bar = document.getElementById('detectionTracks');
  if (!detections || detections.length === 0) {
    bar.textContent = 'No detections';
    return;
  }
  const tracks = detections.map(d =>
    `L${d.logical_id} ${d.class_name} (${(d.confidence * 100).toFixed(0)}%)`
  ).join('  ·  ');
  bar.textContent = tracks;
}

function showAnalyzingStatus(data) {
  const card = document.getElementById('resultCard');
  const header = document.getElementById('resultCardHeader');
  const icon = document.getElementById('resultStatusIcon');
  const title = document.getElementById('resultTitle');
  const sub = document.getElementById('resultSubtitle');
  const body = document.getElementById('resultBody');

  card.style.display = 'block';
  card.className = 'result-card';
  icon.textContent = '🔍';
  icon.style.background = 'rgba(59,130,246,0.1)';
  title.textContent = 'AI Analysis in Progress';
  sub.textContent = `Analyzing ${data.class_name} detection...`;
  body.innerHTML = `<div class="loading-state" style="padding:12px"><div class="spinner"></div><p>VLM is reviewing the 20-second clip...</p></div>`;
}

function showVLMResult(result) {
  const card = document.getElementById('resultCard');
  const icon = document.getElementById('resultStatusIcon');
  const title = document.getElementById('resultTitle');
  const sub = document.getElementById('resultSubtitle');
  const body = document.getElementById('resultBody');

  const isAbn = result.is_abnormal;
  const sev = result.severity || 'unknown';
  const sevClass = { high: 'sev-high', medium: 'sev-medium', low: 'sev-low' }[sev] || 'sev-low';

  card.style.display = 'block';
  card.className = `result-card result-card--${isAbn ? 'abnormal' : 'normal'}`;

  icon.textContent = isAbn ? '🚨' : '✅';
  icon.style.background = isAbn ? 'rgba(239,68,68,0.1)' : 'rgba(34,197,94,0.1)';

  title.textContent = isAbn ? 'ABNORMAL EVENT DETECTED' : 'Event is Normal';
  sub.textContent = `${(result.event_type || '').toUpperCase()} — ${(result.confidence * 100).toFixed(0)}% confidence`;

  body.innerHTML = `
    <div class="result-row">
      <span class="result-key">Severity:</span>
      <span class="result-val"><span class="severity-badge ${sevClass}">${sev}</span></span>
    </div>
    <div class="result-row">
      <span class="result-key">Description:</span>
      <span class="result-val">${result.description || '—'}</span>
    </div>
    <div class="result-row">
      <span class="result-key">Reasoning:</span>
      <span class="result-val">${result.reasoning || '—'}</span>
    </div>
    <div class="result-row">
      <span class="result-key">Analyzed by:</span>
      <span class="result-val">${result._router || 'unknown'}</span>
    </div>
  `;
}

function addSessionAlert(result) {
  const container = document.getElementById('sessionAlerts');
  const list = document.getElementById('sessionAlertsList');

  if (!container || !list) return;

  const descSnippet = (result.description || '').slice(0, 120);
  const isAbn = result.is_abnormal;

  // Prevent duplicate alert entries with the exact same description
  const existing = list.querySelectorAll('.session-alert-desc');
  for (let el of existing) {
    if (el.textContent === descSnippet) return;
  }

  container.style.display = 'block';

  const item = document.createElement('div');
  item.className = 'session-alert-item';
  item.innerHTML = `
    <div class="session-alert-icon">${isAbn ? '🚨' : '✅'}</div>
    <div class="session-alert-body">
      <div class="session-alert-type">${isAbn ? 'ABNORMAL' : 'NORMAL'} — ${(result.event_type || '').toUpperCase()}</div>
      <div class="session-alert-desc">${descSnippet}</div>
    </div>
  `;
  list.prepend(item);
}

function clearUpload() {
  uploadedFile = null;
  currentSessionId = null;
  document.getElementById('dropContent').style.display = 'block';
  document.getElementById('uploadProgress').style.display = 'none';
  document.getElementById('uploadActions').style.display = 'none';
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('progressPct').textContent = '0%';
  document.getElementById('fileInput').value = '';
  document.getElementById('resultCard').style.display = 'none';
  document.getElementById('statusCard').style.display = 'none';
  document.getElementById('videoCanvas').style.display = 'none';
  document.getElementById('videoPlaceholder').style.display = 'flex';
  document.getElementById('detectionBar').style.display = 'none';
}

// ─── Toast Notifications ──────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const existing = document.getElementById('toastContainer');
  const container = existing || (() => {
    const el = document.createElement('div');
    el.id = 'toastContainer';
    el.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:8px';
    document.body.appendChild(el);
    return el;
  })();

  const colors = {
    success: '#22c55e',
    error: '#ef4444',
    info: '#3b82f6',
  };

  const toast = document.createElement('div');
  toast.style.cssText = `
    background: #1e2135;
    border: 1px solid ${colors[type] || colors.info}40;
    border-left: 3px solid ${colors[type] || colors.info};
    color: #f0f2ff;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 13px;
    max-width: 320px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    animation: slideInRight 0.3s ease;
    cursor: pointer;
  `;
  toast.textContent = message;
  toast.onclick = () => toast.remove();

  container.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
}
