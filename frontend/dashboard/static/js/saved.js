/**
 * saved.js — Saved Videos page logic
 */

'use strict';

let currentVideo = null;

document.addEventListener('DOMContentLoaded', loadSavedVideos);

async function loadSavedVideos() {
  const container = document.getElementById('savedContainer');
  try {
    const resp = await fetch('/api/saved');
    const videos = await resp.json();

    document.getElementById('savedCount').textContent =
      `${videos.length} video${videos.length !== 1 ? 's' : ''} saved`;

    if (!videos.length) {
      container.innerHTML = `
        <div class="empty-state" style="grid-column:1/-1">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/>
          </svg>
          <p>No saved videos yet. Save alert videos from the Alerts page.</p>
        </div>`;
      return;
    }

    container.innerHTML = videos.map(v => savedCardHTML(v)).join('');
  } catch (e) {
    container.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><p>Failed to load saved videos.</p></div>`;
  }
}

function savedCardHTML(video) {
  const isAbn = video.is_abnormal;
  const sevClass = { high: 'sev-high', medium: 'sev-medium', low: 'sev-low' }[video.severity] || 'sev-low';
  const icon = isAbn ? '🚨' : '✅';
  const typeLabel = (video.event_type || 'unknown').toUpperCase();
  const dateStr = formatSavedDate(video.saved_at);

  return `
    <div class="saved-card" onclick="openSavedVideo(${JSON.stringify(video).replace(/"/g, '&quot;')})">
      <div class="saved-card-thumb">
        <div class="saved-thumb-placeholder">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/>
          </svg>
          <span>${typeLabel}</span>
        </div>
        <div class="saved-play-btn">
          <svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        </div>
      </div>

      <div class="saved-card-body">
        <div class="saved-card-title">${icon} ${typeLabel} Event</div>
        <div class="saved-card-desc">${video.description || 'No description available.'}</div>
        <div class="saved-card-meta">
          <span class="severity-badge ${sevClass}">${video.severity || '—'}</span>
          <span>${video.size_mb} MB · ${dateStr}</span>
        </div>
      </div>
    </div>`;
}

function openSavedVideo(video) {
  currentVideo = video;

  // Set video source
  const videoEl = document.getElementById('savedVideo');
  videoEl.src = `/api/saved/${encodeURIComponent(video.filename)}/video`;
  videoEl.load();

  // Header
  const isAbn = video.is_abnormal;
  document.getElementById('savedModalHeader').innerHTML = `
    <div style="display:flex;align-items:center;gap:12px">
      <span style="font-size:24px">${isAbn ? '🚨' : '✅'}</span>
      <div>
        <h2 style="font-size:16px;font-weight:700;color:var(--text-primary)">
          ${(video.event_type || 'unknown').toUpperCase()} Recording
        </h2>
        <p style="font-size:12px;color:var(--text-secondary)">${video.filename}</p>
      </div>
    </div>`;

  // Meta info
  const sevClass = { high: 'sev-high', medium: 'sev-medium', low: 'sev-low' }[video.severity] || 'sev-low';
  document.getElementById('savedMeta').innerHTML = `
    <div class="saved-meta-item">
      <div class="saved-meta-label">Event Type</div>
      <div class="saved-meta-value">${(video.event_type || '—').toUpperCase()}</div>
    </div>
    <div class="saved-meta-item">
      <div class="saved-meta-label">Severity</div>
      <div class="saved-meta-value"><span class="severity-badge ${sevClass}">${video.severity || '—'}</span></div>
    </div>
    <div class="saved-meta-item">
      <div class="saved-meta-label">File Size</div>
      <div class="saved-meta-value">${video.size_mb} MB</div>
    </div>
    <div class="saved-meta-item">
      <div class="saved-meta-label">Saved At</div>
      <div class="saved-meta-value">${formatSavedDate(video.saved_at)}</div>
    </div>
    ${video.description ? `<div class="saved-meta-item" style="grid-column:1/-1">
      <div class="saved-meta-label">Description</div>
      <div class="saved-meta-value" style="color:var(--text-secondary);font-size:12px;line-height:1.5">${video.description}</div>
    </div>` : ''}`;

  document.getElementById('savedModal').style.display = 'flex';
}

function closeSavedModal(event) {
  if (event.target === document.getElementById('savedModal')) {
    document.getElementById('savedModal').style.display = 'none';
    document.getElementById('savedVideo').pause();
    currentVideo = null;
  }
}

async function deleteCurrentVideo() {
  if (!currentVideo) return;
  if (!confirm(`Delete "${currentVideo.filename}"? This cannot be undone.`)) return;

  try {
    const resp = await fetch(`/api/saved/${encodeURIComponent(currentVideo.filename)}/delete`, {
      method: 'DELETE'
    });
    const data = await resp.json();
    if (data.success) {
      document.getElementById('savedModal').style.display = 'none';
      document.getElementById('savedVideo').pause();
      currentVideo = null;
      showToast('Video deleted.', 'success');
      loadSavedVideos();
    }
  } catch (e) {
    showToast('Failed to delete video.', 'error');
  }
}

function formatSavedDate(dateStr) {
  if (!dateStr) return '—';
  // Format: YYYYMMDD_HHMMSS
  const match = dateStr.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/);
  if (!match) return dateStr;
  return `${match[1]}-${match[2]}-${match[3]} ${match[4]}:${match[5]}`;
}

function showToast(message, type = 'info') {
  const colors = { success: '#22c55e', error: '#ef4444', info: '#3b82f6' };
  let container = document.getElementById('toastContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toastContainer';
    container.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:8px';
    document.body.appendChild(container);
  }
  const toast = document.createElement('div');
  toast.style.cssText = `background:#1e2135;border:1px solid ${colors[type]}40;border-left:3px solid ${colors[type]};color:#f0f2ff;padding:12px 16px;border-radius:8px;font-size:13px;max-width:320px;box-shadow:0 8px 24px rgba(0,0,0,.5);cursor:pointer;`;
  toast.textContent = message;
  toast.onclick = () => toast.remove();
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}
