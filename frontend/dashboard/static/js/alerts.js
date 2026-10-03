/**
 * alerts.js — Alerts page logic
 * Features: filter, search, mark-as-read badge, video check, save to Saved Videos
 */

'use strict';

const STORAGE_KEY = 'fireguard_read_alert_ids';

let allAlerts = [];
let currentFilter = 'all';
let currentSearchTerm = '';
let currentAlertId = null;

document.addEventListener('DOMContentLoaded', loadAlerts);

// ─── Read State ────────────────────────────────────────────────────────────────
function getReadIds() {
  try { return new Set(JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')); }
  catch { return new Set(); }
}
function saveReadIds(readSet) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...readSet]));
}
function getUnreadCount() {
  const readIds = getReadIds();
  return allAlerts.filter(a => a.is_abnormal && !readIds.has(a.id)).length;
}

function markAllRead() {
  const readIds = getReadIds();
  allAlerts.filter(a => a.is_abnormal).forEach(a => readIds.add(a.id));
  saveReadIds(readIds);
  updateUnreadUI();
  renderAlerts();
  showToast('All alerts marked as read.', 'success');
}

function updateUnreadUI() {
  const unread = getUnreadCount();
  const badge = document.getElementById('alertBadge');
  const btn   = document.getElementById('markReadBtn');
  if (badge) {
    badge.textContent    = unread > 0 ? unread : '';
    badge.style.display  = unread > 0 ? 'inline-block' : 'none';
  }
  if (btn) btn.style.display = unread > 0 ? 'inline-flex' : 'none';
}

// ─── Load ─────────────────────────────────────────────────────────────────────
async function loadAlerts() {
  const container = document.getElementById('alertsContainer');
  try {
    const resp = await fetch('/api/alerts');
    allAlerts = await resp.json();
    renderAlerts();
    updateUnreadUI();
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><p>Failed to load alerts. Please refresh.</p></div>`;
  }
}

// ─── Render ────────────────────────────────────────────────────────────────────
function renderAlerts() {
  const container = document.getElementById('alertsContainer');
  const countEl   = document.getElementById('alertsCount');
  const readIds   = getReadIds();

  let filtered = allAlerts;
  if (currentFilter === 'abnormal') filtered = allAlerts.filter(a => a.is_abnormal);
  else if (currentFilter === 'normal') filtered = allAlerts.filter(a => !a.is_abnormal);

  if (currentSearchTerm) {
    const term = currentSearchTerm.toLowerCase();
    filtered = filtered.filter(a =>
      (a.description || '').toLowerCase().includes(term) ||
      (a.event_type  || '').toLowerCase().includes(term) ||
      (a.camera_id   || '').toLowerCase().includes(term) ||
      (a.severity    || '').toLowerCase().includes(term)
    );
  }

  countEl.textContent = `${filtered.length} alert${filtered.length !== 1 ? 's' : ''}`;

  if (!filtered.length) {
    container.innerHTML = `
      <div class="empty-state" style="grid-column:1/-1">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9"/>
          <path d="M13.73 21a2 2 0 01-3.46 0"/>
        </svg>
        <p>${allAlerts.length === 0 ? 'No alerts yet. Upload a video to start detection.' : 'No alerts match your filter.'}</p>
      </div>`;
    return;
  }

  container.innerHTML = filtered.map(a => alertCardHTML(a, readIds)).join('');
}

function alertCardHTML(alert, readIds) {
  const isAbn    = alert.is_abnormal;
  const isUnread = isAbn && !readIds.has(alert.id);
  const sevClass = { high: 'sev-high', medium: 'sev-medium', low: 'sev-low' }[alert.severity] || 'sev-low';
  const icon     = isAbn ? '🚨' : '✅';
  const conf     = Math.round((alert.confidence || 0) * 100);
  const unreadDot = isUnread
    ? '<span style="width:8px;height:8px;background:#ef4444;border-radius:50%;display:inline-block;margin-right:5px;flex-shrink:0;animation:pulse 1.5s infinite"></span>'
    : '';

  return `
    <div class="alert-card alert-card--${isAbn ? 'abnormal' : 'normal'}${isUnread ? ' alert-card--unread' : ''}" onclick="openAlert('${alert.id}')">
      <div class="alert-card-header">
        <div class="alert-card-type" style="display:flex;align-items:center">
          ${unreadDot}<span class="alert-card-icon">${icon}</span>
          ${(alert.event_type || 'unknown').toUpperCase()}
        </div>
        <span class="severity-badge ${sevClass}">${alert.severity || '—'}</span>
      </div>
      <div class="alert-card-body">
        <div class="alert-card-desc">${alert.description || 'No description available.'}</div>
        <div class="alert-card-meta">
          <span class="alert-card-camera">
            <svg style="width:12px;height:12px;margin-right:3px" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/>
            </svg>
            ${alert.camera_id || '—'}
          </span>
          <span>${conf}% confidence</span>
        </div>
      </div>
      <div class="alert-card-footer">
        ${alert.has_clip
          ? '<span style="font-size:11px;color:var(--teal-400)">📹 Video available</span>'
          : '<span style="font-size:11px;color:#ef444480">⚠️ No video clip</span>'}
        <span style="margin-left:auto;font-size:11px;color:var(--text-muted)">${formatTimestamp(alert.id)}</span>
      </div>
    </div>`;
}

// ─── Filter / Search ──────────────────────────────────────────────────────────
function filterAlerts(type) {
  currentFilter = type;
  document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
  document.getElementById(`filter${type.charAt(0).toUpperCase() + type.slice(1)}`).classList.add('active');
  renderAlerts();
}

function searchAlerts() {
  currentSearchTerm = document.getElementById('alertSearch').value;
  renderAlerts();
}

// ─── Alert Modal ───────────────────────────────────────────────────────────────
function openAlert(alertId) {
  const alert = allAlerts.find(a => a.id === alertId);
  if (!alert) return;

  // Mark as read
  const readIds = getReadIds();
  readIds.add(alertId);
  saveReadIds(readIds);
  updateUnreadUI();
  renderAlerts();

  currentAlertId = alertId;
  const isAbn    = alert.is_abnormal;
  const sevClass = { high: 'sev-high', medium: 'sev-medium', low: 'sev-low' }[alert.severity] || 'sev-low';

  // Header
  document.getElementById('modalHeader').innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
      <span style="font-size:28px">${isAbn ? '🚨' : '✅'}</span>
      <div>
        <h2 style="font-size:18px;font-weight:700;color:var(--text-primary);margin-bottom:2px">
          ${isAbn ? 'Abnormal Event' : 'Normal Event'}
        </h2>
        <p style="font-size:13px;color:var(--text-secondary)">${(alert.event_type || 'unknown').toUpperCase()} · ${alert.camera_id}</p>
      </div>
      <span class="severity-badge ${sevClass}" style="margin-left:auto">${alert.severity}</span>
    </div>`;

  // Video — show warning if no clip
  const videoWrap  = document.getElementById('modalVideoWrap');
  const modalVideo = document.getElementById('modalVideo');
  // Clear old warnings
  videoWrap.querySelectorAll('.video-missing-warn').forEach(el => el.remove());
  modalVideo.style.display = '';

  if (alert.has_clip) {
    modalVideo.src = `/api/alerts/${alertId}/video`;
    videoWrap.style.display = 'block';
  } else {
    videoWrap.style.display = 'block';
    modalVideo.src = '';
    modalVideo.style.display = 'none';
    const warn = document.createElement('div');
    warn.className = 'video-missing-warn';
    warn.style.cssText = 'padding:14px 16px;background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.3);border-radius:8px;color:#f87171;font-size:13px;display:flex;align-items:center;gap:10px;margin-bottom:8px';
    warn.innerHTML = `<svg viewBox="0 0 20 20" fill="currentColor" style="width:18px;height:18px;flex-shrink:0"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg><span>No 20-second video clip was saved for this alert. The clip may not have been captured or was cleaned up.</span>`;
    videoWrap.insertBefore(warn, modalVideo);
  }

  // Details grid
  document.getElementById('modalDetails').innerHTML = `
    <div class="detail-item">
      <div class="detail-label">Confidence</div>
      <div class="detail-value">${Math.round((alert.confidence || 0) * 100)}%</div>
    </div>
    <div class="detail-item">
      <div class="detail-label">Analyzed by</div>
      <div class="detail-value">${alert.router || '—'}</div>
    </div>
    <div class="detail-item">
      <div class="detail-label">Object ID</div>
      <div class="detail-value">${alert.object_id || '—'}</div>
    </div>
    <div class="detail-item">
      <div class="detail-label">Suppressed</div>
      <div class="detail-value">${alert.suppressed ? '⚠️ Yes (ongoing)' : 'No'}</div>
    </div>
    <div class="detail-item detail-desc">
      <div class="detail-label">Description</div>
      <div class="detail-value" style="margin-top:4px">${alert.description || 'No description available.'}</div>
    </div>
    <div class="detail-item detail-desc">
      <div class="detail-label">Reasoning</div>
      <div class="detail-value" style="margin-top:4px">${alert.reasoning || '—'}</div>
    </div>`;

  // Message
  const msgWrap = document.getElementById('modalMessageWrap');
  if (alert.message) {
    document.getElementById('modalMessage').textContent = alert.message;
    msgWrap.style.display = 'block';
  } else {
    msgWrap.style.display = 'none';
  }

  // Footer
  document.getElementById('modalFooter').innerHTML = alert.has_clip
    ? `<button class="btn-primary" id="saveVideoBtn" onclick="saveAlertVideo('${alertId}')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/></svg>
        Save to Saved Videos
      </button>`
    : `<span style="font-size:12px;color:var(--text-muted);font-style:italic">Video clip unavailable — cannot save.</span>`;

  document.getElementById('alertModal').style.display = 'flex';
}

function closeModal(event) {
  if (event.target === document.getElementById('alertModal')) {
    document.getElementById('alertModal').style.display = 'none';
    document.getElementById('modalVideo').pause();
  }
}

async function saveAlertVideo(alertId) {
  const btn = document.getElementById('saveVideoBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }

  try {
    const resp = await fetch(`/api/alerts/${alertId}/save`, { method: 'POST' });
    const data = await resp.json();
    if (data.success) {
      showToast('✅ Saved! Check the Saved Videos tab.', 'success');
      if (btn) {
        btn.innerHTML = '✅ Saved!';
        btn.style.cssText += 'background:rgba(34,197,94,.15);border-color:#22c55e;';
      }
    } else {
      showToast('Save failed: ' + (data.error || 'unknown error'), 'error');
      if (btn) { btn.disabled = false; btn.textContent = 'Save to Saved Videos'; }
    }
  } catch (e) {
    showToast('Network error saving video.', 'error');
    if (btn) { btn.disabled = false; btn.textContent = 'Save to Saved Videos'; }
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function formatTimestamp(alertId) {
  const match = alertId.match(/(\d{8})_(\d{6})/);
  if (!match) return alertId;
  const d = match[1], t = match[2];
  return `${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6,8)} ${t.slice(0,2)}:${t.slice(2,4)}`;
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
  toast.style.cssText = `background:#1e2135;border:1px solid ${colors[type]}40;border-left:3px solid ${colors[type]};color:#f0f2ff;padding:12px 16px;border-radius:8px;font-size:13px;max-width:340px;box-shadow:0 8px 24px rgba(0,0,0,.5);cursor:pointer;`;
  toast.textContent = message;
  toast.onclick = () => toast.remove();
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
}
