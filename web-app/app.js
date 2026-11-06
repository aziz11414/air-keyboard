/**
 * Air Writer Web Client - MediaPipe Hands JS
 * Supports:
 * - Velocity-based calligraphy brush
 * - Neon glow with canvas shadowBlur
 * - Brushes: Neon, Calligraphy, Rainbow, Highlighter
 * - Backgrounds: Camera, Clean Whiteboard, Dark Grid
 * - Hand gestures: 1s Timer to Write, Open Palm Eraser, Thumbs Up Undo
 * - High-Res PNG snapshot download
 */

const videoElement = document.getElementById('webcam');
const bgCanvas = document.getElementById('bg-canvas');
const inkCanvas = document.getElementById('ink-canvas');
const hudCanvas = document.getElementById('hud-canvas');

const bgCtx = bgCanvas.getContext('2d');
const inkCtx = inkCanvas.getContext('2d');
const hudCtx = hudCanvas.getContext('2d');

const gestureBadge = document.getElementById('gesture-badge');
const toastEl = document.getElementById('toast');

// State
let currentColor = '#00f0ff';
let currentBrush = 'NEON'; // 'NEON', 'CALLIGRAPHY', 'RAINBOW', 'HIGHLIGHTER'
let currentBgMode = 'WEBCAM'; // 'WEBCAM', 'WHITEBOARD', 'DARK_SLATE'
let rainbowHue = 0;

// Stroke History for Undo
const strokeHistory = [];
let currentStroke = [];

// Index Finger 1s Timer state
let indexStartTime = null;
let isWritingActive = false;
let prevPoint = null;
let currentThickness = 6;
let gestureCooldownUntil = 0;

// Resize canvases to window
function resizeCanvases() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  bgCanvas.width = w;
  bgCanvas.height = h;
  inkCanvas.width = w;
  inkCanvas.height = h;
  hudCanvas.width = w;
  hudCanvas.height = h;
  redrawAllStrokes();
}
window.addEventListener('resize', resizeCanvases);

// Toast notification
function showToast(text, duration = 1800) {
  toastEl.textContent = text;
  toastEl.style.opacity = '1';
  setTimeout(() => {
    toastEl.style.opacity = '0';
  }, duration);
}

// ─── STROKE RENDERING ───────────────────────────────────────────────
function drawSegment(ctx, p1, p2, color, thickness, brush) {
  ctx.save();
  if (brush === 'NEON') {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = thickness;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = color;
    ctx.shadowBlur = 15;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  } else if (brush === 'HIGHLIGHTER') {
    ctx.strokeStyle = color;
    ctx.lineWidth = 22;
    ctx.lineCap = 'square';
    ctx.globalAlpha = 0.35;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  } else if (brush === 'CALLIGRAPHY') {
    ctx.strokeStyle = color;
    ctx.lineWidth = thickness;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  } else if (brush === 'RAINBOW') {
    rainbowHue = (rainbowHue + 4) % 360;
    const rColor = `hsl(${rainbowHue}, 100%, 55%)`;
    ctx.strokeStyle = rColor;
    ctx.lineWidth = thickness;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }
  ctx.restore();
}

function redrawAllStrokes() {
  inkCtx.clearRect(0, 0, inkCanvas.width, inkCanvas.height);
  for (const s of strokeHistory) {
    const pts = s.points;
    for (let i = 1; i < pts.length; i++) {
      drawSegment(inkCtx, pts[i - 1], pts[i], s.color, s.thickness, s.brush);
    }
  }
}

// ─── GESTURE CLASSIFIER ─────────────────────────────────────────────
function classifyGesture(landmarks, w, h) {
  const pts = landmarks.map(lm => ({ x: (1 - lm.x) * w, y: lm.y * h })); // Mirrored X

  const indexUp = pts[8].y < pts[6].y;
  const middleUp = pts[12].y < pts[10].y;
  const ringUp = pts[16].y < pts[14].y;
  const pinkyUp = pts[20].y < pts[18].y;
  const thumbUp = pts[4].y < pts[2].y - 25;

  // Only 3 clear gestures:
  if (indexUp && middleUp && ringUp && pinkyUp) return { type: 'ERASE', pts };
  if (indexUp && middleUp && !ringUp && !pinkyUp) return { type: 'PEN_UP', pts };
  if (indexUp && !middleUp && !ringUp && !pinkyUp) return { type: 'INDEX_UP', pts };

  return { type: 'IDLE', pts };
}

// ─── MEDIAPIPE ONRESULTS CALLBACK ───────────────────────────────────
function onResults(results) {
  const w = bgCanvas.width;
  const h = bgCanvas.height;
  const now = performance.now() / 1000;

  // 1. Draw Background
  bgCtx.save();
  if (currentBgMode === 'WEBCAM') {
    // Draw mirrored video
    bgCtx.translate(w, 0);
    bgCtx.scale(-1, 1);
    bgCtx.drawImage(results.image, 0, 0, w, h);
  } else if (currentBgMode === 'WHITEBOARD') {
    bgCtx.fillStyle = '#fafafa';
    bgCtx.fillRect(0, 0, w, h);
  } else if (currentBgMode === 'DARK_SLATE') {
    bgCtx.fillStyle = '#111218';
    bgCtx.fillRect(0, 0, w, h);
    // Draw grid lines
    bgCtx.strokeStyle = '#222530';
    bgCtx.lineWidth = 1;
    for (let x = 0; x < w; x += 40) {
      bgCtx.beginPath();
      bgCtx.moveTo(x, 0);
      bgCtx.lineTo(x, h);
      bgCtx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
      bgCtx.beginPath();
      bgCtx.moveTo(0, y);
      bgCtx.lineTo(w, y);
      bgCtx.stroke();
    }
  }
  bgCtx.restore();

  // Clear HUD layer
  hudCtx.clearRect(0, 0, w, h);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const rawLm = results.multiHandLandmarks[0];
    const { type: gesture, pts } = classifyGesture(rawLm, w, h);

    const indexTip = pts[8];
    const cx = indexTip.x;
    const cy = indexTip.y;

    // Draw hand skeleton on HUD
    drawHandSkeleton(hudCtx, pts);

    // Update Gesture Status Badge
    updateBadge(gesture, isWritingActive);

    // ── 1. OPEN PALM: MANUAL SPONGE ERASER ──
    if (gesture === 'ERASE') {
      if (currentStroke.length > 0) {
        strokeHistory.push({
          points: [...currentStroke],
          color: currentColor,
          thickness: currentThickness,
          brush: currentBrush,
        });
        currentStroke = [];
      }
      indexStartTime = null;
      isWritingActive = false;
      prevPoint = null;

      // Palm center
      const palmX = (pts[0].x + pts[5].x + pts[9].x + pts[17].x) / 4;
      const palmY = (pts[0].y + pts[5].y + pts[9].y + pts[17].y) / 4;

      // Erase from ink canvas
      inkCtx.save();
      inkCtx.globalCompositeOperation = 'destination-out';
      inkCtx.beginPath();
      inkCtx.arc(palmX, palmY, 65, 0, Math.PI * 2);
      inkCtx.fill();
      inkCtx.restore();

      // Visual sponge on HUD
      hudCtx.save();
      hudCtx.beginPath();
      hudCtx.arc(palmX, palmY, 65, 0, Math.PI * 2);
      hudCtx.fillStyle = 'rgba(239, 68, 68, 0.35)';
      hudCtx.fill();
      hudCtx.lineWidth = 3;
      hudCtx.strokeStyle = '#ef4444';
      hudCtx.stroke();
      hudCtx.fillStyle = '#ffffff';
      hudCtx.font = 'bold 12px sans-serif';
      hudCtx.fillText('ERASER', palmX - 24, palmY + 4);
      hudCtx.restore();
    }

    // ── 3. INDEX UP: 1-SECOND TIMER TO WRITE ──
    else if (gesture === 'INDEX_UP') {
      if (indexStartTime === null) {
        indexStartTime = now;
        isWritingActive = false;
      }

      const elapsed = now - indexStartTime;

      if (elapsed >= 1.0) {
        // Active Writing
        isWritingActive = true;

        // Calligraphy velocity-based width
        if (currentBrush === 'CALLIGRAPHY' && prevPoint) {
          const dist = Math.hypot(cx - prevPoint.x, cy - prevPoint.y);
          const targetThick = Math.max(3, Math.min(14, 14 - dist / 4.5));
          currentThickness = currentThickness * 0.6 + targetThick * 0.4;
        } else if (currentBrush === 'HIGHLIGHTER') {
          currentThickness = 22;
        } else {
          currentThickness = 6;
        }

        // Draw cursor
        hudCtx.save();
        hudCtx.beginPath();
        hudCtx.arc(cx, cy, currentThickness + 4, 0, Math.PI * 2);
        hudCtx.fillStyle = currentColor;
        hudCtx.fill();
        hudCtx.lineWidth = 2;
        hudCtx.strokeStyle = '#ffffff';
        hudCtx.stroke();
        hudCtx.restore();

        currentStroke.push({ x: cx, y: cy });

        if (prevPoint) {
          drawSegment(inkCtx, prevPoint, { x: cx, y: cy }, currentColor, currentThickness, currentBrush);
        }
        prevPoint = { x: cx, y: cy };
      } else {
        // Countdown
        prevPoint = null;
        const progress = elapsed / 1.0;
        const remaining = (1.0 - elapsed).toFixed(1);

        hudCtx.save();
        hudCtx.beginPath();
        hudCtx.arc(cx, cy, 18, 0, Math.PI * 2);
        hudCtx.strokeStyle = '#4b5563';
        hudCtx.lineWidth = 2;
        hudCtx.stroke();

        hudCtx.beginPath();
        hudCtx.arc(cx, cy, 18, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
        hudCtx.strokeStyle = '#10b981';
        hudCtx.lineWidth = 3;
        hudCtx.stroke();

        hudCtx.fillStyle = '#10b981';
        hudCtx.font = 'bold 12px sans-serif';
        hudCtx.fillText(`${remaining}s`, cx + 24, cy + 4);
        hudCtx.restore();
      }
    } else {
      // Pen lifted
      if (currentStroke.length > 0) {
        strokeHistory.push({
          points: [...currentStroke],
          color: currentColor,
          thickness: currentThickness,
          brush: currentBrush,
        });
        currentStroke = [];
      }
      indexStartTime = null;
      isWritingActive = false;
      prevPoint = null;
    }
  } else {
    // Hand lost
    if (currentStroke.length > 0) {
      strokeHistory.push({
        points: [...currentStroke],
        color: currentColor,
        thickness: currentThickness,
        brush: currentBrush,
      });
      currentStroke = [];
    }
    indexStartTime = null;
    isWritingActive = false;
    prevPoint = null;
    gestureBadge.textContent = 'WAITING FOR HAND';
    gestureBadge.className = 'px-3 py-1 text-xs font-bold rounded border border-gray-700 bg-gray-900 text-gray-400';
  }
}

// ─── SKELETON DRAWER ────────────────────────────────────────────────
function drawHandSkeleton(ctx, pts) {
  const conns = [
    [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],
  ];
  ctx.save();
  ctx.strokeStyle = 'rgba(156, 163, 175, 0.4)';
  ctx.lineWidth = 1.5;
  for (const [s, e] of conns) {
    ctx.beginPath();
    ctx.moveTo(pts[s].x, pts[s].y);
    ctx.lineTo(pts[e].x, pts[e].y);
    ctx.stroke();
  }
  ctx.fillStyle = '#00ff80';
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function updateBadge(gesture, isWriting) {
  if (isWriting) {
    gestureBadge.textContent = '✍️ WRITING ACTIVE';
    gestureBadge.className = 'px-3 py-1 text-xs font-bold rounded border border-emerald-500/50 bg-emerald-950/50 text-emerald-400';
  } else if (gesture === 'INDEX_UP') {
    gestureBadge.textContent = '⏱️ COUNTDOWN (1.0s)';
    gestureBadge.className = 'px-3 py-1 text-xs font-bold rounded border border-cyan-500/50 bg-cyan-950/50 text-cyan-300';
  } else if (gesture === 'PEN_UP') {
    gestureBadge.textContent = '✌️ PEN UP (MOVE)';
    gestureBadge.className = 'px-3 py-1 text-xs font-bold rounded border border-amber-500/50 bg-amber-950/50 text-amber-300';
  } else if (gesture === 'ERASE') {
    gestureBadge.textContent = '🖐️ ERASER (SWIPING)';
    gestureBadge.className = 'px-3 py-1 text-xs font-bold rounded border border-red-500/50 bg-red-950/50 text-red-400';
  } else if (gesture === 'THUMBS_UP') {
    gestureBadge.textContent = '👍 UNDO';
    gestureBadge.className = 'px-3 py-1 text-xs font-bold rounded border border-blue-500/50 bg-blue-950/50 text-blue-300';
  } else {
    gestureBadge.textContent = 'READY';
    gestureBadge.className = 'px-3 py-1 text-xs font-bold rounded border border-gray-700 bg-gray-900 text-gray-300';
  }
}

// ─── UI CONTROLS WIRING ─────────────────────────────────────────────
// Colors
document.querySelectorAll('.color-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('ring-2', 'ring-white'));
    btn.classList.add('ring-2', 'ring-white');
    currentColor = btn.getAttribute('data-color');
    showToast(`🎨 Color: ${currentColor}`);
  });
});

// Brushes
const brushButtons = {
  NEON: document.getElementById('btn-brush-neon'),
  CALLIGRAPHY: document.getElementById('btn-brush-calligraphy'),
  RAINBOW: document.getElementById('btn-brush-rainbow'),
  HIGHLIGHTER: document.getElementById('btn-brush-highlighter'),
};
Object.keys(brushButtons).forEach(key => {
  brushButtons[key].addEventListener('click', () => {
    Object.values(brushButtons).forEach(b => {
      b.className = 'px-3 py-1 text-xs font-semibold rounded text-gray-400 hover:text-white transition';
    });
    brushButtons[key].className = 'px-3 py-1 text-xs font-semibold rounded bg-cyan-500 text-black transition';
    currentBrush = key;
    showToast(`🖌️ Brush: ${key}`);
  });
});

// Background Modes
const bgButtons = {
  WEBCAM: document.getElementById('btn-bg-cam'),
  WHITEBOARD: document.getElementById('btn-bg-white'),
  DARK_SLATE: document.getElementById('btn-bg-dark'),
};
Object.keys(bgButtons).forEach(key => {
  bgButtons[key].addEventListener('click', () => {
    Object.values(bgButtons).forEach(b => {
      b.className = 'px-3 py-1 text-xs font-medium rounded text-gray-400 hover:text-white transition';
    });
    bgButtons[key].className = 'px-3 py-1 text-xs font-medium rounded bg-gray-800 text-white transition';
    currentBgMode = key;
    showToast(`📄 Board Mode: ${key}`);
  });
});

// Undo
document.getElementById('btn-undo').addEventListener('click', () => {
  if (strokeHistory.length > 0) {
    strokeHistory.pop();
    redrawAllStrokes();
    showToast('↩️ Undo');
  }
});

// Clear
document.getElementById('btn-clear').addEventListener('click', () => {
  strokeHistory.length = 0;
  redrawAllStrokes();
  showToast('Canvas Cleared');
});

// Snapshot
document.getElementById('btn-snapshot').addEventListener('click', () => {
  const merged = document.createElement('canvas');
  merged.width = bgCanvas.width;
  merged.height = bgCanvas.height;
  const mCtx = merged.getContext('2d');
  mCtx.drawImage(bgCanvas, 0, 0);
  mCtx.drawImage(inkCanvas, 0, 0);

  const link = document.createElement('a');
  link.download = `air_writer_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.png`;
  link.href = merged.toDataURL('image/png');
  link.click();
  showToast('📸 Snapshot Downloaded!');
});

// ─── INITIALIZATION ─────────────────────────────────────────────────
resizeCanvases();

// Setup MediaPipe Hands
const hands = new Hands({
  locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.65,
  minTrackingConfidence: 0.65,
});

hands.onResults(onResults);

// Setup Camera
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 1280,
  height: 720,
});

camera.start().then(() => {
  showToast('📷 Camera Active & AI Ready!');
}).catch(err => {
  console.error('Camera error:', err);
  showToast('❌ Camera Permission Denied', 4000);
});
