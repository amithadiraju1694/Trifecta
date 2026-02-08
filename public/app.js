// Captured source (camera/screen) is attached here.
const video = document.getElementById('sourceVideo');
// Rendered output canvas (this is what OBS Browser Source displays).
const canvas = document.getElementById('output');
const ctx = canvas.getContext('2d');
// Offscreen canvas used for render composition.
const frameCanvas = document.createElement('canvas');
const frameCtx = frameCanvas.getContext('2d');
// Offscreen canvas for video-only layer (used for masking background replacement).
const videoLayerCanvas = document.createElement('canvas');
const videoLayerCtx = videoLayerCanvas.getContext('2d');
// Offscreen canvas used for compressing frames before sending to ML.
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

const sourceSelect = document.getElementById('sourceSelect');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const toggleFaces = document.getElementById('toggleFaces');
const toggleBg = document.getElementById('toggleBg');
const toggleText = document.getElementById('toggleText');
const connStatus = document.getElementById('connStatus');
const sentCount = document.getElementById('sentCount');
const recvCount = document.getElementById('recvCount');
const latencyEl = document.getElementById('latency');

// Default client config values (overridden by config.yaml via /config.json).
const defaultClientConfig = {
  sampling_ms: 150,
  max_in_flight: 2,
  image: {
    format: 'jpeg',
    jpeg_quality: 0.6,
    max_width: 640,
    max_height: 640
  }
};

let clientConfig = JSON.parse(JSON.stringify(defaultClientConfig));

// WebSocket connection to ML endpoint (mock server).
let socket;
let stream;
let running = false;
let inference = { faces: [], texts: [], seg: null };
let sendCounter = 0;
let recvCounter = 0;
let frameId = 0;
let lastSendAt = 0;
let sending = false;
const pendingFrames = new Set();

// Latest decoded segmentation mask (as an alpha mask for FOREGROUND).
// This is updated only when new inference arrives.
let segMaskCanvas = null;

// Sampling interval (ms) loaded from config.yaml via /config.json.
let sampleIntervalMs = clientConfig.sampling_ms;
let maxInFlight = clientConfig.max_in_flight;
let imageFormat = clientConfig.image.format;
let jpegQuality = clientConfig.image.jpeg_quality;
let maxWidth = clientConfig.image.max_width;
let maxHeight = clientConfig.image.max_height;

function applyClientConfig(overrides) {
  const merged = {
    ...defaultClientConfig,
    ...overrides,
    image: {
      ...defaultClientConfig.image,
      ...(overrides && overrides.image ? overrides.image : {})
    }
  };
  clientConfig = merged;
  sampleIntervalMs = Number(merged.sampling_ms);
  maxInFlight = Number(merged.max_in_flight);
  imageFormat = merged.image.format;
  jpegQuality = Number(merged.image.jpeg_quality);
  maxWidth = Number(merged.image.max_width);
  maxHeight = Number(merged.image.max_height);
}

async function loadConfig() {
  try {
    const res = await fetch('/config.json', { cache: 'no-store' });
    if (!res.ok) return;
    const data = await res.json();
    applyClientConfig(data);
  } catch (err) {
    // Fallback to defaults if config can't be loaded.
  }
}

// Creates the WebSocket connection to the ML endpoint.
// CONFIGURATION: this uses the same host/port as the web page.
// If you move ML to another server, change this URL.
function connectSocket() {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${protocol}://${location.host}`);

  socket.addEventListener('open', () => {
    connStatus.textContent = 'Connected to ML';
  });

  socket.addEventListener('close', () => {
    connStatus.textContent = 'Disconnected';
    pendingFrames.clear();
  });

  socket.addEventListener('message', (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (err) {
      return;
    }
    if (msg.type === 'inference') {
      // If a segmentation mask arrives, decode it once and cache as a canvas.
      const segFormat = msg.seg && msg.seg.format ? String(msg.seg.format).toLowerCase() : '';
      if (segFormat === 'packbits' && msg.seg.data_b64) {
        segMaskCanvas = decodePackbitsToForegroundMaskCanvas(
          msg.seg.data_b64,
          msg.seg.is_background_mask === true
        );
      }

      inference = {
        faces: msg.faces || [],
        texts: msg.texts || [],
        seg: msg.seg || null
      };
      if (msg.id) {
        pendingFrames.delete(msg.id);
      }
      recvCounter += 1;
      recvCount.textContent = recvCounter;
      latencyEl.textContent = `${msg.latencyMs || 0}ms`;
    }
  });
}

function updateCanvasSize() {
  const rawWidth = video.videoWidth || 1280;
  const rawHeight = video.videoHeight || 720;
  const maxWidth = 1280;
  const scale = Math.min(1, maxWidth / rawWidth);
  const renderWidth = Math.max(1, Math.floor(rawWidth * scale));
  const renderHeight = Math.max(1, Math.floor(rawHeight * scale));

  canvas.width = renderWidth;
  canvas.height = renderHeight;
  frameCanvas.width = renderWidth;
  frameCanvas.height = renderHeight;
  videoLayerCanvas.width = renderWidth;
  videoLayerCanvas.height = renderHeight;
}

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) {
    out[i] = bin.charCodeAt(i);
  }
  return out;
}

function readU16BE(bytes, offset) {
  return (bytes[offset] << 8) | bytes[offset + 1];
}

// This decodes the packbits and applies a slight blur to the mask itself.
// This creates the soft "Teams" edge around the person.
function decodePackbitsToForegroundMaskCanvas(dataB64, isBackgroundMask) {
  try {
    const bytes = b64ToBytes(dataB64);
    if (bytes.length < 6) return null;

    const h = readU16BE(bytes, 0);
    const w = readU16BE(bytes, 2);
    const rowStride = readU16BE(bytes, 4);
    const packed = bytes.subarray(6);

    // Basic validation
    if (h === 0 || w === 0 || rowStride === 0) return null;

    // Create a temporary buffer for the raw mask
    const rawMask = document.createElement('canvas');
    rawMask.width = w;
    rawMask.height = h;
    const rawCtx = rawMask.getContext('2d');
    const img = rawCtx.createImageData(w, h);
    const out = img.data;

    for (let y = 0; y < h; y += 1) {
      const rowStart = y * rowStride;
      for (let xb = 0; xb < rowStride; xb += 1) {
        // Safety check for buffer boundaries
        if (rowStart + xb >= packed.length) break;
        
        const b = packed[rowStart + xb];
        for (let bit = 0; bit < 8; bit += 1) {
          const x = xb * 8 + bit;
          if (x >= w) break;
          
          const isOne = (b & (1 << (7 - bit))) !== 0;
          // Determine if this pixel is Background
          // If ML sends is_background_mask=true, then 1=BG, 0=FG
          const isBg = isBackgroundMask ? isOne : !isOne;
          
          // We want the MASK to be OPAQUE at the FOREGROUND
          const alpha = isBg ? 0 : 255;
          
          const idx = (y * w + x) * 4;
          out[idx] = 0;     // R (irrelevant for mask, but keep black)
          out[idx + 1] = 0; // G
          out[idx + 2] = 0; // B
          out[idx + 3] = alpha; // Alpha determines visibility
        }
      }
    }
    
    rawCtx.putImageData(img, 0, 0);

    // PROCESS: Feather the mask
    // We draw the raw mask onto a final canvas with a slight blur.
    // This removes jagged edges.
    const featherCanvas = document.createElement('canvas');
    featherCanvas.width = w;
    featherCanvas.height = h;
    const featherCtx = featherCanvas.getContext('2d');
    
    featherCtx.save();
    // 4px blur creates a soft edge transition
    featherCtx.filter = 'blur(4px)'; 
    featherCtx.drawImage(rawMask, 0, 0);
    featherCtx.restore();

    return featherCanvas;
  } catch (err) {
    console.error("Mask Decode Error:", err);
    return null;
  }
}

// Flags choose which ML outputs to request.
function getFlags() {
  return {
    run_face: toggleFaces.checked,
    run_seg: toggleBg.checked,
    run_text: toggleText.checked
  };
}

// Starts camera or screen capture.
async function startCapture() {
  if (running) return;

  const source = sourceSelect.value;
  if (source === 'camera') {
    // Camera capture.
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
  } else {
    // Screen capture.
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: false
    });
  }

  video.srcObject = stream;
  await video.play();
  updateCanvasSize();
  running = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
}

function stopCapture() {
  if (!stream) return;
  stream.getTracks().forEach((track) => track.stop());
  stream = null;
  running = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

// Decorative background used when background replacement is ON.
function drawBackground(ctx2d, width, height) {
  const gradient = ctx2d.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, '#10272f');
  gradient.addColorStop(0.5, '#195e56');
  gradient.addColorStop(1, '#1b3440');
  ctx2d.fillStyle = gradient;
  ctx2d.fillRect(0, 0, width, height);

  ctx2d.fillStyle = 'rgba(255, 122, 89, 0.25)';
  ctx2d.beginPath();
  ctx2d.arc(width * 0.2, height * 0.25, Math.min(width, height) * 0.18, 0, Math.PI * 2);
  ctx2d.fill();
}

// Converts normalized box values (0-1) to pixel values.
function boxToRect(box, width, height) {
  const x = Math.max(0, box.x * width);
  const y = Math.max(0, box.y * height);
  const w = Math.max(0, box.w * width);
  const h = Math.max(0, box.h * height);
  return { x, y, w, h };
}

function drawBox(ctx2d, box, width, height, color, fill = false) {
  const { x, y, w, h } = boxToRect(box, width, height);
  if (fill) {
    ctx2d.fillStyle = color;
    ctx2d.fillRect(x, y, w, h);
  }
  ctx2d.strokeStyle = color;
  ctx2d.lineWidth = 3;
  ctx2d.strokeRect(x, y, w, h);
  return { x, y, w, h };
}

// Sends one compressed frame to the ML endpoint.
// IMPORTANT: this is the line path that calls the ML backend.
// Data type: a base64-encoded JPEG string derived from the video frame.
// Sends one sampled frame to the ML backend over WebSocket.
// This is the primary ML inference request path.
async function sendFrame() {
  if (sending || !socket || socket.readyState !== WebSocket.OPEN) return;
  const flags = getFlags();
  if (!flags.run_face && !flags.run_seg && !flags.run_text) return;
  if (!video.videoWidth || !video.videoHeight) return;
  // Avoid piling up requests: respect max in-flight.
  if (pendingFrames.size >= maxInFlight) return;

  sending = true;
  lastSendAt = Date.now();
  const baseWidth = canvas.width || maxWidth;
  const baseHeight = canvas.height || maxHeight;
  const scale = Math.min(1, maxWidth / baseWidth, maxHeight / baseHeight);
  const targetWidth = Math.max(1, Math.floor(baseWidth * scale));
  const targetHeight = Math.max(1, Math.floor(baseHeight * scale));

  captureCanvas.width = targetWidth;
  captureCanvas.height = targetHeight;
  // Draw current video frame into the capture canvas.
  captureCtx.drawImage(video, 0, 0, targetWidth, targetHeight);

  const format = (imageFormat || 'jpeg').toLowerCase();
  const mimeType = format === 'png' ? 'image/png' : 'image/jpeg';
  const quality = format === 'png' ? undefined : jpegQuality;

  // Convert the frame to JPEG/PNG (compressed).
  const blob = await new Promise((resolve) =>
    captureCanvas.toBlob(resolve, mimeType, quality)
  );

  if (!blob) {
    sending = false;
    return;
  }

  // Blob -> ArrayBuffer -> base64 string for transport.
  const arrayBuffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  // THIS is the input image data type sent to ML: base64 JPEG/PNG string.
  const image = btoa(binary);

  // ML request payload.
  const message = {
    type: 'frame',
    id: ++frameId,
    ts: Date.now(),
    width: targetWidth,
    height: targetHeight,
    flags,
    image_format: format,
    image
  };

  // Actual send to ML backend happens here (WebSocket JSON payload).
  socket.send(JSON.stringify(message));
  pendingFrames.add(message.id);
  sendCounter += 1;
  sentCount.textContent = sendCounter;
  sending = false;
}

// Handles the "Frosted Glass" Background + Sharp Foreground composition
function render() {
  requestAnimationFrame(render);
  if (!running || video.readyState < 2) return;

  const width = canvas.width;
  const height = canvas.height;
  if (!width || !height) return;

  const flags = getFlags();

  // 1. Clear Main Canvas
  ctx.clearRect(0, 0, width, height);

  if (flags.run_seg) {
    // --- STEP A: Draw The Blurred Background ---
    frameCtx.clearRect(0, 0, width, height);
    
    frameCtx.save();
    // Heavy blur (30px) makes objects "invisible yet not knowing shape" (Frosted Glass)
    // We scale slightly (1.05) to hide edge artifacts at the borders
    frameCtx.filter = 'blur(30px)'; 
    const scale = 1.05;
    const sw = width * scale;
    const sh = height * scale;
    frameCtx.drawImage(video, (width - sw) / 2, (height - sh) / 2, sw, sh);
    frameCtx.restore();

    // --- STEP B: Draw The Sharp Foreground (If Mask Exists) ---
    if (segMaskCanvas) {
      // Prepare the Video Layer (Sharp)
      videoLayerCtx.clearRect(0, 0, width, height);
      videoLayerCtx.drawImage(video, 0, 0, width, height);

      // Composite the Mask: Keep Sharp Video ONLY where Mask is Opaque
      videoLayerCtx.save();
      videoLayerCtx.globalCompositeOperation = 'destination-in';
      // Draw mask stretched to fit current resolution
      videoLayerCtx.drawImage(segMaskCanvas, 0, 0, width, height);
      videoLayerCtx.restore();

      // Draw the Cut-out Foreground on top of the Blurred Background
      frameCtx.drawImage(videoLayerCanvas, 0, 0, width, height);
    } 
    else {
      // Fallback: If Seg is ON but no mask yet, show full blur (privacy mode)
      // or optionally show nothing. Currently shows full blur.
    }

    // Output to main display
    ctx.drawImage(frameCanvas, 0, 0, width, height);

  } else {
    // Standard Pass-through (No Blur)
    ctx.drawImage(video, 0, 0, width, height);
  }

  // Draw Overlays (Faces/Text) - Kept from your original logic
  if (flags.run_face && inference.faces.length) {
    inference.faces.forEach((face) => {
      // Optional: Blur faces extra if requested, or just draw box
      drawBox(ctx, face, width, height, 'rgba(255, 122, 89, 0.9)');
    });
  }

  if (flags.run_text && inference.texts.length) {
    inference.texts.forEach((textBox) => {
      drawBox(ctx, textBox, width, height, 'rgba(255, 214, 0, 0.9)', true);
    });
  }

  // Sampling gate: only send frames every sampleIntervalMs.
  if (Date.now() - lastSendAt > sampleIntervalMs) {
    sendFrame();
  }
}



startBtn.addEventListener('click', () => {
  startCapture().catch((err) => {
    connStatus.textContent = `Capture error: ${err.message}`;
  });
});

stopBtn.addEventListener('click', () => {
  stopCapture();
});

loadConfig().finally(() => {
  connectSocket();
  render();
});
