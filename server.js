const path = require('path');
const http = require('http');
const fs = require('fs');
const express = require('express');
const WebSocket = require('ws');
const YAML = require('yaml');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PORT = process.env.PORT || 3000;
const CONFIG_PATH = process.env.TRIFECTA_CONFIG || path.join(__dirname, 'config.yaml');

function mergeDeep(base, override) {
  if (!override) return base;
  const output = Array.isArray(base) ? [...base] : { ...base };
  Object.keys(override).forEach((key) => {
    const value = override[key];
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      output[key] = mergeDeep(base[key] || {}, value);
    } else if (value !== undefined) {
      output[key] = value;
    }
  });
  return output;
}

function loadConfig() {
  const defaults = {
    client: {
      sampling_ms: 150,
      max_in_flight: 2,
      image: {
        format: 'jpeg',
        jpeg_quality: 0.6,
        max_width: 640,
        max_height: 640
      }
    },
    backend: {
      base_url: 'https://amithadiraju1694-trifecta-backend.hf.space',
      endpoints: {
        seg: '/run_segmentation',
        face: '/run_facemask',
        text: '/run_text'
      },
      timeout_ms: 1500,
      max_concurrent_calls: 6,
      use_mock: false
    }
  };

  try {
    const raw = fs.readFileSync(CONFIG_PATH, 'utf8');
    const parsed = YAML.parse(raw);
    return mergeDeep(defaults, parsed);
  } catch (err) {
    return defaults;
  }
}

const config = loadConfig();
const clientConfig = config.client || {};
const backendConfig = config.backend || {};

const HF_BASE_URL = process.env.HF_BASE_URL || backendConfig.base_url;
const HF_ENDPOINTS = backendConfig.endpoints || {
  seg: '/run_segmentation',
  face: '/run_facemask',
  text: '/run_text'
};
const HF_TIMEOUT_MS = Number(process.env.HF_TIMEOUT_MS || backendConfig.timeout_ms || 1500);
const HF_MAX_CONCURRENT = Number(
  process.env.HF_MAX_CONCURRENT || backendConfig.max_concurrent_calls || 6
);
const USE_MOCK =
  (process.env.USE_MOCK || '').toLowerCase() === 'true' || backendConfig.use_mock === true;

function createLimiter(maxConcurrent) {
  const max = Math.max(1, maxConcurrent || 1);
  let active = 0;
  const queue = [];

  const runNext = () => {
    if (active >= max || queue.length === 0) return;
    const job = queue.shift();
    if (!job) return;
    active += 1;
    job()
      .catch(() => {})
      .finally(() => {
        active -= 1;
        runNext();
      });
  };

  return (fn) =>
    new Promise((resolve, reject) => {
      const task = async () => {
        try {
          const result = await fn();
          resolve(result);
        } catch (err) {
          reject(err);
        }
      };
      queue.push(task);
      runNext();
    });
}

const hfLimiter = createLimiter(HF_MAX_CONCURRENT);

// Serves the browser client (OBS Browser Source loads this page).
app.use(express.static(path.join(__dirname, 'public')));
// Exposes client config to the browser.
app.get('/config.json', (req, res) => {
  res.json(clientConfig);
});

// Mock ML inference: returns lightweight metadata (no images).
// Input flags decide which outputs are produced.
function makeMockInference({ run_seg, run_face, run_text }) {
  const now = Date.now();
  const t = now / 1000;
  const faces = [];
  const texts = [];
  let seg = null;

  if (run_face) {
    const size = 0.18 + 0.02 * Math.sin(t * 1.4);
    const cx = 0.5 + 0.08 * Math.sin(t * 1.1);
    const cy = 0.35 + 0.05 * Math.cos(t * 1.3);
    faces.push({
      x: cx - size / 2,
      y: cy - size / 2,
      w: size,
      h: size
    });
  }

  if (run_text) {
    const w = 0.32 + 0.04 * Math.sin(t * 0.9);
    const h = 0.08;
    texts.push({ x: 0.1, y: 0.72 + 0.03 * Math.sin(t * 1.2), w, h });
    texts.push({ x: 0.55, y: 0.15 + 0.02 * Math.cos(t * 0.7), w: 0.3, h: 0.07 });
  }

  if (run_seg) {
    seg = {
      kind: 'circle',
      cx: 0.5 + 0.05 * Math.sin(t * 0.8),
      cy: 0.52 + 0.03 * Math.cos(t * 1.1),
      r: 0.38
    };
  }

  return { faces, texts, seg };
}

// Builds the request payload for the HF Space endpoints.
// IMPORTANT: Input image data type is base64 JPEG/PNG (data URL string).
function buildHfPayload(msg) {
  const format = (msg.image_format || 'jpeg').toLowerCase();
  const mime = format === 'png' ? 'image/png' : 'image/jpeg';
  const dataUrl = `data:${mime};base64,${msg.image}`;
  return {
    image: dataUrl,
    image_format: format,
    width: msg.width,
    height: msg.height,
    ts: msg.ts
  };
}

// Calls a specific HF Space endpoint and returns parsed JSON.
async function callHf(endpoint, payload) {
  return hfLimiter(async () => {
    const url = `${HF_BASE_URL}${endpoint}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), HF_TIMEOUT_MS);

    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal
    });

    clearTimeout(timeout);
    if (!res.ok) {
      throw new Error(`HF request failed: ${endpoint} ${res.status}`);
    }
    return res.json();
  });
}

function normalizeFaces(result) {
  return result.faces || result.boxes || result.detections || [];
}

function normalizeTexts(result) {
  return result.texts || result.boxes || result.detections || [];
}

function normalizeSeg(result) {
  return result.seg || result.mask || result;
}

// WebSocket endpoint: this IS the "ML endpoint" the client talks to.
// The client sends compressed frames; we reply with mock metadata.
wss.on('connection', (ws) => {
  ws.send(JSON.stringify({ type: 'hello', message: 'ml-ready' }));

  ws.on('message', async (data) => {
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch (err) {
      ws.send(JSON.stringify({ type: 'error', message: 'invalid-json' }));
      return;
    }

    // Only process frame messages (image + flags).
    if (msg.type !== 'frame') {
      return;
    }

    const start = Date.now();
    const flags = msg.flags || {};
    const requestPayload = buildHfPayload(msg);

    let inference;
    if (USE_MOCK) {
      inference = makeMockInference(flags);
    } else {
      const tasks = [];
      const keys = [];

      if (flags.run_seg) {
        keys.push('seg');
        tasks.push(callHf(HF_ENDPOINTS.seg, requestPayload));
      }
      if (flags.run_face) {
        keys.push('face');
        tasks.push(callHf(HF_ENDPOINTS.face, requestPayload));
      }
      if (flags.run_text) {
        keys.push('text');
        tasks.push(callHf(HF_ENDPOINTS.text, requestPayload));
      }

      const settled = await Promise.allSettled(tasks);
      const results = {};
      settled.forEach((item, index) => {
        const key = keys[index];
        if (item.status === 'fulfilled') {
          results[key] = item.value;
        }
      });

      inference = {
        faces: flags.run_face ? normalizeFaces(results.face || {}) : [],
        texts: flags.run_text ? normalizeTexts(results.text || {}) : [],
        seg: flags.run_seg ? normalizeSeg(results.seg || null) : null
      };
    }

    // "Inference response" sent back to client.
    const payload = {
      type: 'inference',
      id: msg.id,
      ts: msg.ts,
      latencyMs: Date.now() - start,
      faces: inference.faces,
      texts: inference.texts,
      seg: inference.seg
    };

    const delay = 12 + Math.floor(Math.random() * 18);
    setTimeout(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
      }
    }, delay);
  });
});

server.listen(PORT, () => {
  console.log(`OBS mock server running on http://localhost:${PORT}`);
});
