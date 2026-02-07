# Trifecta

Mock OBS browser-source app with WebSocket streaming and ML inference metadata.

## Quick start

```bash
npm install
npm start
```

Open `http://localhost:3000` in a browser (or add it as an OBS **Browser Source**).

## Config

Edit `config.yaml` to control client sampling rate, image format, image size,
and backend endpoints/timeouts. The browser loads the client section from
`/config.json` at startup.
`node_preprocess.js` contains Node equivalents of your Python preprocessing
helpers (requires `sharp` and `tar-stream`).

## OBS usage

1. Add a **Browser Source** in OBS.
2. Set the URL to `http://localhost:3000`.
3. Click **Start** and pick Camera or Screen.
4. Toggle **Blur faces**, **Replace background**, or **Highlight text**.
5. Start **OBS Virtual Camera** to output the processed feed.

Note: some OBS builds restrict screen capture in Browser Sources. If screen share fails,
open `http://localhost:3000` in Chrome and add it to OBS via Window Capture.

## What's mocked

- The client captures frames, JPEG compresses them, and sends them over WebSocket.
- The server returns fake face boxes, text boxes, and a circular segmentation mask.
- All rendering (blur, background replacement, overlays) happens locally.
