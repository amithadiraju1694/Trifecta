/* eslint-disable no-console */
const sharp = require('sharp');
const tar = require('tar-stream');

// Deterministic RNG (like numpy default_rng with a seed).
function mulberry32(seed) {
  let t = seed >>> 0;
  return function rand() {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

// Generate random RGB images as Uint8Array (Node equivalent of numpy integers).
function randomImages(numRows, imageSize, seed) {
  const rng = mulberry32(seed || 0);
  const images = [];
  const pixels = imageSize * imageSize * 3;

  for (let i = 0; i < numRows; i += 1) {
    const data = new Uint8Array(pixels);
    for (let j = 0; j < pixels; j += 1) {
      data[j] = Math.floor(rng() * 256);
    }
    images.push({ data, width: imageSize, height: imageSize });
  }
  return images;
}

// Encode a single RGB image into compressed bytes (Node equivalent of _encode_image).
async function encodeImage(img, imageFormat, jpegQuality = 90) {
  const ext = (imageFormat || 'jpeg').toLowerCase().replace('.', '');
  const input = sharp(Buffer.from(img.data), {
    raw: { width: img.width, height: img.height, channels: 3 }
  });

  if (ext === 'jpg' || ext === 'jpeg') {
    return input.jpeg({ quality: jpegQuality }).toBuffer();
  }
  return input.png().toBuffer();
}

function streamToBuffer(stream) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    stream.on('data', (chunk) => chunks.push(chunk));
    stream.on('end', () => resolve(Buffer.concat(chunks)));
    stream.on('error', reject);
  });
}

// Build a WebDataset-compatible tar payload (application/x-tar).
async function buildWebdatasetTar(encodedImages, imageFormat) {
  const ext = (imageFormat || 'jpeg').toLowerCase().replace('.', '');
  const pack = tar.pack();

  encodedImages.forEach((buf, index) => {
    const name = `${String(index).padStart(6, '0')}.${ext}`;
    pack.entry({ name }, buf);
  });
  pack.finalize();

  return streamToBuffer(pack);
}

// Node equivalent of your Python preprocessing wrapper.
async function buildPayload({
  numRows,
  imageSize,
  imageFormat,
  dtype = 'bytes',
  seed = 0
}) {
  const images = randomImages(numRows, imageSize, seed);
  const encoded = await Promise.all(images.map((img) => encodeImage(img, imageFormat)));

  if (dtype === 'bytes') {
    return {
      payload: encoded[0],
      headers: { 'Content-Type': `image/${imageFormat.toLowerCase().replace('.', '')}` }
    };
  }

  const payload = await buildWebdatasetTar(encoded, imageFormat);
  return { payload, headers: { 'Content-Type': 'application/x-tar' } };
}

module.exports = {
  randomImages,
  encodeImage,
  buildWebdatasetTar,
  buildPayload
};

if (require.main === module) {
  buildPayload({ numRows: 2, imageSize: 64, imageFormat: 'jpeg', dtype: 'bytes', seed: 42 })
    .then((result) => {
      console.log('Payload bytes:', result.payload.length);
      console.log('Headers:', result.headers);
    })
    .catch((err) => {
      console.error(err);
    });
}
