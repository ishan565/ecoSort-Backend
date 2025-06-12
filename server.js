const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');

const app = express();
const port = process.env.PORT || 5001;

app.use(cors()); // Allow all origins

// Multer config
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 },
});

let model;

// Load model once at server start
(async () => {
  try {
    console.log('Loading model...');
    model = await tf.loadLayersModel('file://tfjs_model_dir/model.json');
    console.log('Model loaded!');
  } catch (err) {
    console.error('Model load error:', err);
  }
})();

// Categories (must match frontend model training)
const categories = [
  'Plastic Bottle',
  'Glass Jar',
  'Organic Waste',
  'Paper',
  'Aluminum Can',
  'Textile',
  'Electronic Waste',
  'General Trash',
];

const disposalTips = {
  'Plastic Bottle': 'Rinse and place in plastic recycling. Check local rules for caps.',
  'Glass Jar': 'Clean and place in glass recycling. Remove lids.',
  'Organic Waste': 'Compost or put in organic waste bin.',
  'Paper': 'Place dry paper in paper recycling.',
  'Aluminum Can': 'Rinse and flatten. Put in metal recycling.',
  'Textile': 'Donate if reusable. Otherwise, recycle or discard as trash.',
  'Electronic Waste': 'Take to an e-waste collection center. Do not throw in trash.',
  'General Trash': 'Dispose in landfill bin.',
};

const recyclingCenters = [
  {
    name: 'Green Earth Recycling',
    address: '123 Elm St, Springfield',
    latitude: 40.7128,
    longitude: -74.006,
    materials: ['Plastic', 'Glass', 'Paper'],
  },
  {
    name: 'Eco Waste Solutions',
    address: '456 Oak St, Springfield',
    latitude: 40.7135,
    longitude: -74.007,
    materials: ['Organic Waste', 'Electronic Waste'],
  },
  {
    name: 'Recycle Hub',
    address: '789 Pine St, Springfield',
    latitude: 40.7142,
    longitude: -74.008,
    materials: ['Aluminum', 'Textile', 'General Trash'],
  },
];

// Distance helper
function getDistanceFromLatLonInKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = deg2rad(lat2 - lat1);
  const dLon = deg2rad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) *
    Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}
function deg2rad(deg) {
  return deg * (Math.PI / 180);
}

// POST /classify
app.post('/classify', upload.single('image'), async (req, res) => {
  if (!model) return res.status(503).json({ error: 'Model not loaded' });
  if (!req.file) return res.status(400).json({ error: 'Image is required' });

  try {
    const imageBuffer = await sharp(req.file.buffer)
      .resize(300, 300)
      .removeAlpha()
      .raw()
      .toBuffer();

    const imgTensor = tf.tensor3d(imageBuffer, [300, 300, 3], 'int32');
    const inputTensor = imgTensor.toFloat().div(tf.scalar(255)).expandDims();

    const prediction = model.predict(inputTensor);
    const predictionArray = prediction.dataSync();
    const maxIndex = predictionArray.indexOf(Math.max(...predictionArray));

    const category = categories[maxIndex];
    const confidence = Number(predictionArray[maxIndex].toFixed(4));
    const disposal = disposalTips[category];

    inputTensor.dispose();
    imgTensor.dispose();
    prediction.dispose();

    res.json({ category, confidence, disposal });
  } catch (err) {
    console.error('Classification error:', err);
    res.status(500).json({ error: 'Error classifying image' });
  }
});

// GET /recycling-centers?lat=...&lon=...
app.get('/recycling-centers', (req, res) => {
  const { lat, lon } = req.query;
  if (!lat || !lon) return res.status(400).json({ error: 'Lat/lon required' });

  const userLat = parseFloat(lat);
  const userLon = parseFloat(lon);

  const centers = recyclingCenters
    .map((center) => {
      const distance = getDistanceFromLatLonInKm(
        userLat, userLon,
        center.latitude, center.longitude
      );
      return { ...center, distance: distance.toFixed(2) + ' km' };
    })
    .filter((c) => parseFloat(c.distance) <= 10)
    .sort((a, b) => parseFloat(a.distance) - parseFloat(b.distance));

  res.json(centers);
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Start server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});