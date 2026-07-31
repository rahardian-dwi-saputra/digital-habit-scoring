const ort = require('onnxruntime-node');
const { MODEL_PATHS } = require('../constants/models');

const sessions = {};

async function initONNX() {
  try {
    sessions.happiness = await ort.InferenceSession.create(MODEL_PATHS.HAPPINESS);
    sessions.mentalHealth = await ort.InferenceSession.create(MODEL_PATHS.MENTAL_HEALTH);
    sessions.anxiety = await ort.InferenceSession.create(MODEL_PATHS.ANXIETY);
    console.log('Ketiga Model ONNX berhasil dimuat!');
  } catch (err) {
    console.error('Gagal inisialisasi ONNX Session:', err);
  }
}

function getSessions() {
  return sessions;
}

module.exports = {
  initONNX,
  getSessions
};