function errorHandler(err, req, res, next) {
  console.error('Error saat prediksi:', err);
  
  if (err.message === 'MODEL_NOT_READY') {
    return res.status(503).json({
      success: false,
      error: 'Model ONNX belum siap, silakan coba beberapa saat lagi.'
    });
  }

  return res.status(500).json({
    success: false,
    error: err.message || 'Terjadi kesalahan pada server.'
  });
}

module.exports = errorHandler;