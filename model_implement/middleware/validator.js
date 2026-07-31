// Middleware opsional untuk memvalidasi body request sebelum diproses
function validatePredictInput(req, res, next) {

  next();
}

module.exports = { validatePredictInput };