const express = require('express');
const router = express.Router();
const { handlePredict } = require('../controllers/predict.controller');
const { validatePredictInput } = require('../middleware/validator');

router.post('/', validatePredictInput, handlePredict);

module.exports = router;