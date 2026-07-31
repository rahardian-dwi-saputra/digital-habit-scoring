const path = require('path');

const EXPORTS_DIR = path.join(__dirname, '..', '..', 'model_development', 'exports');

module.exports = {
  CONFIG_PATHS: {
    HAPPINESS: path.join(EXPORTS_DIR, 'predict_happiness_config.json'),
    MENTAL_HEALTH: path.join(EXPORTS_DIR, 'predict_mental_health_config.json'),
    ANXIETY: path.join(EXPORTS_DIR, 'predict_anxiety_config.json')
  },
  MODEL_PATHS: {
    HAPPINESS: path.join(EXPORTS_DIR, 'predict_happiness_model.onnx'),
    MENTAL_HEALTH: path.join(EXPORTS_DIR, 'predict_mental_health_model.onnx'),
    ANXIETY: path.join(EXPORTS_DIR, 'predict_anxiety_model.onnx')
  }
};