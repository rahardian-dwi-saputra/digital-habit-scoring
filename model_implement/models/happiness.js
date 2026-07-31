const loadJson = require('../utils/loadJson');
const { CONFIG_PATHS } = require('../constants/models');

const happinessConfig = loadJson(CONFIG_PATHS.HAPPINESS);

module.exports = happinessConfig;