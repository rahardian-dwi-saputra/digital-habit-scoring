const loadJson = require('../utils/loadJson');
const { CONFIG_PATHS } = require('../constants/models');

const mentalHealthConfig = loadJson(CONFIG_PATHS.MENTAL_HEALTH);

module.exports = mentalHealthConfig;