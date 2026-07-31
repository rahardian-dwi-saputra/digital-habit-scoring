const loadJson = require('../utils/loadJson');
const { CONFIG_PATHS } = require('../constants/models');

const anxietyConfig = loadJson(CONFIG_PATHS.ANXIETY);

module.exports = anxietyConfig;