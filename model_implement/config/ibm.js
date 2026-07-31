const env = require('./env');
module.exports = {
  apiKey: env.IBM_API_KEY,
  region: env.IBM_REGION,
  projectId: env.IBM_PROJECT_ID,
  modelId: env.IBM_MODEL,
  version: env.IBM_VERSION,
};