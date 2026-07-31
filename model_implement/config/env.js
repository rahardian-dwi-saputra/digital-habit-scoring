require('dotenv').config();

module.exports = {
    PORT: process.env.PORT || 3001,

    IBM_API_KEY: process.env.IBM_API_KEY,
    IBM_PROJECT_ID: process.env.IBM_PROJECT_ID,
    IBM_REGION: process.env.IBM_REGION || 'us-south',
    IBM_MODEL: process.env.IBM_MODEL || 'ibm/granite-3-8b-instruct',
    IBM_VERSION: process.env.IBM_VERSION || '2023-05-29',
};