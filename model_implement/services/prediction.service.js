const mapSleepHoursToQuality = require('../utils/sleepMapper');
const genericPreprocess = require('../utils/preprocess');
const runSessionInference = require('../utils/inference');
const clamp = require('../utils/clamp');
const { getSessions } = require('../config/onnx');

const happinessConfig = require('../models/happiness');
const mentalHealthConfig = require('../models/mentalHealth');
const anxietyConfig = require('../models/anxiety');

async function predictMentalStatus(bodyData) {
  const sessions = getSessions();
  if (!sessions.happiness || !sessions.mentalHealth || !sessions.anxiety) {
    throw new Error('MODEL_NOT_READY');
  }

  const sleepHours = parseFloat(bodyData.sleepHours || 0);
  const mappedSleepQuality = mapSleepHoursToQuality(sleepHours);

  const rawData = {
    daily_social_media_hours: parseFloat(bodyData.dailySocialMediaHours || 0),
    'sleep_quality(1-10)': mappedSleepQuality,
    'stress_level(1-10)': parseFloat(bodyData.stressLevel || 0),
    sleep_hours_per_night: sleepHours,
    physical_activity: parseFloat(bodyData.physicalActivity || 0),
    gender: bodyData.gender,
    social_media_platform: bodyData.platform
  };

  const features1 = genericPreprocess(rawData, happinessConfig);
  const features2 = genericPreprocess(rawData, mentalHealthConfig);

  const rawDataModel3 = { ...rawData };
  if (!['Instagram', 'TikTok'].includes(rawDataModel3.social_media_platform)) {
    rawDataModel3.social_media_platform = 'Other';
  }

  const features3 = genericPreprocess(rawDataModel3, anxietyConfig);

  const [happinessResult, mentalHealthResult, anxietyResult] = await Promise.all([
    runSessionInference(sessions.happiness, features1),
    runSessionInference(sessions.mentalHealth, features2),
    runSessionInference(sessions.anxiety, features3)
  ]);

  const mlResults = {
    happinessIndex: clamp(happinessResult),
    mentalHealthScore: clamp(mentalHealthResult),
    anxietyScore: clamp(anxietyResult)
  };

  return { rawData, mlResults };
}

module.exports = { predictMentalStatus };