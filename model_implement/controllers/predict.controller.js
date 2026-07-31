const { predictMentalStatus } = require('../services/prediction.service');
const { explainWithWatson } = require('../services/ai.service');

async function handlePredict(req, res, next) {
  try {
    const { rawData, mlResults } = await predictMentalStatus(req.body);
    const watsonExplanation = await explainWithWatson(rawData, mlResults);

    return res.json({
      success: true,
      results: mlResults,
      explanation: watsonExplanation,
    });
  } catch (error) {
    next(error);
  }
}

module.exports = { handlePredict };