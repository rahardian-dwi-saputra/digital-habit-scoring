const { buildExplanationPrompt } = require('./prompt.service');
const watsonService = require("./watson.service");

async function explainWithWatson(inputData, results) {
  const prompt = buildExplanationPrompt(inputData, results);
  return await watsonService.generate(prompt);
}

module.exports = { explainWithWatson };