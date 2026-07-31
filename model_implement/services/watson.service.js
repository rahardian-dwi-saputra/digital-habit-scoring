const axios = require("axios");
const ibmConfig = require("../config/ibm");

async function getAccessToken() {
  const response = await axios.post(
    "https://iam.cloud.ibm.com/identity/token",
    `grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=${ibmConfig.apiKey}`,
    {
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
    }
  );

  return response.data.access_token;
}

async function generate(prompt) {
  const token = await getAccessToken();

  const response = await axios.post(
    `https://${ibmConfig.region}.ml.cloud.ibm.com/ml/v1/text/chat?version=${ibmConfig.version}`,
    {
      model_id: ibmConfig.modelId,
      project_id: ibmConfig.projectId,
      messages: [
        {
          role: "system",
          content:
            "Kamu adalah AI Assistant yang memberikan analisis berdasarkan data yang diberikan. Jangan membuat diagnosis medis atau psikologis.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      max_tokens: 600,
      temperature: 0.4,
      top_p: 0.9,
      repetition_penalty: 1.05
    },
    {
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
    }
  );

  return response.data.choices[0].message.content;
}

module.exports = { generate };