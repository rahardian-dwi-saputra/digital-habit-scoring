/**
 * Preprocessor Dinamis berdasarkan Config JSON
 */
function genericPreprocess(rawData, config) {
  const processedData = {};

  // Standard Scaling
  if (config.numeric_features) {
    for (const [colName, stats] of Object.entries(config.numeric_features)) {
      const rawVal = Number(rawData[colName]) ?? 0;
      const scaledVal = (rawVal - stats.mean) / stats.std;
      processedData[`num__${colName}`] = scaledVal;
    }
  }

  // One-Hot Encoding
  if (config.categorical_features) {
    for (const [colName, catInfo] of Object.entries(config.categorical_features)) {
      const userVal = String(rawData[colName] || '').trim();

      catInfo.encoded_categories.forEach((category) => {
        const featureKey = `cat__${colName}_${category}`;
        processedData[featureKey] = (userVal === category) ? 1.0 : 0.0;
      });
    }
  }

  // Mengurutkan vektor sesuai `feature_names_out`
  return config.feature_names_out.map(
    (featureName) => processedData[featureName] ?? 0.0
  );
}

module.exports = genericPreprocess;