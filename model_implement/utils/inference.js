const ort = require('onnxruntime-node');

async function runSessionInference(session, featureVector) {
  const inputTensor = new ort.Tensor(
    'float32',
    Float32Array.from(featureVector),
    [1, featureVector.length]
  );

  const inputName = session.inputNames[0];
  const feeds = { [inputName]: inputTensor };

  const results = await session.run(feeds);
  const outputName = session.outputNames[0];

  return results[outputName].data[0];
}

module.exports = runSessionInference;