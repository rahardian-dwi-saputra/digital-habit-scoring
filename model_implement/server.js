const app = require('./app');
const env = require('./config/env');
const { initONNX } = require('./config/onnx');

async function startServer() {
  await initONNX();

  app.listen(env.PORT, () => {
    console.log(`Server berjalan di http://localhost:${env.PORT}`);
  });
}

startServer();