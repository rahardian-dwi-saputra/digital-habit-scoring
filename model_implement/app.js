const express = require('express');
const path = require('path');
const predictRoutes = require('./routes/predict.routes');
const errorHandler = require('./middleware/errorHandler');

const app = express();

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/api/predict', predictRoutes);

app.use(errorHandler);

module.exports = app;