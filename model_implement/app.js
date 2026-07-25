require('dotenv').config();

const express = require('express');
const ort = require('onnxruntime-node');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const EXPORTS_DIR = path.join(__dirname, '..', 'model_development', 'exports');

const configPath1 = path.join(EXPORTS_DIR, 'predict_happiness_config.json');
const configPath2 = path.join(EXPORTS_DIR, 'predict_mental_health_config.json');
const configPath3 = path.join(EXPORTS_DIR, 'predict_anxiety_config.json');

const modelPath1 = path.join(EXPORTS_DIR, 'predict_happiness_model.onnx');
const modelPath2 = path.join(EXPORTS_DIR, 'predict_mental_health_model.onnx');
const modelPath3 = path.join(EXPORTS_DIR, 'predict_anxiety_model.onnx');

const config1 = JSON.parse(fs.readFileSync(configPath1, 'utf-8'));
const config2 = JSON.parse(fs.readFileSync(configPath2, 'utf-8'));
const config3 = JSON.parse(fs.readFileSync(configPath3, 'utf-8'));

let session1, session2, session3;

async function initONNX() {
    try {
        session1 = await ort.InferenceSession.create(modelPath1);
        session2 = await ort.InferenceSession.create(modelPath2);
        session3 = await ort.InferenceSession.create(modelPath3);
        console.log('Ketiga Model ONNX & Config berhasil dimuat!');
    } catch (err) {
        console.error('Gagal inisialisasi ONNX Session:', err);
    }
}
initONNX();

/**
 * Memetakan sleep_hours (jam) ke nilai sleep_quality (skala 1 - 10)
 */
function mapSleepHoursToQuality(sleepHours) {
    if (sleepHours >= 7.0 && sleepHours <= 9.0) {
        return 10.0;
    } else if (sleepHours < 7.0) {
        const score = 1.0 + (sleepHours / 7.0) * 9.0;
        return Math.max(1.0, Math.min(10.0, score));
    } else {
        const score = 10.0 - ((sleepHours - 9.0) / 5.0) * 9.0;
        return Math.max(1.0, Math.min(10.0, score));
    }
}

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

/**
 * Helper Inferensi ONNX
 */
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

// Endpoint Prediksi
app.post('/api/predict', async (req, res) => {
    try {
        if (!session1 || !session2 || !session3) {
            return res.status(503).json({ 
                success: false, 
                error: 'Model ONNX belum siap, silakan coba beberapa saat lagi.' 
            });
        }

        const sleepHours = parseFloat(req.body.sleepHours || req.body.sleep_hours_per_night || 0);
        const mappedSleepQuality = mapSleepHoursToQuality(sleepHours);

        const rawData = {
            daily_social_media_hours: parseFloat(req.body.dailyScreenTime || req.body.daily_social_media_hours || 0),
            'sleep_quality(1-10)': mappedSleepQuality,
            'stress_level(1-10)': parseFloat(req.body.stressLevel || req.body['stress_level(1-10)'] || 0),
            sleep_hours_per_night: sleepHours,
            physical_activity: parseFloat(req.body.physicalActivity || req.body.physical_activity || 0),
            gender: req.body.gender,
            social_media_platform: req.body.platform || req.body.social_media_platform
        };

        // Preprocessing Vektor Fitur
        const features1 = genericPreprocess(rawData, config1);
        const features2 = genericPreprocess(rawData, config2);
        const features3 = genericPreprocess(rawData, config3);

        // Menjalankan Prediksi Ketiga Model secara Paralel
        const [happinessResult, mentalHealthResult, anxietyResult] = await Promise.all([
            runSessionInference(session1, features1),
            runSessionInference(session2, features2),
            runSessionInference(session3, features3)
        ]);

        return res.json({
            success: true,
            results: {
                happinessIndex: happinessResult,
                mentalHealthScore: mentalHealthResult,
                anxietyScore: anxietyResult
            }
        });
    } catch (error) {
        console.error('Error saat prediksi:', error);
        return res.status(500).json({ success: false, error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`Server berjalan di http://localhost:${PORT}`);
});