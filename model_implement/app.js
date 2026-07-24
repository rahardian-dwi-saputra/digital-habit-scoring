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

// Model 1 (Happiness Index)
const configPath1 = path.join(EXPORTS_DIR, 'predict_happiness_config.json');
const modelPath1 = path.join(EXPORTS_DIR, 'predict_happiness_model.onnx');

// Model 2 (Mental Health Score)
const configPath2 = path.join(EXPORTS_DIR, 'predict_mental_health_config.json');
const modelPath2 = path.join(EXPORTS_DIR, 'predict_mental_health_model.onnx');

const config1 = require(configPath1);
const config2 = require(configPath2);

function scaleValue(value, mean, std) {
    return (value - mean) / std;
}

/**
 * Memetakan Sleep_Hours_Per_Night (jam) ke nilai Sleep_Quality (skala 1 - 10)
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
 * Helper Preprocessing Input
 */
function preprocessModel1(rawData) {
    const scaledNumerics = {
        'Daily_Screen_Time(hrs)': scaleValue(
            rawData.dailyScreenTime,
            config1.numeric_features['Daily_Screen_Time(hrs)'].mean,
            config1.numeric_features['Daily_Screen_Time(hrs)'].std
        ),
        'Sleep_Quality(1-10)': scaleValue(
            rawData.sleepQuality,
            config1.numeric_features['Sleep_Quality(1-10)'].mean,
            config1.numeric_features['Sleep_Quality(1-10)'].std
        ),
        'Stress_Level(1-10)': scaleValue(
            rawData.stressLevel,
            config1.numeric_features['Stress_Level(1-10)'].mean,
            config1.numeric_features['Stress_Level(1-10)'].std
        )
    };

    const genderCategories = config1.categorical_features.Gender.encoded_categories;
    const genderOhe = {};
    genderCategories.forEach(cat => {
        genderOhe[cat] = (rawData.gender === cat) ? 1.0 : 0.0;
    });

    const platformCategories = config1.categorical_features.Social_Media_Platform.encoded_categories;
    const platformOhe = {};
    platformCategories.forEach(cat => {
        platformOhe[cat] = (rawData.platform === cat) ? 1.0 : 0.0;
    });

    return config1.feature_names_out.map(featureName => {
        if (featureName.startsWith('num__')) {
            const key = featureName.replace('num__', '');
            return scaledNumerics[key] ?? 0.0;
        } 
        if (featureName.startsWith('cat__Gender_')) {
            const category = featureName.replace('cat__Gender_', '');
            return genderOhe[category] ?? 0.0;
        } 
        if (featureName.startsWith('cat__Social_Media_Platform_')) {
            const category = featureName.replace('cat__Social_Media_Platform_', '');
            return platformOhe[category] ?? 0.0;
        }
        return 0.0;
    });
}

function preprocessModel2(rawData) {
    const scaledNumerics = {
        'Avg_Daily_Usage_Hours': scaleValue(
            rawData.dailyScreenTime,
            config2.numeric_features['Avg_Daily_Usage_Hours'].mean,
            config2.numeric_features['Avg_Daily_Usage_Hours'].std
        ),
        'Sleep_Hours_Per_Night': scaleValue(
            rawData.sleepHours,
            config2.numeric_features['Sleep_Hours_Per_Night'].mean,
            config2.numeric_features['Sleep_Hours_Per_Night'].std
        )
    };

    const genderCategories = config2.categorical_features.Gender.encoded_categories;
    const genderOhe = {};
    genderCategories.forEach(cat => {
        genderOhe[cat] = (rawData.gender === cat) ? 1.0 : 0.0;
    });

    const platformCategories = config2.categorical_features.Most_Used_Platform.encoded_categories;
    const platformOhe = {};
    platformCategories.forEach(cat => {
        platformOhe[cat] = (rawData.platform === cat) ? 1.0 : 0.0;
    });

    return config2.feature_names_out.map(featureName => {
        if (featureName.startsWith('scaler__')) {
            const key = featureName.replace('scaler__', '');
            return scaledNumerics[key] ?? 0.0;
        } 
        if (featureName.startsWith('ohe__Gender_')) {
            const category = featureName.replace('ohe__Gender_', '');
            return genderOhe[category] ?? 0.0;
        } 
        if (featureName.startsWith('ohe__Most_Used_Platform_')) {
            const category = featureName.replace('ohe__Most_Used_Platform_', '');
            return platformOhe[category] ?? 0.0;
        }
        return 0.0;
    });
}

/**
 * Fungsi Inferensi ONNX
 */
async function runModelInference(modelPath, featureVector) {
    const session = await ort.InferenceSession.create(modelPath);
    const numFeatures = featureVector.length;

    const inputTensor = new ort.Tensor(
        'float32',
        Float32Array.from(featureVector),
        [1, numFeatures]
    );

    const inputName = session.inputNames[0];
    const feeds = { [inputName]: inputTensor };

    const results = await session.run(feeds);
    const outputName = session.outputNames[0];

    return results[outputName].data[0];
}

app.post('/api/predict', async (req, res) => {
    try {
        const sleepHours = parseFloat(req.body.sleepHours);
        const mappedSleepQuality = mapSleepHoursToQuality(sleepHours);

        const rawData = {
            gender: req.body.gender,
            dailyScreenTime: parseFloat(req.body.dailyScreenTime),
            sleepQuality: mappedSleepQuality,
            sleepHours: parseFloat(req.body.sleepHours),
            stressLevel: parseFloat(req.body.stressLevel),
            platform: req.body.platform
        };

        // Preprocessing
        const features1 = preprocessModel1(rawData);
        const features2 = preprocessModel2(rawData);

        // Jalankan Prediksi Kedua Model secara Paralel
        const [happinessResult, mentalHealthResult] = await Promise.all([
            runModelInference(modelPath1, features1),
            runModelInference(modelPath2, features2)
        ]);

        res.json({
            success: true,
            results: {
                happinessIndex: happinessResult,
                mentalHealthScore: mentalHealthResult
            }
        });
    } catch (error) {
        console.error('Error saat prediksi:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`Server berjalan di http://localhost:${PORT}`);
});