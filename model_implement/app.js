require('dotenv').config();

const { GoogleGenerativeAI } = require('@google/generative-ai');
// Inisialisasi Gemini API
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const aiModel = genAI.getGenerativeModel({ model: 'gemini-3.6-flash' });

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
 * Helper untuk membuat penjelasan AI dari hasil skor ML
 */
async function generateExplanation(inputData, results) {
  const prompt = `
    Kamu adalah asisten/pakar kesehatan mental dan psikologi.
    Seorang pengguna memasukkan data profil berikut:
    - Jenis Kelamin: ${inputData.gender}
    - Platform Utama: ${inputData.social_media_platform}
    - Durasi Medsos: ${inputData.daily_social_media_hours} jam/hari
    - Durasi Tidur: ${inputData.sleep_hours_per_night} jam/malam
    - Tingkat Stress Saat Ini: ${inputData['stress_level(1-10)']}/10
    - Aktivitas Fisik: ${inputData.physical_activity} jam/hari

    Berdasarkan model Machine Learning kami, didapatkan skor indikator psikologis dengan rentang skala 1 hingga 10 (1 = paling rendah/buruk, 10 = paling tinggi/baik):

    1. Happiness Index: ${results.happinessIndex.toFixed(2)} / 10
    2. Mental Health Score: ${results.mentalHealthScore.toFixed(2)} / 10
    3. Anxiety Score: ${results.anxietyScore.toFixed(2)} / 10

    Catatan konteks skala untuk analisis:
    - Happiness Index: Nilai 1 (sangat tidak bahagia) hingga 10 (sangat bahagia).
    - Mental Health Score: Nilai 1 (kesehatan mental sangat buruk) hingga 10 (kesehatan mental sangat baik).
    - Anxiety Score: Nilai 1 (tingkat kecemasan sangat rendah/tenang) hingga 10 (tingkat kecemasan sangat tinggi/kritis).

    Tugasmu:
    Berikan penjelasan singkat (maksimal 3 paragraf) dalam Bahasa Indonesia yang ramah, empati, dan suportif:
    1. Analisis hubungan antara kebiasaan pengguna (medsos, tidur, aktivitas fisik, dan stres) dengan skor ML yang didapatkan.
    2. Berikan 2-3 saran praktis dan positif yang realistis untuk membantu meningkatkan kondisi mereka.

    Aturan Penulisan:
    - Hindari bahasa medis yang terlalu kaku, klinis, atau menakutkan.
    - Gunakan nada bicara yang merangkul dan tidak menghakimi.
  `;

  try {
    const response = await aiModel.generateContent(prompt);
    return response.response.text();
  } catch (err) {
    console.error('Error Generative AI:', err);
    return 'Maaf, gagal memuat penjelasan otomatis AI saat ini.';
  }
}

/**
 * Memetakan durasi tidur (jam) ke skala kualitas tidur (1 - 10).
 * 
 * Logic Pemetaan:
 * - Titik ideal: 7 - 9 jam -> Nilai 10
 * - Kurang tidur: Menurun 1.5 poin per jam di bawah 7 jam
 * - Kelebihan tidur: Menurun 1.0 poin per jam di atas 9 jam
 * 
 * @param {number} sleepHours - Jumlah jam tidur (bisa desimal, misal: 6.5)
 * @returns {number} Nilai kualitas tidur (Integer 1 - 10)
 */
function mapSleepHoursToQuality(sleepHours) {
    const hours = parseFloat(sleepHours);
    if (isNaN(hours) || hours <= 0) {
        return 1;
    }

    let quality;

    if (hours >= 7 && hours <= 9) {  // Rentang ideal (7 - 9 jam)
        quality = 10;
    } else if (hours < 7) { // Kurang dari 7 jam
        quality = 10 - (7 - hours) * 1.5;
    } else { // Lebih dari 9 jam
        quality = 10 - (hours - 9) * 1.0;
    }

    let roundedQuality = Math.round(quality);

    if (roundedQuality > 10) roundedQuality = 10;
    if (roundedQuality < 1) roundedQuality = 1;

    return roundedQuality;
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

        const sleepHours = parseFloat(req.body.sleepHours || 0);
        const mappedSleepQuality = mapSleepHoursToQuality(sleepHours);

        const rawData = {
            daily_social_media_hours: parseFloat(req.body.dailySocialMediaHours || 0),
            'sleep_quality(1-10)': mappedSleepQuality,
            'stress_level(1-10)': parseFloat(req.body.stressLevel || 0),
            sleep_hours_per_night: sleepHours,
            physical_activity: parseFloat(req.body.physicalActivity || 0),
            gender: req.body.gender,
            social_media_platform: req.body.platform
        };

        // Preprocessing Vektor Fitur
        const features1 = genericPreprocess(rawData, config1);
        const features2 = genericPreprocess(rawData, config2);

        const rawDataModel3 = { ...rawData };
        if (!['Instagram', 'TikTok'].includes(rawDataModel3.social_media_platform)) {
            rawDataModel3.social_media_platform = 'Other';
        }
        
        const features3 = genericPreprocess(rawDataModel3, config3);
        
        // Menjalankan Prediksi Ketiga Model secara Paralel
        const [happinessResult, mentalHealthResult, anxietyResult] = await Promise.all([
            runSessionInference(session1, features1),
            runSessionInference(session2, features2),
            runSessionInference(session3, features3)
        ]);

        const clamp = (num) => Math.min(Math.max(num, 1), 10);

        const mlResults = {
            happinessIndex: clamp(happinessResult),
            mentalHealthScore: clamp(mentalHealthResult),
            anxietyScore: clamp(anxietyResult)
        };

        const aiExplanation = await generateExplanation(rawData, mlResults);

        return res.json({
            success: true,
            results: mlResults,
            explanation: aiExplanation
        });
    } catch (error) {
        console.error('Error saat prediksi:', error);
        return res.status(500).json({ success: false, error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`Server berjalan di http://localhost:${PORT}`);
});