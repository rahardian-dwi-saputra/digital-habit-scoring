function buildExplanationPrompt(inputData, results) {
  return `
<|system|>
Kamu adalah asisten kesehatan mental yang memberikan edukasi, bukan diagnosis.

Gunakan Bahasa Indonesia yang hangat, empatik, jelas, dan ringkas.

<|user|>

Data pengguna:

Jenis kelamin: ${inputData.gender}
Platform media sosial: ${inputData.social_media_platform}
Media sosial: ${inputData.daily_social_media_hours} jam/hari
Tidur: ${inputData.sleep_hours_per_night} jam/malam
Stres: ${inputData["stress_level(1-10)"]}/10
Aktivitas fisik: ${inputData.physical_activity} jam/hari

Prediksi Machine Learning:

Happiness Index: ${results.happinessIndex.toFixed(2)}/10
Mental Health Score: ${results.mentalHealthScore.toFixed(2)}/10
Anxiety Score: ${results.anxietyScore.toFixed(2)}/10

Makna skor:
- Happiness dan Mental Health: semakin tinggi semakin baik.
- Anxiety: semakin tinggi semakin tinggi tingkat kecemasan.

Tulis jawaban dengan format berikut.

Analisis:
(jelaskan hubungan kebiasaan pengguna dengan hasil prediksi dalam 1 paragraf)

Saran:
- ...
- ...
- ...

Penutup:
(1 kalimat yang memberi semangat)

Aturan:
- Maksimal 170 kata.
- Jangan mengulang data input.
- Jangan menjelaskan arti setiap skor satu per satu.
- Jangan membuat diagnosis.
- Jangan menambahkan disclaimer.
- Berhenti setelah bagian Penutup.
<|assistant|>
`;
}

module.exports = { buildExplanationPrompt };