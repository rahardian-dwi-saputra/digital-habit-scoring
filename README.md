# Prediksi Kebahagiaan & Kesehatan Mental Berdasarkan Penggunaan Media Sosial dan Gaya Hidup
*(Happiness & Mental Health Prediction Based on Social Media Usage and Lifestyle)*

## 🇮🇩 Bahasa Indonesia

### 1. Problem Statement (Pernyataan Masalah)
Penggunaan media sosial yang tinggi sering kali berdampak signifikan terhadap kesejahteraan mental dan tingkat kebahagiaan seseorang. Namun, banyak individu yang tidak menyadari bagaimana kebiasaan harian—seperti durasi layar (*screen time*), pola tidur, tingkat stres, dan aktivitas fisik—secara akumulatif memicu kecemasan (*anxiety*) atau menurunkan tingkat kebahagiaan (*happiness index*). Tanpa adanya alat evaluasi mandiri yang dapat diakses dengan mudah, sulit bagi masyarakat awam untuk mendapatkan gambaran awal serta analisis personal mengenai kondisi kesehatan mental dan kebiasaan digital mereka.

### 2. Solution Description (Deskripsi Solusi)
Aplikasi ini menyediakan platform berbasis web yang memungkinkan pengguna melakukan asesmen mandiri secara cepat. Pengguna cukup memasukkan data kebiasaan harian, meliputi:
* Platform media sosial yang digunakan dan durasi pemakaian.
* Tingkat stres harian.
* Durasi waktu tidur.
* Durasi aktivitas fisik.

Sistem akan mengkalkulasi estimasi **Happiness Index**, **Mental Health Score**, dan **Anxiety Score**. Selanjutnya, hasil tersebut dianalisis secara mendalam menggunakan IBM Granite untuk memberikan wawasan kontekstual, faktor risiko, serta saran tindakan yang personal, empati, dan intuitif.

### 3. AI Approach and Architecture (Pendekatan AI dan Arsitektur)
Sistem menggunakan pendekatan hibrida yang menggabungkan Pemrosesan Data Tabular/Logika Prediksi dengan *Large Language Model* (LLM):

1. **Input & Data Ingestion (Frontend/Client):** Pengguna menginput parameter kuantitatif (durasi sosial media, jam tidur, aktivitas fisik, dll.) melalui antarmuka web.
2. **Kalkulasi & Preprocessing (Backend/Core Engine):** Data diproses oleh algoritma prediksi/skoring untuk menghasilkan matriks nilai angka (*Happiness Index*, *Mental Health Score*, dan *Anxiety Score*).
3. **IBM Granite AI Analysis (IBM Granite AI Integration):** 
   * Matriks nilai beserta variabel gaya hidup dikompilasi menjadi *prompt* terstruktur (*Prompt Engineering*).
   * *Prompt* dikirimkan ke **IBM Granite API** untuk melakukan analisis kualitatif.
   * IMB Granite memproses konteks dan menghasilkan respon dalam bentuk ringkasan kondisi, interpretasi hubungan antar-variabel (misal: pengaruh kurang tidur terhadap durasi media sosial dan stres), serta rekomendasi praktis.
4. **Presentation Layer:** Hasil berupa angka indikator dan teks analisis dari IBM Granite AI ditampilkan kembali secara interaktif kepada pengguna.

## Informasi dataset
- [Mental Health & Social Media Balance Dataset](https://www.kaggle.com/datasets/prince7489/mental-health-and-social-media-balance-dataset)
- [Impact of Social Media on Health](https://www.kaggle.com/datasets/sumeakash/impact-of-social-media-on-health)
- [Teen Social Media Usage & Mental Health](https://www.kaggle.com/datasets/sureshbeekhani/teen-social-media-usage-and-mental-health)

## 🇬🇧 English

### 1. Problem Statement
High social media usage often has a significant impact on an individual's mental well-being and overall happiness. However, many people are unaware of how their daily habits—such as screen time, sleep patterns, stress levels, and physical activity—cumulatively trigger anxiety or lower their happiness index. Without accessible self-assessment tools, it is difficult for everyday users to gain initial insights and personalized analysis regarding their mental health and digital habits.

### 2. Solution Description
This application provides an accessible web-based platform for quick user self-assessments. Users input their daily lifestyle habits, including:
* Social media platforms used and duration of use.
* Daily stress levels.
* Sleep duration.
* Physical activity duration.

The system calculates estimated metrics for **Happiness Index**, **Mental Health Score**, and **Anxiety Score**. Subsequently, these scores are deeply analyzed using IBM Granite to deliver empathetic, actionable, and personalized contextual insights and recommendations.

### 3. AI Approach and Architecture
The system adopts a hybrid approach combining Tabular Data Processing/Predictive Scoring Logic with a Large Language Model (LLM):

1. **Input & Data Ingestion (Frontend/Client):** Users submit quantitative parameter data (social media usage duration, sleep hours, physical activity, etc.) via a user-friendly web interface.
2. **Scoring & Preprocessing (Backend/Core Engine):** Data is processed by a scoring/predictive algorithm to generate numerical metric values (*Happiness Index*, *Mental Health Score*, and *Anxiety Score*).
3. **IBM Granite AI Analysis (IMB Granite AI Integration):** 
   * The calculated metrics and daily lifestyle variables are formatted into a structured prompt (*Prompt Engineering*).
   * The prompt is dispatched to the **IMB Granite API** for qualitative interpretation.
   * IMB Granite AI evaluates the context and returns tailored insights, identifying correlations between habits (e.g., how lack of sleep combined with high screen time elevates anxiety) alongside practical well-being recommendations.
4. **Presentation Layer:** Numerical indicators and IMB Granite AI's qualitative narrative analysis are formatted and displayed interactively on the user dashboard.

## 📂 Struktur Proyek

| Direktori | Deskripsi |
| :--- | :--- |
| `model_development` | Berisi seluruh kode Python untuk *data preprocessing*, pelatihan model, hingga ekspor ke format ONNX. |
| `model_implement` | Berisi aplikasi backend Node.js (Express) & antarmuka web untuk mengimplementasikan hasil prediksi. |

---

## ⚡ Prasyarat Sistem

| Teknologi | Fungsi |
| :--- | :--- |
| **Conda** | Pengelolaan *virtual environment* Python agar dependensi ML terisolasi dengan baik. |
| **Node.js** | Runtime untuk menjalankan server web dan memproses prediksi model ONNX. |

## 🚀 Panduan Instalasi & Penggunaan (Node.js App)

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi backend/frontend inferensi Machine Learning berbasis Node.js.

### 📋 Prasyarat

Sebelum memulai, pastikan kamu telah menginstall:
* [Node.js](https://nodejs.org/) (Versi 18 atau yang lebih baru)
* [Git](https://git-scm.com/)

### 📥 Langkah Instalasi
#### 1. Clone Repositori
Buka terminal dan jalankan perintah berikut untuk meng-clone proyek ke komputer lokal kamu:
```sh
git clone https://github.com/rahardian-dwi-saputra/digital-habit-scoring.git
cd digital-habit-scoring/model_implement
```
### 2. Install Dependency
Jalankan perintah ini untuk memasang paket-paket yang diperlukan (`express`, `onnxruntime-node`, `@google/generative-ai`, dll):
```sh
npm install
```
### 3. Konfigurasi Environment Variable
Buat file bernama `.env` di dalam direktori `model_implement/` melalui cmd
```sh
copy .env.example .env
```
Lalu tambahkan kunci API Gemini ke dalam file tersebut
```sh
PORT=3001
GEMINI_API_KEY=masukkan_api_key_gemini_kamu_di_sini
```
### 🏃 Memulai Aplikasi
Jalankan server menggunakan Node.js:
```sh
npm start
```
Buka browser dan akses `http://localhost:3001` untuk membuka antarmuka web.