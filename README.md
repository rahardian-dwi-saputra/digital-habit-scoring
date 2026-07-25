# Digital Habit Scoring
Digital Habit & Mental Health Scoring System

## Informasi dataset
- [Mental Health & Social Media Balance Dataset](https://www.kaggle.com/datasets/prince7489/mental-health-and-social-media-balance-dataset)
- [Impact of Social Media on Health](https://www.kaggle.com/datasets/sumeakash/impact-of-social-media-on-health)
- [Teen Social Media Usage & Mental Health](https://www.kaggle.com/datasets/sureshbeekhani/teen-social-media-usage-and-mental-health)

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
node app.js
```
Buka browser dan akses `http://localhost:3001` untuk membuka antarmuka web.