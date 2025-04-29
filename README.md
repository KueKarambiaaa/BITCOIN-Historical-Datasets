# Prediksi Harga Bitcoin menggunakan Machine Learning

## 📌 Ringkasan Proyek
Proyek ini bertujuan untuk memprediksi **harga penutupan Bitcoin** menggunakan data historis dari tahun 2018 hingga 2024. Beberapa model machine learning berbasis regresi digunakan untuk mengevaluasi performa prediksi harga.

## 💼 Latar Belakang
Bitcoin adalah aset dengan volatilitas tinggi dan menjadi fokus utama dalam bidang **keuangan, perdagangan, dan data sains**. Prediksi harga yang akurat sangat penting untuk trader, investor, dan analis.

Sumber dataset: [Kaggle - Bitcoin Historical Data (2018-2024)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

## 🎯 Pemahaman Bisnis

### Pernyataan Masalah (Problem Statement)
Bagaimana memprediksi harga penutupan (close price) Bitcoin berdasarkan data historis untuk membantu pengambilan keputusan investasi?

### Tujuan (Goals)
Membangun model machine learning untuk memprediksi harga penutupan Bitcoin dengan akurasi tinggi.

### Pernyataan Solusi (Solution Statement)
Beberapa pendekatan yang digunakan:
- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

Model terbaik akan dipilih berdasarkan evaluasi dengan metrik MAE, MSE, RMSE, dan R².

## 📊 Pemahaman Data
- Periode data: 2018 hingga awal 2024
- Fitur penting: `Open`, `High`, `Low`, `Volume`, dengan target `Close`
- Format tanggal telah dikonversi ke tipe `datetime`

## 🛠️ Persiapan Data
- Menghapus nilai kosong (null)
- Konversi tipe data jika diperlukan
- Fitur input: `Open`, `High`, `Low`, `Volume`
- Target: `Close`
- Pembagian data: 80% data latih, 20% data uji
- Standardisasi fitur dengan `StandardScaler`

## 🤖 Pemodelan
Model yang digunakan:
1. **Linear Regression**
   - Kelebihan: cepat, mudah diinterpretasi
   - Kekurangan: sensitif terhadap outlier

2. **Random Forest Regressor**
   - Kelebihan: akurasi tinggi, tidak terlalu overfit
   - Kekurangan: lebih lambat dibanding model linier

3. **XGBoost Regressor**
   - Kelebihan: akurasi sangat tinggi, efisien
   - Kekurangan: memerlukan tuning parameter yang kompleks

### Model Terbaik: XGBoost
Model ini dipilih karena menghasilkan **nilai R² tertinggi** dan **RMSE terendah**.

## 📈 Evaluasi
Metrik evaluasi yang digunakan:
- **MAE (Mean Absolute Error)**: rata-rata kesalahan absolut antara nilai aktual dan prediksi
- **MSE (Mean Squared Error)**: menghitung rata-rata kuadrat kesalahan; penalti lebih tinggi untuk kesalahan besar
- **RMSE (Root Mean Squared Error)**: akar dari MSE, nilai satuannya sama dengan target
- **R² (R-squared)**: proporsi variansi target yang dapat dijelaskan oleh model

## 📦 File dalam Proyek Ini
- `Bitcoin_Historical_Data_(ML).ipynb`: Notebook utama
- `bitcoin_model.py`: Script python dari notebook
- `README.md`: Laporan proyek dalam format Markdown

## ✅ Kesimpulan
Dengan pendekatan machine learning, khususnya XGBoost, proyek ini berhasil membangun model prediksi harga penutupan Bitcoin yang cukup akurat berdasarkan data historis. Model ini dapat menjadi dasar pengembangan sistem pendukung keputusan untuk investasi aset kripto.
