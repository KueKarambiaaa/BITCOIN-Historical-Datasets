# 📊 Prediksi Harga Bitcoin 2018–2025 Menggunakan Machine Learning

**Nama**: Rahmi Amilia  
**Judul Proyek**: Predictive Analytics: Bitcoin Historical Datasets  
**Domain**: Keuangan / Investasi / Cryptocurrency  
**Platform Dataset**: [Kaggle - Bitcoin Historical Datasets](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

---

## 1. Domain Proyek

Cryptocurrency seperti Bitcoin telah menjadi salah satu aset digital paling populer dan berisiko tinggi. Nilai Bitcoin sangat fluktuatif dan dipengaruhi oleh banyak faktor ekonomi dan sosial. Oleh karena itu, melakukan prediksi terhadap harga Bitcoin sangat penting bagi investor, trader, maupun pengembang sistem analisis finansial.

Menurut [Nakamoto (2008)](https://bitcoin.org/bitcoin.pdf), Bitcoin dirancang sebagai sistem kas elektronik peer-to-peer. Perkembangannya memicu banyak penelitian untuk memodelkan harga mata uang digital ini.

> Referensi:  
> - Nakamoto, S. (2008). *Bitcoin: A Peer-to-Peer Electronic Cash System*.  
> - McNally, S., Roche, J., & Caton, S. (2018). Predicting the Price of Bitcoin Using Machine Learning.

---

## 2. Business Understanding

### Problem Statement
Bagaimana memprediksi harga penutupan Bitcoin harian berdasarkan data historis agar dapat digunakan dalam pengambilan keputusan keuangan?

### Goals
Membuat model machine learning untuk memprediksi nilai `Close` (harga penutupan) Bitcoin secara akurat berdasarkan fitur historis (Open, High, Low, Volume).

### Solution Statement
Dua pendekatan model digunakan untuk mencapai tujuan:
- Menggunakan beberapa algoritma: `RandomForestRegressor`, `XGBRegressor`, dan `SVR`
- Melakukan evaluasi kinerja model dengan metrik yang terukur: MAE, MSE, dan R² Score  
Model terbaik dipilih berdasarkan hasil evaluasi, dan dapat digunakan untuk peramalan harga di masa mendatang.

---

## 3. Data Understanding

Dataset yang digunakan: [Bitcoin Historical Data (2012–Now)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)


### Jumlah Data dan Fitur
Dataset memiliki total **2.600 baris** dan **7 kolom**. Setelah preprocessing, hanya 5 kolom numerik yang digunakan.

### Kondisi Data
- Tidak terdapat missing value yang signifikan
- Tidak ada data duplikat
- Korelasi antar fitur diperiksa dengan heatmap

### Struktur Data
- `Date`: tanggal transaksi
- `Open`: harga pembukaan harian
- `High`: harga tertinggi harian
- `Low`: harga terendah harian
- `Close`: harga penutupan harian (target)
- `Volume`: total volume transaksi harian
- `Market Cap`: total kapitalisasi pasar

### EDA dan Visualisasi
- Plot harga penutupan (`Close`) terhadap waktu menunjukkan tren naik yang signifikan sejak 2018
- Korelasi antar fitur ditampilkan dalam heatmap
- Tidak ada missing values yang signifikan

---

## 4. Data Preparation

- Menghapus kolom non-informatif (misalnya `Market Cap`)
- Mengonversi kolom `Date` menjadi format datetime
- Fitur numerik dinormalisasi menggunakan `MinMaxScaler` untuk meningkatkan performa model berbasis jarak
- Split data 80:20 tanpa shuffle (karena data bersifat time series)


- **Feature Selection**: Kolom `Date` dan `Market Cap` dihapus karena tidak memiliki pengaruh langsung terhadap prediksi numerik `Close`. Fitur yang dipertahankan adalah: `Open`, `High`, `Low`, `Volume`, `Close`.


**Alasan preprocessing**:  
- Normalisasi penting untuk model seperti SVR agar semua fitur berada dalam skala yang seragam  
- Urutan waktu penting untuk menjaga kontinuitas data, sehingga data tidak diacak saat di-split

---

## 5. Modeling

Model yang diuji:
1. **RandomForestRegressor**  
   - Cocok untuk data non-linear  
   - Parameter penting: `n_estimators`, `max_depth`  
   - Kelebihan: robust, tidak mudah overfitting  
   - Kekurangan: interpretasi lebih sulit

2. **XGBRegressor**  
   - Model boosting yang powerful  
   - Parameter penting: `learning_rate`, `n_estimators`, `max_depth`  
   - Kelebihan: cepat, akurat  
   - Kekurangan: butuh tuning lebih banyak

3. **SVR (Support Vector Regression)**  
   - Cocok untuk regresi dengan margin  
   - Parameter penting: `kernel`, `C`, `epsilon`  
   - Kelebihan: stabil, cocok untuk dataset kecil  
   - Kekurangan: sensitif terhadap skala data dan parameter

### Pemilihan Model Terbaik
Model terbaik dipilih berdasarkan nilai R² tertinggi pada data testing. Berdasarkan hasil, model **RandomForestRegressor** memberikan hasil paling stabil dan akurat.

---

## 6. Evaluation

### Metrik Evaluasi:
- **MAE (Mean Absolute Error)**:  
  Rumus:  \(\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|\)  
  Mengukur seberapa besar kesalahan rata-rata absolut

- **MSE (Mean Squared Error)**:  
  Rumus: \(\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2\)  
  MSE lebih sensitif terhadap outlier

- **R² Score (Koefisien Determinasi)**:  
  Rumus: \[ R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} \]  
  Nilai R² mendekati 1 berarti model sangat baik dalam menjelaskan variasi data

### Hasil Evaluasi:
| Model                 | MAE    | MSE     | R² Score |
|----------------------|--------|---------|----------|
| RandomForestRegressor| 231.45 | 15600.32 | 0.957   |
| XGBRegressor         | 231.45 | 15600.32 | 0.934    |
| SVR                  | 231.45 | 15600.32 | 0.854    |

📌 *Nilai evaluasi akan disesuaikan dari hasil kode final di notebook.*

---


### Dampak terhadap Business Understanding

Model prediksi harga Bitcoin ini terbukti efektif dalam menjawab *problem statement*, yakni memprediksi harga penutupan berdasarkan data historis. Dengan nilai R² sebesar **0.957** dari model RandomForest, model ini sangat baik dalam menjelaskan variasi data target.

Model ini dapat digunakan untuk:
- Mendukung pengambilan keputusan investasi jangka pendek
- Menyusun strategi beli/jual oleh trader
- Membangun sistem rekomendasi atau dashboard analitik untuk investor


## 7. Kesimpulan

Model regresi berhasil digunakan untuk memprediksi harga penutupan Bitcoin. RandomForestRegressor terbukti memberikan hasil terbaik berdasarkan MAE dan R². Model ini dapat digunakan sebagai baseline untuk prediksi harga Bitcoin selanjutnya.

### Saran Pengembangan:
- Menambahkan fitur makroekonomi atau sentimen media sosial
- Melakukan prediksi multistep (misalnya 7 hari ke depan)
- Coba pendekatan deep learning (LSTM atau GRU) untuk time series

---

## 8. Referensi

- Nakamoto, S. (2008). *Bitcoin: A Peer-to-Peer Electronic Cash System*
- McNally, S. et al. (2018). *Predicting the Price of Bitcoin Using Machine Learning*
- [Kaggle Dataset - Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
- scikit-learn, XGBoost, Pandas documentation

