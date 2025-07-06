# Laporan Proyek Machine Learning - Sistem Rekomendasi Tanaman Berbasis Kondisi Tanah dan Iklim

#### Domain Proyek : Pertanian

Sektor pertanian menjadi pilar penting perekonomian global dan penopang ketahanan pangan. Namun, petani kerap mengalami kesulitan menentukan jenis tanaman yang cocok dengan karakteristik lahan dan pola iklim setempat, terutama bagi petani tradisional dengan akses terbatas pada teknologi. Kesalahan pemilihan komoditas berisiko menurunkan hasil panen, memicu kerugian finansial, hingga gagal tanam – situasi yang makin kompleks akibat iklim yang semakin ekstrem.

Solusi inovatif ditawarkan melalui analisis data menggunakan machine learning. Teknologi ini mampu memproses parameter seperti kadar unsur hara tanah (N, P, K), data iklim lokal (suhu, presipitasi, kelembaban), dan tingkat keasaman tanah untuk menghasilkan rekomendasi tanaman optimal. Pendekatan berbasis analisis data ini diharapkan dapat meningkatkan akurasi keputusan bertani, memaksimalkan produktivitas lahan, dan memitigasi risiko kerusakan hasil pertanian.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang yang telah dijelaskan, terdapat beberapa permasalahan utama yang perlu diselesaikan:

- Permasalahan 1: Bagaimana cara memberikan rekomendasi tanaman yang tepat berdasarkan kondisi tanah dan iklim untuk memaksimalkan produktivitas pertanian?
- Permasalahan 2: Bagaimana mengembangkan model machine learning yang dapat mengklasifikasikan jenis tanaman dengan akurasi tinggi berdasarkan parameter lingkungan?
- Permasalahan 3: Parameter mana yang paling berpengaruh dalam menentukan jenis tanaman yang sesuai dengan kondisi lingkungan tertentu?

### Goals

Tujuan dari proyek ini adalah:

- Tujuan 1: Mengembangkan sistem rekomendasi tanaman yang dapat memberikan saran jenis tanaman optimal berdasarkan kondisi tanah dan iklim dengan tingkat akurasi tinggi.
- Tujuan 2: Membuat model klasifikasi machine learning yang mampu memprediksi jenis tanaman dengan akurasi minimal 95% berdasarkan 7 parameter input (N, P, K, temperature, humidity, pH, rainfall).
- Tujuan 3: Mengidentifikasi dan menganalisis parameter yang paling berpengaruh dalam penentuan jenis tanaman untuk memberikan insight yang berguna bagi petani.

### Solution Statements

Untuk mencapai tujuan yang telah ditetapkan, akan diterapkan beberapa pendekatan solusi:

- Solusi 1: Menggunakan beberapa algoritma machine learning yang berbeda untuk klasifikasi, yaitu Random Forest, Decision Tree, Naive Bayes, dan Support Vector Machine (SVM), kemudian membandingkan performanya untuk memilih model terbaik.
- Solusi 2: Melakukan hyperparameter tuning pada model terbaik untuk meningkatkan performa dan akurasi prediksi.
- Solusi 3: Menerapkan teknik evaluasi komprehensif menggunakan multiple metrics (accuracy, precision, recall, F1-score) dan cross-validation untuk memastikan robustness model.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah Crop Recommendation Dataset yang diunduh dari Kaggle . Dataset ini berisi informasi mengenai kondisi tanah dan iklim yang optimal untuk berbagai jenis tanaman.

Sumber dataset :🔗 [Link ke dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

### Crop Recommendation Dataset

Dataset ini berisi 2200 entri dengan 8 fitur yang terkait dengan kondisi pertanian dan rekomendasi tanaman.

#### Struktur Dataset

| Kolom       | Tipe Data | Deskripsi                           |
| ----------- | --------- | ----------------------------------- |
| N           | int64     | Kadar Nitrogen dalam tanah          |
| P           | int64     | Kadar Fosfor dalam tanah            |
| K           | int64     | Kadar Kalium dalam tanah            |
| temperature | float64   | Suhu lingkungan (dalam °C)          |
| humidity    | float64   | Kelembaban relatif (dalam %)        |
| ph          | float64   | Tingkat pH tanah                    |
| rainfall    | float64   | Curah hujan (dalam mm)              |
| label       | object    | Jenis tanaman yang direkomendasikan |

#### Karakteristik Dataset

- **Jumlah Entri**: 2200
- **Range Indeks**: 0 sampai 2199
- **Tidak ada nilai null** pada semua kolom
- **Tipe Data**:
  - Numerik: int64 (N, P, K), float64 (temperature, humidity, ph, rainfall)
  - Kategorikal: object (label)

##### Variabel-variabel pada Crop Recommendation Dataset adalah sebagai berikut:

- N: Kandungan nitrogen dalam tanah (kg/ha)
- P: Kandungan fosfor dalam tanah (kg/ha)
- K: Kandungan kalium dalam tanah (kg/ha)
- temperature: Suhu rata-rata lingkungan (°C)
- humidity: Kelembaban relatif rata-rata (%)
- ph: Tingkat keasaman tanah (skala pH)
- rainfall: Curah hujan rata-rata (mm)
- label: Jenis tanaman yang direkomendasikan (target variable)

## Data Preparation

#### 1. Label Encoding

Mengubah variabel target (label tanaman) dari bentuk kategorikal menjadi numerik menggunakan LabelEncoder untuk memungkinkan proses training model

```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
```

Alasan: Algoritma machine learning memerlukan data numerik untuk proses komputasi, sehingga kategori tanaman perlu dikonversi ke bentuk numerik.

#### 2. Feature Scaling/Normalisasi

Menerapkan StandardScaler untuk menormalisasi fitur-fitur numerik, terutama untuk algoritma yang sensitif terhadap skala data seperti SVM dan Naive Bayes.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Alasan: Fitur-fitur memiliki rentang nilai yang berbeda (contoh: pH berkisar 3-9, sementara rainfall bisa mencapai 300). Normalisasi diperlukan agar tidak ada fitur yang mendominasi karena nilai yang lebih besar.

#### 3. Train-Test Split

Membagi dataset menjadi data training (80%) dan testing (20%) dengan stratified sampling untuk mempertahankan proporsi setiap kelas.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,  test_size=0.2,random_state=42, stratify=y_encoded)
```

Alasan: Stratified sampling memastikan setiap kelas tanaman terwakili secara proporsional di data training dan testing, sehingga evaluasi model lebih reliable.

## Modeling

Empat algoritma klasifikasi digunakan dalam proyek ini:

### 1. Random Forest

Cara Kerja: Random Forest adalah algoritma ensemble yang menggabungkan banyak pohon keputusan (decision tree) untuk membuat prediksi. Setiap pohon dalam forest dilatih pada subset data yang berbeda menggunakan teknik bootstrap sampling, dan setiap split dalam pohon hanya mempertimbangkan subset fitur secara acak. Prediksi akhir ditentukan melalui voting mayoritas dari semua pohon.

#### Parameter yang Digunakan:

- `n_estimators=100`: Menggunakan 100 pohon dalam ensemble
- `random_state=42`: Untuk reproducibility hasil
- `n_jobs=-1`: Menggunakan semua core CPU untuk training paralel

#### Keunggulan:

- Robust terhadap overfitting
- Dapat menangani data dengan fitur beragam
- Memberikan feature importance untuk interpretasi
- Tidak sensitif terhadap outliers

hasil = Akurasi: 99.55%

### 2. Decision Tree

Cara Kerja:Decision Tree membangun model dengan membuat pohon keputusan yang mempartisi data berdasarkan aturan if-else. Algoritma ini memilih fitur dan threshold yang memberikan information gain atau penurunan impurity terbesar pada setiap split. Proses ini berlanjut secara rekursif hingga mencapai kondisi stopping criteria.

#### Parameter yang Digunakan

- `random_state=42`: Untuk memastikan hasil yang konsisten (reproducibility)
- `max_depth=15`: Membatasi kedalaman pohon untuk mencegah overfitting
- `min_samples_split=10`: Minimum jumlah sampel yang diperlukan untuk melakukan pemisahan (split)

#### Keunggulan

- Sangat mudah diinterpretasi dan divisualisasikan
- Tidak memerlukan feature scaling
- Dapat menangani data numerik dan kategorikal
- Cepat dalam proses training dan prediksi

hasil : Akurasi: 98.18%

### 3. Naive Bayes (Gaussian)

Cara Kerja:
Naive Bayes menerapkan teorema Bayes dengan asumsi "naive" bahwa semua fitur saling independen. Untuk data kontinu, Gaussian Naive Bayes mengasumsikan bahwa likelihood dari fitur mengikuti distribusi normal. Algoritma menghitung probabilitas posterior untuk setiap kelas dan memilih kelas dengan probabilitas tertinggi.

#### Parameter yang Digunakan

- Menggunakan parameter default (tidak ada hyperparameter khusus yang disetel)
- Data telah dinormalisasi untuk performa optimal

#### Keunggulan

- Sangat cepat dalam training dan prediksi
- Efektif untuk dataset kecil hingga menengah
- Robust terhadap noise
- Tidak memerlukan tuning parameter yang kompleks

hasil : Akurasi: 99.55%

### 4. Support Vector Machine (SVM)

Cara Kerja: SVM bekerja dengan mencari hyperplane optimal yang memisahkan kelas-kelas data dengan margin maksimal. Untuk data yang tidak dapat dipisahkan secara linear, SVM menggunakan kernel trick (dalam hal ini RBF/Radial Basis Function) untuk memetakan data ke dimensi yang lebih tinggi dimana pemisahan linear menjadi mungkin.

#### Parameter yang Digunakan

- `kernel='rbf'`: Menggunakan Radial Basis Function (RBF) kernel
- `C=1.0`: Parameter regularisasi untuk mengontrol trade-off antara margin dan error
- `gamma='scale'`: Parameter kernel yang mengontrol pengaruh dari satu training example
- `random_state=42`: Untuk memastikan hasil yang konsisten (reproducibility)

#### Keunggulan

- Sangat efektif untuk data berdimensi tinggi
- Memory efficient karena hanya menggunakan subset dari data training (support vectors)
- Versatile dengan berbagai pilihan kernel
- Efektif untuk kasus dengan clear margin of separation

hasil : Akurasi: 98.41%

### Proses Training dan Hasil Awal

Setiap model dilatih menggunakan konfigurasi data yang sesuai:

- Random Forest dan Decision Tree: menggunakan data asli (tidak perlu normalisasi)
- Naive Bayes dan SVM: menggunakan data yang telah dinormalisasi

#### hasil akurasi awal

| Model         | Akurasi |
| ------------- | ------- |
| Random Forest | 99.55%  |
| Naive Bayes   | 99.55%  |
| SVM           | 98.41%  |
| Decision Tree | 98.18%  |

## Hyperparameter Tuning

Berdasarkan hasil evaluasi awal, Random Forest dan Naive Bayes menunjukkan performa terbaik dengan akurasi 99.55%. Random Forest dipilih untuk proses hyperparameter tuning karena memiliki lebih banyak parameter yang dapat dioptimasi dan memberikan feature importance untuk interpretasi.

#### Proses Tuning Random Forest

Random Forest dipilih untuk dituning karena memiliki performa terbaik. Proses tuning dilakukan menggunakan `GridSearchCV` dengan kombinasi parameter berikut:

- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 15, 20, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

#### Hasil Tuning

- **Parameter terbaik**:  
  `{'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}`
- **Akurasi setelah tuning**: 99.55%
- **Catatan**: Tidak ada peningkatan akurasi yang signifikan karena model awal sudah berada pada kondisi optimal

## Model Terbaik

**Random Forest** dipilih sebagai model terbaik karena:

1. Akurasi tertinggi: **99.55%**
2. Konsisten dalam cross-validation: **99.27% ± 0.78%**
3. Robust dan tidak overfitting
4. Menyediakan feature importance untuk interpretasi

## Evaluasi Model

Digunakan metrik evaluasi untuk kasus klasifikasi multi-class:

- **Accuracy**  
  Mengukur proporsi prediksi yang benar dari total prediksi.  
  Formula: `(TP + TN) / (TP + TN + FP + FN)`

- **Precision**  
  Mengukur proporsi prediksi positif yang benar.  
  Formula: `TP / (TP + FP)`

- **Recall**  
  Mengukur proporsi data positif yang berhasil diprediksi benar.  
  Formula: `TP / (TP + FN)`

- **F1-Score**  
  Rata-rata harmonik dari precision dan recall.  
  Formula: `2 × (Precision × Recall) / (Precision + Recall)`

### Hasil Evaluasi Model

#### Perbandingan Akurasi Semua Model

| Model                 | Akurasi | Ranking |
| --------------------- | ------- | ------- |
| Random Forest         | 99.55%  | 1       |
| Naive Bayes           | 99.55%  | 1       |
| Random Forest (Tuned) | 99.55%  | 1       |
| SVM                   | 98.41%  | 4       |
| Decision Tree         | 98.18%  | 5       |

---

### Detailed Performance - Random Forest (Model Terbaik)

##### Classification Report

- **Macro Average Precision**: 100%
- **Macro Average Recall**: 100%
- **Macro Average F1-Score**: 100%
- **Overall Accuracy**: 99.55%

##### Cross-Validation Results

- **CV Scores**: [99.77%, 99.09%, 99.55%, 99.32%, 98.64%]
- **Mean CV Score**: 99.27% ± 0.78%
- **Standard Deviation**: 0.39%

---

### Feature Importance Analysis

Model Random Forest memberikan insight tentang fitur yang paling berpengaruh:

1. **Rainfall** (22.31%) – Curah hujan menjadi faktor paling penting
2. **Humidity** (21.69%) – Kelembaban udara faktor kedua terpenting
3. **K (Kalium)** (18.34%) – Kandungan kalium dalam tanah
4. **P (Fosfor)** (14.56%) – Kandungan fosfor dalam tanah
5. **N (Nitrogen)** (10.20%) – Kandungan nitrogen dalam tanah
6. **Temperature** (7.55%) – Suhu lingkungan
7. **pH** (5.35%) – Tingkat keasaman tanah

---

### Analisis Confusion Matrix

Model Random Forest menunjukkan performa yang sangat baik dengan hanya **2 kesalahan prediksi** dari total **440 data uji**:

- **1 kesalahan** pada klasifikasi **blackgram** (1 sample diprediksi sebagai kelas lain)
- **1 kesalahan** pada klasifikasi **rice** (1 sample diprediksi sebagai kelas lain)

## Hubungan dengan Business Understanding

#### Menjawab Problem Statements

**Problem Statement 1:**  
_"Bagaimana cara memberikan rekomendasi tanaman yang tepat berdasarkan kondisi tanah dan iklim?"_  
**Jawaban:** Model Random Forest berhasil memberikan rekomendasi tanaman dengan akurasi **99.55%**, yang berarti dari 1000 rekomendasi, hanya sekitar 4–5 yang mungkin tidak tepat. Ini menunjukkan bahwa sistem mampu memberikan rekomendasi yang sangat akurat berdasarkan **7 parameter lingkungan**.

---

**Problem Statement 2:**  
_"Bagaimana mengembangkan model machine learning yang dapat mengklasifikasikan jenis tanaman dengan akurasi tinggi?"_  
**Jawaban:** Berhasil dikembangkan model Random Forest yang mencapai **akurasi 99.55%**, melebihi target minimal **95%** yang ditetapkan. Model ini mampu mengklasifikasikan **22 jenis tanaman** berbeda dengan tingkat kesalahan yang sangat rendah.

---

**Problem Statement 3:**  
_"Parameter mana yang paling berpengaruh dalam menentukan jenis tanaman?"_  
**Jawaban:** Analisis feature importance menunjukkan bahwa **rainfall (22.31%)** dan **humidity (21.69%)** merupakan faktor paling berpengaruh, diikuti oleh unsur hara **K (18.34%)** dan **P (14.56%)**. Insight ini sangat berguna bagi petani untuk fokus pada **parameter-parameter kunci**.

---

### Pencapaian Goals

| Goal                                                                   | Status                                                                                                                   |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Mengembangkan sistem rekomendasi tanaman dengan tingkat akurasi tinggi | TERCAPAI - Sistem berhasil dikembangkan dengan akurasi **99.55%**, sangat tinggi untuk aplikasi praktis.                 |
| Membuat model dengan akurasi minimal 95%                               | TERCAPAI - Akurasi **99.55%** jauh melebihi target.                                                                      |
| Mengidentifikasi parameter paling berpengaruh                          | TERCAPAI - Berhasil diidentifikasi bahwa faktor **iklim (rainfall, humidity)** lebih berpengaruh dibanding faktor tanah. |

---

### Dampak Solution Statements

**Solution 1:** _"Menggunakan beberapa algoritma dan membandingkan performa"_  
**Dampak:** **EFEKTIF** – Perbandingan 4 algoritma memungkinkan pemilihan model terbaik. Random Forest dan Naive Bayes menunjukkan performa superior, namun Random Forest dipilih karena **interpretabilitas yang lebih baik**.

---

**Solution 2:** _"Melakukan hyperparameter tuning"_  
**Dampak:** **TERBATAS** – Meskipun tuning tidak meningkatkan akurasi, proses ini memastikan **konfigurasi optimal** dan memberikan keyakinan bahwa model sudah mencapai **potensi maksimal**.

---

**Solution 3:** _"Evaluasi komprehensif dengan multiple metrics"_  
**Dampak:** **SANGAT EFEKTIF** – Cross-validation menunjukkan model **robust** (std dev hanya **0.78%**), dan penggunaan berbagai metrik evaluasi memberikan **gambaran menyeluruh** terhadap performa model.

## Interpretasi Hasil dan Implikasi Bisnis

### Keunggulan Model

- **Akurasi Tinggi (99.55%)**: Model dapat diandalkan untuk memberikan rekomendasi tanaman yang sangat akurat.
- **Robustness**: Nilai standard deviation CV yang rendah (0.78%) menunjukkan bahwa model stabil dan konsisten di berbagai subset data.
- **Interpretabilitas**: Analisis feature importance memberikan insight bisnis yang bernilai bagi pengambilan keputusan.
- **Generalisasi**: Performa konsisten pada data test menandakan bahwa model memiliki kemampuan generalisasi yang baik terhadap data baru.

---

### Nilai Bisnis

- **Peningkatan Produktivitas**: Petani dapat memilih jenis tanaman yang optimal dengan tingkat kepercayaan tinggi berdasarkan data lingkungan.
- **Pengurangan Risiko**: Kemungkinan gagal tanam dapat dikurangi secara signifikan melalui prediksi yang tepat.
- **Optimisasi Sumber Daya**: Dapat diarahkan fokus monitoring pada parameter kunci seperti curah hujan dan kelembaban untuk efisiensi operasional.
- **Skalabilitas**: Model berpotensi untuk diterapkan di berbagai wilayah dengan karakteristik tanah dan iklim yang berbeda.

---

### Limitasi dan Saran Pengembangan

- **Data Diversity**: Dataset masih terbatas pada 22 jenis tanaman. Perlu ekspansi agar model dapat memberikan rekomendasi untuk lebih banyak jenis tanaman.
- **Temporal Factors**: Model belum mempertimbangkan faktor musiman yang bisa memengaruhi hasil tanam.
- **Regional Adaptation**: Validasi lebih lanjut diperlukan untuk menjamin keakuratan model di berbagai kondisi geografis.

---

Model ini telah berhasil membuktikan bahwa machine learning dapat memberikan solusi yang praktis dan akurat dalam konteks rekomendasi tanaman untuk sektor pertanian. Potensinya sangat besar untuk meningkatkan produktivitas dan mengurangi risiko dalam praktik pertanian di berbagai wilayah.

## Pip instal /requiermet

numpy
pandas
matplotlib
seaborn
scikit-learn
kaggle
kagglehub
