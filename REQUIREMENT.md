# Requirement Proyek Machine Learning & Deployment

## 1. Pengembangan Model Deep Learning

### 1.1 Arsitektur Model

- Membangun model Deep Learning menggunakan:
  - TensorFlow Functional API **atau**
  - TensorFlow Model Subclassing
- Arsitektur model harus disesuaikan dengan dataset dan permasalahan bisnis yang telah ditentukan oleh tim Data Science (jika tersedia).

### 1.2 Implementasi Komponen Kustom

Model wajib mengimplementasikan minimal **satu** komponen kustom lanjutan berikut:

- Custom Layer
- Custom Loss Function
- Custom Callback

### 1.3 Custom Training Loop

- Mengimplementasikan proses training dan evaluation loop secara penuh menggunakan `tf.GradientTape`.
- Tidak menggunakan training loop standar (`model.fit()`) sebagai proses utama pelatihan.

### 1.4 Monitoring Training

- Mengintegrasikan TensorBoard untuk memantau proses pelatihan model.
- Menampilkan metrik pelatihan dan evaluasi secara menyeluruh.
- Menyertakan log TensorBoard yang dihasilkan ke dalam repository proyek.

### 1.5 Target Performa Model

Model harus memenuhi salah satu kriteria berikut sesuai jenis permasalahan:

#### Klasifikasi

- Akurasi (Accuracy) minimal **85%**

#### Regresi

- Mean Absolute Error (MAE) maksimal **0,02**

## 2. Penyimpanan dan Deployment Model

### 2.1 Ekspor Model

- Menyimpan model yang telah selesai dilatih dalam format TensorFlow siap produksi:
  - `.keras` **atau**
  - `SavedModel`

### 2.2 Inference

- Membuat kode sederhana untuk melakukan proses inference menggunakan model yang telah diekspor.

## 3. Pengembangan REST API

### 3.1 Framework API

Mengembangkan REST API mandiri menggunakan salah satu framework berikut:

- FastAPI
- Flask

### 3.2 Integrasi Model

REST API harus mampu:

- Memuat model hasil pelatihan.
- Menerima input dari pengguna.
- Menjalankan proses inferensi.
- Mengembalikan hasil prediksi dalam format JSON.

## 4. Integrasi Generative AI

### 4.1 Fitur Tambahan

- Menggunakan API Generative AI sebagai fitur tambahan atau fitur sekunder pada aplikasi.
- Implementasi dapat berupa:
  - Penjelasan hasil prediksi.
  - Ringkasan data.
  - Rekomendasi berbasis hasil model.
  - Chat assistant atau fitur AI pendukung lainnya.

## 5. Deliverables

Repository akhir minimal berisi:

- Source code pelatihan model.
- Implementasi custom component.
- Custom training loop menggunakan `tf.GradientTape`.
- Model hasil pelatihan (`.keras` atau `SavedModel`).
- Kode inference.
- REST API (FastAPI/Flask).
- Integrasi Generative AI.
- Log TensorBoard.
- Dokumentasi penggunaan dan deployment.
- File `requirements.txt`.
- README proyek.
