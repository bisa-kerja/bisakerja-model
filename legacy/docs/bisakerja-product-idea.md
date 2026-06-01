# Template Ide Produk — Bisakerja (Final Version)

## 1. Nama Ide

**Bisakerja: AI-Powered Career Decision Engine untuk Pencari Kerja Indonesia**

## 2. Latar Belakang Masalah

### Target Pengguna

- Fresh graduate, early career (0–3 tahun pengalaman), dan career switcher di bidang digital/tech
- Pengguna yang aktif mencari kerja namun kesulitan menentukan strategi yang tepat dan efisien

### Masalah Utama

- Lowongan kerja tersebar di berbagai platform sehingga proses pencarian menjadi tidak efisien dan memakan waktu
- Pengguna tidak memiliki visibilitas terhadap tingkat kecocokan (job fit) mereka dengan suatu lowongan
- Tidak adanya sistem yang mampu mengidentifikasi kesenjangan skill (skill gap) terhadap pekerjaan yang diinginkan
- Proses melamar kerja cenderung berbasis trial & error tanpa strategi yang jelas
- Tidak terdapat feedback loop untuk mengevaluasi hasil lamaran (diterima, ditolak, atau tidak ada respons)
- Minimnya guidance berbasis data untuk menentukan langkah karir selanjutnya

### Dampak

- Pengguna membuang waktu dan energi pada lowongan yang kurang relevan
- Peluang kerja yang sesuai sering terlewat
- Proses pencarian kerja menjadi tidak terarah dan menurunkan motivasi
- Pengguna tidak memiliki insight untuk meningkatkan peluang keberhasilan

### Kenapa Penting

- Dunia kerja semakin kompetitif dan berbasis skill
- Pencari kerja membutuhkan decision support system, bukan sekadar platform listing
- Pendekatan berbasis data dan AI dapat membantu meningkatkan efisiensi dan peluang diterima kerja

## 3. Solusi

Bisakerja dikembangkan sebagai **Career Decision Engine** yang membantu pengguna dalam:

- Menemukan lowongan kerja yang relevan
- Memahami tingkat kecocokan diri terhadap suatu pekerjaan
- Mengidentifikasi skill yang perlu ditingkatkan
- Menentukan strategi melamar yang optimal
- Mengevaluasi performa pencarian kerja secara berkelanjutan

### Integrasi Solusi

- Agregasi lowongan kerja
- AI-based job fit scoring
- Skill gap analysis
- Career strategy recommendation
- Application intelligence (feedback loop)

## 4. Fitur MVP

### Core Features

- Agregasi lowongan kerja dari 1–2 sumber utama
- Pencarian lowongan dengan filter:
  - keyword
  - lokasi
  - tipe kerja
- Detail lowongan yang terstruktur dan mudah dipahami
- Sistem autentikasi (register/login)

### Preferences User

- Skill
- Lokasi
- Tipe kerja
- Ekspektasi gaji

### AI & Intelligence Features (Fokus Utama)

#### 1. Job Fit Scoring (Explainable)

Menghitung skor kecocokan user terhadap lowongan (0–100)

**Berdasarkan:**

- Skill match
- Experience match
- Preferences match

**Output:**

- Persentase kecocokan
- Breakdown alasan (explainable AI)

#### 2. Skill Gap Analysis

Mengidentifikasi gap antara skill user dan requirement job

**Output:**

- Daftar skill yang belum dimiliki
- Prioritas skill (high/medium/low)
- Rekomendasi pembelajaran (basic)

#### 3. Career Strategy Recommendation (Key Feature)

Memberikan rekomendasi langkah strategis berbasis data

**Contoh:**

- Apakah job layak dilamar sekarang
- Skill apa yang harus diprioritaskan
- Estimasi kesiapan (readiness level)

**Output:**

- Actionable next steps
- Probabilitas keberhasilan (estimasi sederhana)

#### 4. AI Career Copilot (Basic)

Membantu user memahami konteks lowongan:

- Ringkasan job description
- Highlight requirement penting
- Insight tambahan

#### 5. Application Intelligence (Feedback Loop)

Melacak proses lamaran:

- Applied
- Interview
- Rejected

**Analisis:**

- Pola keberhasilan/penolakan
- Identifikasi bottleneck (misalnya skill tertentu)

**Output:**

- Insight untuk perbaikan strategi

### Supporting Features

- Bookmark lowongan
- Application tracker sederhana
- Riwayat aktivitas pengguna

## 5. Unique Value (UVP)

### Keunggulan Utama

Mengubah proses pencarian kerja dari sekadar browsing menjadi **data-driven decision making**

### Pembeda dari Kompetitor

- Job Fit Scoring yang explainable dan transparan
- Skill Gap Analysis yang actionable
- Career Strategy Recommendation yang memberikan arahan nyata
- Feedback loop berbasis data dari hasil lamaran

### Nilai Unik

- Mengurangi trial & error dalam melamar kerja
- Memberikan insight berbasis data untuk meningkatkan peluang diterima
- Membantu pengguna membangun strategi karir yang terarah

## 6. Target User

### Primary Segment

- Fresh graduate bidang digital/tech
- Early career (0–3 tahun pengalaman)

### Secondary Segment

- Career switcher ke bidang digital/tech

### Karakteristik Umum

- Digital-savvy
- Aktif mencari peluang kerja
- Membutuhkan guidance berbasis data

## 7. Model Bisnis (Optional)

### Model Utama: Freemium (B2C)

#### Free

- Akses lowongan
- Job fit scoring terbatas
- Bookmark & tracker dasar

#### Pro (Future Scope)

- Skill gap analysis lengkap
- Career strategy lebih advanced
- Insight & analytics lebih detail

## 8. Validasi Awal (Optional)

### Problem Validation

Interview pencari kerja:

- kesulitan menentukan job yang cocok
- pengalaman trial & error saat melamar

### Solution Validation

Uji:

- Akurasi job fit scoring
- Relevansi skill gap analysis
- Kegunaan rekomendasi strategi

### Indikator Keberhasilan

- User kembali menggunakan platform (retention)
- Penggunaan fitur AI secara berulang
- User merasa terbantu dalam pengambilan keputusan

## 9. Scope & Timeline (High-Level)

### Scope MVP

- Job aggregation (basic)
- Job fit scoring
- Skill gap analysis
- Career strategy recommendation (basic)
- Simple tracker

### Timeline (6–8 minggu)

#### Week 1–2

- Setup project, database, auth
- Basic job aggregation

#### Week 3–4

- Job search & preferences
- Job fit scoring

#### Week 5–6

- Skill gap analysis
- Career strategy recommendation

#### Week 7–8

- UI/UX refinement
- Testing & demo preparation

## 10. Teknologi yang Digunakan

### Frontend

- React / Next.js
- Tailwind CSS

### Backend

- Express.js

### Database

- PostgreSQL (Prisma ORM)

### AI / Processing

- NLP (OpenAI / model sederhana)
- Rule-based scoring system

### Infrastructure (Optional)

- Docker
- VPS deployment

## 11. Future Development (Optional)

- Integrasi learning platform (course recommendation)
- CV analyzer & auto-improvement
- Interview preparation assistant
- Employer dashboard
- Mobile application
