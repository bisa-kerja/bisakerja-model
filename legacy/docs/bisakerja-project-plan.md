# Dokumen Project Plan

**Coding Camp 2026 powered by DBS Foundation**

## Informasi Umum

- **ID Tim Capstone Project**: CC26-PSU263
- **Tema Capstone**: Future-Ready Work & Economy
- **Nama/Judul Proyek**: Bisakerja

### List Anggota

- CDCC183D6X0746 - Tasya Anggraeni Firdaus - Data Scientist - [Aktif]
- CDCC183D6Y1799 - Dzikri Albantani - Data Scientist - [Aktif]
- CFCC183D6Y1124 - Salman Abdurrahman - Full-Stack Web Developer - [Aktif]
- CFCC183D6Y1549 - Agel Saputra - Full-Stack Web Developer - [Aktif]
- CACC183D6X0902 - Linda David - AI Engineer - [Aktif]

## Ringkasan Eksekutif

Bisakerja hadir sebagai respons atas dinamika pasar tenaga kerja Indonesia yang kompetitif namun terhambat oleh inefisiensi pencocokan kualifikasi. Berdasarkan data Badan Pusat Statistik (BPS) per Agustus 2025, Tingkat Pengangguran Terbuka (TPT) nasional tercatat sebesar 4,85% atau mencakup 7,46 juta jiwa. Meskipun secara umum menurun, tingkat pengangguran untuk lulusan universitas masih berada pada angka yang mengkhawatirkan yaitu 5,83%. Kondisi ini diperburuk oleh fenomena horizontal mismatch atau ketidaksesuaian bidang studi dengan pekerjaan yang mencapai 73,97% pada kelompok lulusan vokasi muda.

Masalah utama dalam ekosistem ini adalah rendahnya visibilitas terhadap tingkat kecocokan kerja serta fragmentasi informasi lowongan di berbagai platform yang memaksa pencari kerja menggunakan metode trial and error dalam melamar pekerjaan. Dampaknya adalah rendahnya tingkat respons lamaran yang hanya berkisar 2%, serta meningkatnya beban psikososial akibat fenomena ghosting rekrutmen yang dialami oleh 41% kandidat.

Berdasarkan permasalahan tersebut, penelitian ini merumuskan pertanyaan mengenai:

- Bagaimana memanfaatkan teknologi Artificial Intelligence untuk mengklasifikasikan profil kompetensi pencari kerja dengan persyaratan industri secara akurat
- Bagaimana merancang sistem pendukung keputusan yang memberikan penjelasan transparan terkait skor kecocokan
- Bagaimana memetakan kesenjangan keterampilan (skill gap) secara objektif

Bisakerja dirancang sebagai **Career Decision Engine berbasis web** dengan fitur Job Fit Scoring otomatis dan Application Intelligence untuk melacak progres lamaran secara cerdas.

Proyek ini dipilih karena tim melihat peluang besar untuk mentransformasi proses pencarian kerja menjadi pengambilan keputusan strategis berbasis data yang diproyeksikan mampu meningkatkan peluang keberhasilan lamaran hingga **15–40%** guna mendukung produktivitas ekonomi menuju visi Indonesia Emas 2045.

## Cakupan Proyek dan Hasil Kerja

### Cakupan Proyek

Cakupan proyek Bisakerja difokuskan pada pengembangan sebuah sistem pendukung keputusan karir (**Career Decision Engine**) berbasis web yang dirancang untuk meningkatkan efisiensi pencarian kerja melalui integrasi data dan kecerdasan buatan.

Proyek ini memfokuskan batasan operasionalnya pada:

- Pengembangan algoritma Job Fit Scoring
- Skill Gap Analysis

Dengan durasi pengerjaan **4–5 minggu**, tim membagi tanggung jawab secara spesifik mulai dari:

- Pengadaan data pipeline otomatis
- Pemodelan AI yang transparan
- Pembangunan infrastruktur full-stack

Pendekatan interaksi:

- **Guest user**: akses terbatas (search & browse)
- **Authenticated user**: akses penuh (analysis, recommendation, tracking)

### Fitur Utama yang Menjadi Fokus Pengembangan

#### 1. Agregasi Lowongan Kerja

- Pengumpulan data dari:
  - Glints
  - Jobstreet
  - Kalibrr
  - Dealls

#### 2. Job Fit Scoring

- Skor kecocokan (0–100) berdasarkan:
  - Keahlian
  - Pengalaman
  - Preferensi

#### 3. Explainable AI (XAI)

- Penjelasan transparan terkait skor kecocokan

#### 4. Skill Gap Analysis

- Identifikasi gap kompetensi
- Rekomendasi prioritas skill

#### 5. Application Tracker

- Status:
  - Applied
  - Interview
  - Rejected

#### 6. User Preference Management

- Skill
- Lokasi
- Ekspektasi gaji

### Batasan di Luar Cakupan Proyek

- Mobile native app (Android/iOS)
- Auto apply ke platform eksternal
- Integrasi ATS perusahaan
- Payment gateway
- OCR CV tingkat lanjut

## Deskripsi Tugas dan Tanggung Jawab Tim

### Dzikri Albantani — Data Science

**Deskripsi:**

- Data scraping & EDA
- Automated scraping pipeline
- Streamlit dashboard
- A/B testing

**Output:**

- Dataset lowongan
- EDA report
- Scraping pipeline
- Dashboard validasi
- Dokumentasi teknis

### Tasya Anggraeni Firdaus — Data Science

**Deskripsi:**

- Problem definition
- Data pipeline & cleaning
- Automated scraper
- Trend analysis
- Validation

**Output:**

- Dokumen landasan masalah
- Dataset terstandarisasi
- Visualisasi data
- Dashboard validasi
- Dokumentasi final

### Linda David — AI Engineer

**Deskripsi:**

- Model Deep Learning (TensorFlow)
- Explainable AI (XAI)
- FastAPI inference
- Integration & testing

**Output:**

- Model `.keras`
- REST API
- Dokumentasi & testing evidence

### Salman Abdurrahman — Full-Stack Backend

**Deskripsi:**

- Arsitektur server & database
- Auth & security
- REST API
- Data pipeline integration
- AI integration

**Output:**

- Database schema
- API services
- Auth system
- Data integration
- AI connector
- Dokumentasi backend

### Agel Saputra — Full-Stack Frontend

**Deskripsi:**

- UI/UX design (Figma)
- Frontend development
- API integration
- AI integration

**Output:**

- Wireframe & prototype
- Responsive UI
- Fully integrated frontend

## Milestone Proyek

| Minggu | Deskripsi                       |
| ------ | ------------------------------- |
| M1     | Setup fondasi, dataset, UI/UX   |
| M2     | Data scraping, frontend slicing |
| M3     | Automation & model training     |
| M4     | Validation, XAI, tracker        |
| M5     | Integration AI + backend        |
| M6     | Testing & finalization          |

## Jadwal Pengerjaan

Penjadwalan ini mencakup aktivitas harian untuk monitoring progres serta sinkronisasi melalui weekly standup.

## Uraian Job Desk per Learning Path

### Data Science

- Data acquisition
- Cleaning & preprocessing
- Automated scraping
- Trend analysis
- Validation (Streamlit)

### Artificial Intelligence

- Model design (TensorFlow)
- Feature engineering
- Training
- Explainable AI
- FastAPI deployment

### Full Stack

- Backend (API, DB, auth)
- Frontend (UI, slicing)
- Integration (DS + AI)
- Testing & deployment

## Sumber Daya Proyek

### Bahasa Pemrograman

- TypeScript
- Python

### Framework

- React
- ExpressJS
- FastAPI
- TensorFlow
- Sentence-Transformers

### Database

- PostgreSQL

### ORM

- Prisma

### Tools

- Git
- GitHub
- TensorBoard
- Joblib

### Deployment

- Docker
- Heroku

### Dataset

- IndoTech-Job Dataset
- TechTalent-Profile Dataset
- ID-TechSkill Taxonomy

## Daftar Pustaka

[1] BPS, 2025  
https://m.kumparan.com/kumparanbisnis/pengangguran-jangka-panjang-mendominasi-31-susah-cari-kerja-lebih-dari-setahun-26BRtwRGX3o

[2] G. A. Adrian, 2025  
https://jurnalbisnismahasiswa.com/index.php/jurnal/article/download/926/488/4093

[3] A. Setiyana, 2024  
https://ejournal.brin.go.id/jki/article/download/5513/10605/38472

[4] SMERU, 2024  
https://smeru.or.id/sites/default/files/publication/wp_employers_and_jobseekers_eng_2024-10-8.pdf

[5] AIApply, 2025  
https://aiapply.co/blog/best-way-to-apply-for-jobs

[6] Ruangkerja, 2023  
https://www.ruangkerja.id/blog/fenomena-ghosting-dunia-kerja

[7] S. Roy et al., 2023  
https://www.researchgate.net/publication/400558524

[8] K. R. Chetan, 2025  
https://ijirt.org/publishedpaper/IJIRT189601_PAPER.pdf

[9] A. M. Salih et al., 2024  
https://www.scribd.com/document/824180839

## Rencana Manajemen Risiko dan Isu

### Strengths

- Decision Support System berbasis profil
- Job Fit Scoring + Skill Gap Analysis
- Data-driven problem

### Weaknesses

- Waktu terbatas (MVP)
- Ketergantungan dataset
- Validasi model awal

### Opportunities

- Demand tinggi talenta tech
- Segmen Gen Z besar
- Potensi integrasi edukasi

### Threats

- Bias AI
- Kompetitor besar
- Literasi digital rendah

### Mitigasi Risiko

- Fokus baseline model sejak minggu 4
- Weekly sync internal tim
- Version control (Git)
- Prioritasi fitur inti jika ada kendala
