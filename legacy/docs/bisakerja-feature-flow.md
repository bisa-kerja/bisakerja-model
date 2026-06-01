# Bisakerja — Feature Flow & User Journey

## 🧭 Global Navigation

- Home / Loker Page
- Mentoring
- AI CV Analyzer
- Profile
- Notification

# 🔐 AUTH & ENTRY FLOW

## START FLOW

- Home Page
- Register
- Login
- SSO Google
- Forgot Password
  - Kirim email
  - User menerima link reset password
  - Redirect ke halaman ganti password

# 🧑‍💻 ONBOARDING — JOB SEEKER

## Step 1 — Basic Info

- Username
- Email
- No WA
- Password
- Konfirmasi Password
- Upload Foto Profile

## Step 2 — CV Upload

- Upload CV (Optional / Skippable)

## Step 3 — Career Preferences

- Status:
  - Fresh Graduate
  - Early Career
  - Career Switcher
- Job seeking status:
  - Secepatnya
  - 1 bulan
  - 3 bulan
- Target role (searchable select + quick access common roles)
- Lokasi kerja:
  - Pilih provinsi → pilih kota
- Tipe kerja
- Skills / keahlian
- Range gaji
- Toggle:
  - Aktifkan notifikasi email sesuai preferensi

## Step 4 — Verification

- Email verification (OTP)
- Finish

# 🧑‍🏫 ONBOARDING — MENTOR

## Step 1 — Basic Info

- Username
- Email
- No WA
- Password
- Konfirmasi Password

## Step 2 — Professional Info

- Experience (multi add)
- Education (multi add)
- Top skills (multi select)

## Step 3 — Verification

- Email verification (OTP)
- Finish

# 🏠 HOME / LOKER PAGE

## 🔍 Search & Filter

### Search Input

- Search job by title (text input)

### Filter

- Expertise
- Jenis pekerjaan
- Tipe pekerjaan
- Lokasi
- Gaji
- Category job by divisi

### Sorting

- Paling relevan
- Baru dipost
- Gaji tertinggi
- Gaji terendah

### Additional

- Toggle:
  - Aktifkan notifikasi email sesuai preferensi

## 📋 Job List

### Per Job Card

- Nama role
- Nama perusahaan
- Logo perusahaan
- Jenis pekerjaan
- Tipe pekerjaan
- Lokasi
- Experience
- Gaji
- Diposting kapan
- Bookmark

### Behavior

- Infinite scroll

## 📄 Job Detail

- Nama role
- Nama perusahaan
- Logo perusahaan
- Deskripsi pekerjaan
- Jenis pekerjaan
- Tipe pekerjaan
- Lokasi
- Experience
- Gaji
- Diposting kapan

### Actions

- Bookmark
- Share link
- Link ke platform (apply)
  - Redirect ke platform external

# 📌 BOOKMARK & TRACKER

## Bookmark Pekerjaan

- List job yang disimpan

## Tracker Pekerjaan

- Otomatis masuk saat klik "Lamar"

### Status

- Diproses
- Interview
- Ditolak
- Diterima

### Features

- Search job yang sudah disimpan

# 🤖 AI CV ANALYZER

## Input

- Pilih bahasa output:
  - Bahasa Indonesia
  - English

- Upload CV

- Pilih job listing untuk perbandingan:
  - Dari bookmark
  - Search job by title

## Output Analysis

### 1. Overall Impression

- Summary CV
- Contoh:
  - Score 85%
  - Insight kualitas CV

### 2. Job Fit Alignment

- Persentase kecocokan skill & experience terhadap job

### 3. ATS Friendliness Score

- Skor ATS
- Saran perbaikan

### 4. Keyword Optimization

- Rekomendasi keyword tambahan

### 5. Experience Quantification

- Evaluasi pencapaian berbasis angka/data

### 6. Actionable Improvement

- 3 langkah tercepat untuk improve CV

## Additional Feature

- Generate CV terbaik berdasarkan hasil analisis

# 🧑‍🏫 MENTORING

## 🔍 Search & Filter Mentor

### Search Input

- Search mentor

### Filter

- Expertise
- Jenis pekerjaan
- Category mentor by divisi

### Sorting

- A - Z

## 📋 Mentor List

### Per Card

- Foto profile
- Nama role saat ini
- Nama perusahaan
- Logo perusahaan
- Education terakhir
- Badge skills
- Experience
- Slot availability

## 📄 Mentor Detail

- Foto profile
- Bio mentor
- Nama role saat ini
- Nama perusahaan
- Logo perusahaan
- Education terakhir
- Badge skills / topik keahlian
- Experience
- Slot availability
- Jadwal availability
- Statistik

## 📊 Dashboard Mentoring (Mentor)

- Foto Profile
- Username
- Latest job experience

### Features

- Atur jadwal mentoring

# 🔔 NOTIFICATION

- Popup notifikasi
- Notifikasi terkait:
  - Job recommendation
  - Status lamaran
  - Aktivitas mentoring

# 👤 PROFILE

## Informasi Umum

- Email
- No WA
- Username
- Ganti password

## Data User

- Foto profile
- Latest job experience

# ⚙️ PREFERENSI KARIR (Editable)

- Status:
  - Fresh Graduate
  - Early Career
  - Career Switcher
- Job seeking status:
  - Secepatnya
  - 1 bulan
  - 3 bulan
- Target role (searchable select)
- Lokasi kerja
- Tipe kerja
- Skills
- Range gaji
- Toggle notifikasi email

# 🧩 NOTES (SYSTEM BEHAVIOR)

- Job recommendation bisa based on:
  - Preference user
  - Bookmark history
- AI Analyzer terhubung ke:
  - Job listing
  - CV parsing
- Tracker auto update saat user klik "apply"
