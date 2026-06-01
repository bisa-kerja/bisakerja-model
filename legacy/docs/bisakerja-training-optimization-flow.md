 # Alur Optimasi Pelatihan Model Bisakerja

Dokumen ini menjadi alur kerja pelatihan model deep learning untuk Bisakerja. Fokusnya adalah pencocokan semantik antara CV/profil kandidat dan deskripsi lowongan, skor kecocokan kerja, analisis CV, dan persiapan rekomendasi lowongan.

Sasaran alur pelatihan:

- Teknologi utama: TensorFlow, Keras, Hugging Face Transformers, Sentence Transformers, Scikit-learn, Pandas, NumPy.
- Format model utama: `.keras`.
- Target minimum validasi internal: akurasi >= 85% dan MAE <= 0.02.
- Arsitektur wajib punya komponen custom lanjutan: minimal layer custom, fungsi loss custom, atau callback custom.
- Output siap production: model punya versi, bisa direproduksi, bisa dilacak lewat log, bisa diexport, dan mudah diintegrasikan ke API inferensi.

Catatan penting: target akurasi >= 85% dan MAE <= 0.02 adalah gerbang rilis, bukan jaminan. Jika label lemah, noisy, atau tidak seimbang, model tidak boleh dipaksa lolos dengan kebocoran data. Perbaiki pelabelan, pembagian data, dan evaluasi lebih dulu.

Catatan bahasa dokumen:

- Penjelasan memakai Bahasa Indonesia.
- Nama endpoint, field JSON, enum, nama library, nama model, path file, dan kode tetap mengikuti kontrak teknis aslinya.

Pembaruan asumsi data:

- Dataset terbaru akan sepenuhnya berbahasa Inggris untuk `description`, `requirements`, `skills`, field profil, dan teks turunan.
- File CSV terbaru belum diexport, jadi alur harus punya gerbang validasi bahasa sebelum pelatihan.
- Jika raw file lama masih mengandung Bahasa Indonesia, jangan jalankan full training. Jalankan hanya smoke test pipeline.
- Karena dataset target hanya berbahasa Inggris, model embedding default harus mengutamakan Bahasa Inggris. Model multibahasa hanya fallback untuk data campuran atau migrasi.

## 1. Konteks Data Saat Ini

File data awal:

- `indotech_job_cleaned.csv`: data lowongan kerja.
- `techtalent_profile_cleaned.csv`: data profil kandidat sintetik/terstruktur.

Kolom penting `indotech_job_cleaned.csv`:

- Identitas: `job_id`, `company_name`, `title`, `normalized_title`, `category`.
- Teks job: `description`, `requirement_summary`, `requirements_concat`.
- Metadata: `work_type`, `employment_type`, `experience_level`, `province`, `city`.
- Salary: `salary_min`, `salary_max`, `salary_currency`, `salary_display`, `has_salary_info`.
- Skill: `skills_top_10_names`, `skills_clean`.
- Bahasa dan status: `language_signal`, `status`, `source_posted_at`.

Kolom penting `techtalent_profile_cleaned.csv`:

- `ID`
- `Skills`
- `Projects`
- `Education`
- `Experience`
- `Job_Role`
- `Required_Skills`

Masalah utama data saat ini:

- Belum ada label eksplisit `fit_score`, `match_label`, atau histori interaksi user-job.
- Profil kandidat lebih ringkas daripada CV nyata.
- Job text panjang dan noisy.
- Skill overlap bisa dibuat, tetapi ground truth kecocokan perlu dibangun.
- File mentah yang ada di repo saat dokumen ini ditulis masih berisi sinyal `ID`/`MIXED`; dataset final Bahasa Inggris belum diexport.

Implikasi:

- Training awal harus dimulai dari weak supervision + rule-based label.
- Validasi akhir butuh human-reviewed sample agar metrik tidak semu.
- Recommendation metrics seperti precision@k dan recall@k butuh pasangan kandidat-job relevan.
- Pelatihan penuh harus menunggu export dataset Bahasa Inggris terbaru agar embedding, label, dan evaluasi konsisten.

## 1A. Gerbang Validasi Bahasa Inggris

Sebelum pelatihan penuh, validasi bahwa dataset sudah hanya berbahasa Inggris.

Kolom wajib berbahasa Inggris:

- Jobs: `description`, `requirement_summary`, `requirements_concat`, `skills_clean`, `skills_top_10_names`, `title`, `normalized_title`.
- Profiles: `Skills`, `Projects`, `Education`, `Experience`, `Job_Role`, `Required_Skills`.

Aturan lulus:

```text
language_pass = true only if:
  english_ratio >= 0.98 for sampled long-text fields
  indonesian_marker_count == 0 for critical fields
  language_signal in {"EN", "MIXED"} is audited and final output is EN
  no old Indonesian phrases remain in requirements/description/profile text
```

Contoh validasi cepat:

```python
INDONESIAN_MARKERS = {
    "kami", "mencari", "pengalaman", "pendidikan", "minimal", "lulusan",
    "bersedia", "ditempatkan", "kemampuan", "tanggung jawab", "kualifikasi",
    "diutamakan", "memiliki", "menguasai", "bekerja sama",
}

def count_indonesian_markers(text):
    value = str(text).lower()
    return sum(marker in value for marker in INDONESIAN_MARKERS)

text_cols_jobs = [
    "description",
    "requirement_summary",
    "requirements_concat",
    "skills_clean",
    "skills_top_10_names",
]
text_cols_profiles = [
    "Skills",
    "Projects",
    "Education",
    "Experience",
    "Job_Role",
    "Required_Skills",
]

job_marker_count = sum(
    jobs[col].fillna("").map(count_indonesian_markers).sum()
    for col in text_cols_jobs
    if col in jobs.columns
)
profile_marker_count = sum(
    profiles[col].fillna("").map(count_indonesian_markers).sum()
    for col in text_cols_profiles
    if col in profiles.columns
)

assert job_marker_count == 0, f"Indonesian markers found in jobs: {job_marker_count}"
assert profile_marker_count == 0, f"Indonesian markers found in profiles: {profile_marker_count}"
```

Catatan:

- Marker gate bukan language detector sempurna. Tujuannya menangkap sisa dataset lama.
- Untuk final, tambahkan `fasttext-langdetect`, `lingua-language-detector`, atau audit sample manual.
- Simpan `language_report.json` sebagai artifact tiap run.

## 2. Output Model

Model training sebaiknya menghasilkan tiga output:

1. `fit_score`
   - Range 0 sampai 1.
   - Dipakai sebagai Job Fit Score 0 sampai 100 di produk.
   - Metric utama: MAE, RMSE, Pearson/Spearman optional.

2. `fit_class`
   - Kelas: `low_fit`, `medium_fit`, `high_fit`.
   - Bisa juga binary: `not_recommended`, `recommended`.
   - Metric utama: accuracy, macro F1, precision, recall.

3. `recommendation_score`
   - Skor ranking untuk mengurutkan rekomendasi lowongan.
   - Metrik utama: precision@k, recall@k, MAP@k/NDCG@k opsional, cosine similarity.

Output multi-tugas disarankan karena Bisakerja butuh skor numerik, kategori yang bisa dijelaskan, dan ranking.

## 2A. Kontrak Endpoint Backend

`openapi.json` saat ini punya dua endpoint AI yang harus didukung model:

- `POST /api/v1/ai/job-fit`
- `POST /api/v1/ai/cv-analyzer`

Backend API tetap menjadi pemilik autentikasi, `jobId`, profil pengguna, preferensi, bookmark, unggah PDF, dan penyimpanan hasil. Model API FastAPI sebaiknya menerima muatan data yang sudah dirapikan oleh backend, bukan langsung mengikuti muatan publik dari pengguna.

Kontrak internal yang disarankan untuk Model API:

```text
Backend API
  -> ambil user, profil, preferensi, job, dan file CV jika ada
  -> normalisasi muatan data
  -> panggil Model API FastAPI
  -> validasi respons dengan skema ketat
  -> bungkus respons sesuai format backend
```

Target output untuk `/api/v1/ai/job-fit`:

| Field backend                       | Sumber model / aturan                 | Catatan pelatihan                                                                   |
| ----------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------- |
| `fitScore`                          | `fit_score * 100`                     | Regresi utama, dibulatkan 0 sampai 100                                              |
| `readinessLevel`                    | pemetaan dari `fit_score` + gap wajib | Kelas turunan: siap, siap dengan gap kecil, perlu persiapan, belum direkomendasikan |
| `recommendation.decision`           | pemetaan dari skor dan risiko gap     | Jangan hanya dari skor; skill wajib hilang harus menurunkan keputusan               |
| `recommendation.successProbability` | kalibrasi dari `fit_score`            | Perlu cek kalibrasi agar tidak terlalu percaya diri                                 |
| `breakdown.skillMatch.score`        | `skill_coverage` dan `skill_overlap`  | Harus bisa dijelaskan dari fitur skill                                              |
| `breakdown.experienceMatch.score`   | `experience_match`                    | Jangan murni dari layer neural                                                      |
| `breakdown.preferenceMatch.score`   | lokasi, work type, salary, preferensi | Bisa berbasis aturan                                                                |
| `skillGaps`                         | skill wajib yang belum cocok          | Wajib deterministik dan mudah diaudit                                               |

Target output untuk `/api/v1/ai/cv-analyzer`:

| Field backend                             | Sumber model / aturan                | Catatan pelatihan                           |
| ----------------------------------------- | ------------------------------------ | ------------------------------------------- |
| `overallImpression.score`                 | gabungan skor CV dan job-fit         | Output akhir untuk kualitas CV terhadap job |
| `jobFitAlignment.score`                   | `fit_score * 100` + sinyal kecocokan | Konsisten dengan endpoint job-fit           |
| `jobFitAlignment.matchedSignals`          | skill/keyword/proyek yang cocok      | Deterministik dari ekstraksi fitur          |
| `jobFitAlignment.missingSignals`          | skill/keyword wajib yang hilang      | Deterministik                               |
| `atsFriendliness.score`                   | fitur struktur CV                    | Butuh dataset/aturan khusus ATS             |
| `keywordOptimization.recommendedKeywords` | gap keyword job terhadap CV          | Berbasis aturan dari requirement            |
| `experienceQuantification.score`          | jumlah bukti metrik/dampak di CV     | Butuh fitur parsing CV                      |
| `actionableImprovements`                  | template berbasis gap                | Aman dibuat berbasis aturan dulu            |

Implikasi pelatihan:

- Model utama cukup memprediksi skor dan kelas kecocokan.
- Rincian skill gap, keyword, preferensi, dan alasan harus berasal dari fitur input yang deterministik.
- CV analyzer butuh pipeline ekstraksi PDF ke teks, normalisasi bagian CV, ekstraksi skill, ekstraksi pengalaman, dan fitur ATS.
- Output FastAPI harus memakai skema Pydantic ketat agar backend cepat tahu jika respons model berubah.

Persiapan endpoint ketiga:

- Endpoint rekomendasi job nanti bisa memakai artefak yang sama.
- Alur: embedding profil/CV -> ambil kandidat job top 100 sampai 500 -> ranking dengan model `.keras` -> kembalikan daftar lowongan.
- Karena itu pelatihan sekarang harus menyimpan embedding job, skema fitur, dan mapping `job_id`.

Contoh skema internal FastAPI yang disarankan:

```python
from pydantic import BaseModel, ConfigDict, Field


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str


class JobFitModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    fit_score: float = Field(ge=0, le=1)
    fit_score_100: int = Field(ge=0, le=100)
    readiness_level: str
    decision: str
    success_probability: float = Field(ge=0, le=1)
    matched_skills: list[str]
    missing_skills: list[str]
    skill_match_score: int = Field(ge=0, le=100)
    experience_match_score: int = Field(ge=0, le=100)
    preference_match_score: int = Field(ge=0, le=100)
    model: ModelInfo
```

Prinsip:

- FastAPI memakai `response_model` agar respons tervalidasi.
- Pydantic memakai `extra="forbid"` agar field liar tidak lolos.
- Backend API boleh mengubah snake_case internal menjadi camelCase sesuai OpenAPI publik.
- Model API tidak boleh mengarang field yang tidak bisa dijelaskan dari skor, fitur, atau aturan deterministik.

## 2B. Ringkasan Langkah untuk Pemula

Urutan kerja paling aman:

1. Export dataset terbaru yang sudah berbahasa Inggris.
2. Jalankan validasi skema dan gerbang Bahasa Inggris.
3. Bangun teks profil dan teks lowongan dengan format konsisten.
4. Normalisasi skill, pengalaman, lokasi, preferensi, dan salary.
5. Buat pasangan profil-lowongan dengan pembuatan kandidat, bukan full cross join.
6. Buat label lemah dari skill, semantik, role, pengalaman, dan preferensi.
7. Hitung embedding profil dan lowongan.
8. Simpan embedding, fitur, label, dan split ke artifact.
9. Latih model Keras dua tower dari embedding dan fitur terstruktur.
10. Evaluasi dengan data validasi, data test, dan sampel review manual.
11. Simpan model `.keras`, artifact preprocessing, manifest, dan laporan metrik.
12. Uji model bisa dimuat ulang dan output sesuai kontrak FastAPI.

Jika salah satu langkah gagal, berhenti dan perbaiki penyebabnya. Jangan lanjut ke pelatihan besar sebelum uji cepat lulus.

## 3. Dataset Ideal

Struktur dataset training ideal:

```text
datasets/
  raw/
    indotech_job_cleaned.csv
    techtalent_profile_cleaned.csv
  interim/
    jobs_normalized.parquet
    profiles_normalized.parquet
  labels/
    profile_job_pairs_v1.parquet
    human_reviewed_pairs_v1.parquet
  processed/
    train.parquet
    val.parquet
    test.parquet
    job_embeddings.npy
    profile_embeddings.npy
  metadata/
    skill_taxonomy.csv
    label_rules.yaml
    split_manifest.json
```

Schema ideal `profile_job_pairs_v1.parquet`:

| Kolom                 | Tipe        | Sumber    | Fungsi                                               |
| --------------------- | ----------- | --------- | ---------------------------------------------------- |
| `pair_id`             | string      | generated | ID pasangan kandidat-job                             |
| `profile_id`          | string/int  | profile   | kandidat                                             |
| `job_id`              | string      | job       | lowongan                                             |
| `profile_text`        | string      | generated | gabungan skill, project, role, education, experience |
| `job_text`            | string      | generated | gabungan title, description, requirement, skills     |
| `profile_skills`      | list/string | profile   | feature skill                                        |
| `job_skills`          | list/string | job       | feature skill                                        |
| `role_match`          | float       | rule      | similarity role                                      |
| `skill_overlap`       | float       | rule      | Jaccard/weighted overlap                             |
| `experience_match`    | float       | rule      | match level pengalaman                               |
| `location_match`      | float       | optional  | preferensi lokasi                                    |
| `salary_match`        | float       | optional  | preferensi gaji                                      |
| `semantic_similarity` | float       | embedding | cosine profile-job                                   |
| `fit_score`           | float       | label     | target 0 sampai 1                                    |
| `fit_class`           | int/string  | label     | low/medium/high                                      |
| `source_label`        | string      | metadata  | weak/human/interaction                               |
| `split`               | string      | manifest  | train/val/test                                       |

## 4. Strategi Labeling

Urutan labeling modern:

1. Baseline label lemah
   - Buat label awal dari aturan deterministik.
   - Cocok untuk bootstrapping.
   - Jangan jadikan final truth.

2. Human-reviewed labels
   - Ambil sampel berimbang dari low/medium/high.
   - Minimal 300 sampai 1000 pair untuk validasi MVP.
   - Reviewer menilai fit 0 sampai 100 dan alasan.

3. Interaction labels
   - Setelah produk jalan: bookmark, apply click, interview, rejected, accepted.
   - Gunakan implicit feedback.
   - Apply click bukan selalu match positif; accepted/interview lebih kuat.

4. Continuous relabeling
   - Sampling data dengan confidence rendah.
   - Sampling job baru, role baru, dan skill baru.

Formula weak label awal:

```text
fit_score =
  0.40 * skill_overlap_weighted +
  0.25 * semantic_similarity +
  0.15 * role_similarity +
  0.10 * experience_match +
  0.05 * education_match +
  0.05 * preference_match
```

Mapping class:

```text
fit_score >= 0.75 -> high_fit
0.45 <= fit_score < 0.75 -> medium_fit
fit_score < 0.45 -> low_fit
```

Negative sampling:

- Positive pair: role/skill/expertise dekat.
- Hard negative: semantic similarity tinggi tetapi skill wajib kurang.
- Random negative: role/category jauh.
- Rasio awal: 1 positive : 2 hard negative : 2 random negative.

Aturan anti-leakage:

- Jangan pakai `Required_Skills` dari profil sebagai target langsung jika itu dibuat dari role yang sama dengan job target.
- Split berdasarkan `profile_id` dan `job_id`, bukan row acak murni.
- Test set harus berisi job dan profile yang tidak muncul di train jika ingin mengukur generalisasi.

## 5. Text Construction

Bangun teks model secara konsisten.

Profile text:

```text
role: {Job_Role}
skills: {Skills}
projects: {Projects}
education: {Education}
experience: {Experience}
required skills: {Required_Skills}
```

Job text:

```text
title: {normalized_title}
category: {category}
description: {description}
requirements: {requirements_concat}
skills: {skills_clean}
work type: {work_type}
employment: {employment_type}
experience level: {experience_level}
location: {city}, {province}
```

Best practice:

- Pisahkan field dengan label eksplisit agar encoder memahami konteks.
- Simpan text versi raw, cleaned, dan normalized.
- Jangan menghapus kata teknis seperti `C++`, `.NET`, `Node.js`, `CI/CD`, `S1`, `D3`.
- Jangan lower-case semua jika model multilingual/cased masih memanfaatkan casing.

## 6. NLP Preprocessing Pipeline

Pipeline:

1. Load data
   - Pandas `read_csv`.
   - Validasi missing values dan duplicate IDs.
   - Convert tanggal dan salary ke tipe numerik.

- Jalankan gerbang validasi Bahasa Inggris sebelum pembuatan pasangan data.

2. Normalize text
   - Decode HTML entities.
   - Normalize whitespace.
   - Remove broken unicode/BOM.
   - Keep punctuation penting untuk skill teknis.

3. Normalize skill
   - Split dengan delimiter `|`, `,`, `;`, `||`.
   - Trim whitespace.
   - Alias map: `js -> javascript`, `tf -> tensorflow`, `postgres -> postgresql`.
   - Deduplicate.

4. Normalize categorical
   - `experience_level`: ENTRY_LEVEL, JUNIOR, MID, SENIOR, LEAD.
   - `work_type`: ONSITE, HYBRID, REMOTE.
   - `employment_type`: FULL_TIME, CONTRACT, INTERN, PART_TIME.

5. Pair generation
   - Cross join terbatas berdasarkan category/role kandidat.
   - Add hard negatives dari top semantic candidate yang gagal skill wajib.

6. Feature generation
   - Embeddings.
   - Cosine similarity.
   - Skill overlap.
   - Experience match.
   - Salary match.
   - Location/work type match.

7. Split
   - Train/validation/test.
   - Stratified by `fit_class`.
   - Group-aware by `profile_id` atau `job_id`.

8. Export processed data
   - Parquet untuk tabular.
   - `.npy` atau `.npz` untuk embeddings.
   - JSON manifest untuk seed, version, split.

## 7. Tokenization Strategy

Untuk transformer encoder:

- Gunakan `AutoTokenizer.from_pretrained(model_name)`.
- Pakai `padding=True` atau `padding='max_length'`.
- Pakai `truncation=True`.
- Pakai `max_length=256` untuk profile text pendek, `max_length=384` atau 512 untuk job text.
- Untuk mixed precision GPU, set padding ke multiple of 8 jika tooling mendukung.

Untuk pipeline awal yang export `.keras` lebih stabil:

- Generate embeddings offline.
- Keras model menerima tensor numerik, bukan raw string.
- Tokenizer dan encoder disimpan sebagai preprocessing artifact terpisah.

Alasan:

- `.keras` paling aman untuk menyimpan model Keras murni dengan input numerik.
- Hugging Face tokenizer bukan layer Keras native.
- Transformer penuh di dalam model menambah ukuran, latensi, dan risiko serialization custom.

## 8. Perbandingan Model Embedding

Karena dataset target dan output model core saat ini difokuskan ke Bahasa Inggris, gunakan model English retrieval yang kuat sebagai default training.

Default training embedding model:

```text
intfloat/e5-base-v2
```

| Model                                     | Bahasa      | Kelebihan                         | Kekurangan                     | Rekomendasi                     |
| ----------------------------------------- | ----------- | --------------------------------- | ------------------------------ | ------------------------------- |
| `intfloat/e5-base-v2`                     | Inggris     | kuat untuk retrieval + matching   | perlu prefix query/passage     | default training utama          |
| `sentence-transformers/all-mpnet-base-v2` | Inggris     | kualitas STS kuat                 | kurang retrieval-oriented      | pembanding semantic baseline    |
| `sentence-transformers/all-MiniLM-L6-v2`  | Inggris     | sangat cepat, murah, stabil       | kualitas di bawah model base   | baseline cepat saja             |
| `BAAI/bge-base-en-v1.5`                   | Inggris     | kuat untuk pencarian semantik     | perlu evaluasi threshold ulang | pembanding retrieval benchmark  |
| `intfloat/e5-large-v2`                    | Inggris     | kualitas tinggi                   | mahal saat pelatihan/inferensi | benchmark akhir                 |
| `BAAI/bge-large-en-v1.5`                  | Inggris     | kualitas tinggi                   | VRAM/latensi lebih besar       | benchmark offline               |
| `BAAI/bge-m3`                             | multibahasa | kuat jika data campur bahasa      | lebih berat                    | fallback jika gerbang EN gagal  |
| `intfloat/multilingual-e5-base`           | multibahasa | kuat untuk migrasi ID/EN          | bukan default data Inggris     | fallback migrasi                |

Pilihan awal:

- Default training utama: `intfloat/e5-base-v2`.
- Pembanding semantic baseline: `sentence-transformers/all-mpnet-base-v2`.
- Baseline cepat: `sentence-transformers/all-MiniLM-L6-v2`.
- Benchmark akhir: `intfloat/e5-large-v2` atau `BAAI/bge-large-en-v1.5`.

Prompting:

- Untuk E5/BGE retrieval, encode profile sebagai query dan job sebagai passage/document.
- Dengan Sentence Transformers terbaru, prefer `encode_query()` dan `encode_document()` jika model mendukung prompt routing.
- Jika memakai `encode()`, set prefix manual:

```text
profile_text = "query: " + profile_text
job_text = "passage: " + job_text
```

Embedding normalization:

- Set `normalize_embeddings=True`.
- Setelah normalized, dot product setara cosine similarity.
- Simpan model name, revision jika dipin, embedding dim, pooling, dan prompt strategy di manifest.

## 9. Semantic Similarity

Hitung cosine similarity:

```text
cosine(profile_embedding, job_embedding)
```

Gunakan untuk:

- Feature model.
- Label lemah.
- Hard negative mining.
- Retrieval candidate generation.
- Evaluation semantic quality.

Jangan hanya memakai cosine similarity sebagai final score. Job fit juga dipengaruhi skill wajib, pengalaman, salary, lokasi, dan preferensi.

## 10. Feature Engineering

Feature numerik disarankan:

- `semantic_similarity`: cosine profile-job.
- `skill_jaccard`: intersection / union.
- `skill_coverage`: matched required skills / required job skills.
- `missing_required_skill_count`.
- `experience_match`: 0 sampai 1.
- `education_match`: 0 sampai 1.
- `salary_match`: 0 sampai 1.
- `location_match`: 0 sampai 1.
- `work_type_match`: 0 sampai 1.
- `role_similarity`: cosine embedding role text.
- `category_match`: binary/categorical encoded.

Feature text embedding:

- `profile_embedding`: vector 384/768/1024 dim.
- `job_embedding`: vector 384/768/1024 dim.
- `abs_diff`: `abs(profile_embedding - job_embedding)`.
- `elementwise_product`: `profile_embedding * job_embedding`.

Input final:

```text
[profile_embedding, job_embedding, abs_diff, elementwise_product, structured_features]
```

## 11. Arsitektur Model yang Disarankan

Arsitektur utama: two-tower interaction model.

Alasan:

- Cocok untuk semantic matching.
- Bisa dipakai untuk scoring pair kandidat-job.
- Bisa dikembangkan untuk recommendation retrieval.
- Kompatibel `.keras` jika input berupa tensor numerik.
- Mudah dimonitor, di-debug, dan di-export.

Alur:

```text
Profile text -> embedding model -> profile_embedding ----\
                                                        Interaction Layer -> Dense Head -> fit_score
Job text     -> embedding model -> job_embedding --------/                 -> fit_class
Structured features ------------------------------------/                  -> recommendation_score
```

Model detail:

```text
Input A: profile_embedding [dim]
Input B: job_embedding [dim]
Input C: structured_features [n]

Tower A:
  Dense 512 + LayerNorm + GELU + Dropout
  Dense 256 + LayerNorm + GELU

Tower B:
  Dense 512 + LayerNorm + GELU + Dropout
  Dense 256 + LayerNorm + GELU

Custom Layer:
  CosineInteractionLayer
  output: cosine, abs_diff, product

Fusion:
  concat(profile_repr, job_repr, interaction, structured_features)
  Dense 256 + BatchNorm/LayerNorm + GELU + Dropout
  Dense 128 + GELU

Outputs:
  fit_score: Dense(1, sigmoid)
  fit_class: Dense(3, softmax)
  recommendation_score: Dense(1, sigmoid)
```

Functional API lebih disarankan untuk versi pertama karena:

- Input/output multi-head jelas.
- Serialisasi `.keras` lebih sederhana.
- Model graph mudah divisualisasi.

Model Subclassing digunakan jika:

- Perlu custom `train_step`.
- Perlu pairwise ranking loss rumit.
- Perlu TensorFlow Recommenders task dengan retrieval/ranking.

## 12. Advanced Custom Component

Minimal implementasi: Custom Layer `CosineInteractionLayer`.

Contoh kode:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@keras.saving.register_keras_serializable(package="bisakerja")
class CosineInteractionLayer(layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        left, right = inputs
        left_norm = tf.math.l2_normalize(left, axis=-1, epsilon=self.epsilon)
        right_norm = tf.math.l2_normalize(right, axis=-1, epsilon=self.epsilon)
        cosine = tf.reduce_sum(left_norm * right_norm, axis=-1, keepdims=True)
        abs_diff = tf.abs(left - right)
        product = left * right
        return tf.concat([cosine, abs_diff, product], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config
```

Tambahan custom callback untuk release gate:

```python
class TargetGateCallback(keras.callbacks.Callback):
    def __init__(self, min_accuracy=0.85, max_mae=0.02):
        super().__init__()
        self.min_accuracy = min_accuracy
        self.max_mae = max_mae

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("val_fit_class_accuracy")
        mae = logs.get("val_fit_score_mae")
        if acc is not None and mae is not None:
            logs["target_gate_passed"] = float(acc >= self.min_accuracy and mae <= self.max_mae)
```

Custom component harus punya:

- Unit test shape.
- Tes serialisasi `model.save()` lalu `keras.models.load_model()`.
- Logging ke TensorBoard.

## 13. Loss Function Strategy

Multi-task loss:

```text
total_loss =
  0.50 * fit_score_loss +
  0.30 * fit_class_loss +
  0.20 * recommendation_loss
```

Rekomendasi loss:

- `fit_score_loss`: Huber atau MAE/MSE.
- `fit_class_loss`: categorical cross entropy atau sparse categorical cross entropy.
- `recommendation_loss`: binary cross entropy untuk implicit labels, atau pairwise ranking loss untuk ranking advanced.

Custom loss optional:

```python
def weighted_huber(y_true, y_pred):
    huber = keras.losses.Huber(delta=0.05, reduction="none")(y_true, y_pred)
    weights = 1.0 + 2.0 * tf.cast(y_true >= 0.75, tf.float32)
    return tf.reduce_mean(huber * weights)
```

Alasan:

- Job high-fit lebih penting untuk produk recommendation.
- Huber lebih stabil dari MSE pada noisy weak labels.
- MAE tetap dipantau sebagai target bisnis.

## 14. Batching Strategy

Batching untuk precomputed embeddings:

- Batch size CPU: 64 sampai 256.
- Batch size GPU 8 GB: 256 sampai 1024.
- Batch size GPU 16 GB: 1024 sampai 4096.
- Gunakan `tf.data.Dataset`.
- Gunakan `shuffle`, `batch`, `cache` jika muat RAM, `prefetch(tf.data.AUTOTUNE)`.

Batching untuk transformer fine-tuning:

- Batch size GPU 8 GB: 8 sampai 16.
- Batch size GPU 16 GB: 16 sampai 32.
- Gunakan gradient accumulation jika batch efektif perlu lebih besar.
- Sequence length dipantau karena bottleneck utama ada di attention.

Negative sampling per batch:

- Pastikan tiap batch punya positive dan negative.
- Untuk ranking, masukkan hard negative dalam batch.
- Hindari batch semua low-fit karena class imbalance.

## 15. Training Pipeline

Pipeline training:

```text
1. Load config
2. Set seed
3. Load raw CSV
4. Validate schema
5. Validasi gerbang Bahasa Inggris
6. Normalize jobs/profiles
7. Build profile-job pairs
8. Generate weak labels
9. Generate embeddings
10. Build structured features
11. Split train/val/test
12. Build tf.data datasets
13. Build Keras model
14. Train with callbacks
15. Evaluate holdout test
16. Save `.keras`, metrics, config, plots, logs
17. Validate save-load parity
18. Register model artifact
```

Contoh kode:

```python
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

jobs = pd.read_csv("datasets/raw/indotech_job_cleaned.csv")
profiles = pd.read_csv("datasets/raw/techtalent_profile_cleaned.csv")

language_report = validate_english_dataset(jobs, profiles)
save_json(language_report, "training/reports/language_report_v1.json")
assert language_report["passed"], language_report

jobs = normalize_jobs(jobs)
profiles = normalize_profiles(profiles)
pairs = build_profile_job_pairs(profiles, jobs)
pairs = add_weak_labels(pairs)
pairs = add_embeddings_and_features(pairs)

train_df, temp_df = train_test_split(
    pairs,
    test_size=0.30,
    random_state=SEED,
    stratify=pairs["fit_class"],
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["fit_class"],
)

# Split acak hanya untuk baseline cepat. Metrik final wajib memakai group-aware split.

train_ds = make_tf_dataset(train_df, batch_size=512, shuffle=True)
val_ds = make_tf_dataset(val_df, batch_size=512, shuffle=False)
test_ds = make_tf_dataset(test_df, batch_size=512, shuffle=False)

model = build_bisakerja_matcher(
    embedding_dim=384,
    structured_dim=12,
    dropout=0.20,
)

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
    loss={
        "fit_score": keras.losses.Huber(delta=0.05),
        "fit_class": keras.losses.SparseCategoricalCrossentropy(),
        "recommendation_score": keras.losses.BinaryCrossentropy(),
    },
    loss_weights={
        "fit_score": 0.50,
        "fit_class": 0.30,
        "recommendation_score": 0.20,
    },
    metrics={
        "fit_score": [keras.metrics.MeanAbsoluteError(name="mae")],
        "fit_class": [keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        "recommendation_score": [keras.metrics.AUC(name="auc")],
    },
)

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir="training/logs/bisakerja_matcher_v1",
        histogram_freq=0,
        write_graph=True,
        update_freq="epoch",
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="training/checkpoints/bisakerja_matcher_best.keras",
        monitor="val_fit_score_mae",
        mode="min",
        save_best_only=True,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_fit_score_mae",
        mode="min",
        patience=8,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_fit_score_mae",
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    ),
    TargetGateCallback(min_accuracy=0.85, max_mae=0.02),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=80,
    callbacks=callbacks,
)

test_metrics = model.evaluate(test_ds, return_dict=True)
model.save("training/artifacts/bisakerja_matcher_v1.keras")
save_metrics_json(test_metrics, "training/reports/test_metrics_v1.json")
```

Untuk evaluasi final, split acak di atas hanya baseline cepat. Split production harus group-aware agar profile/job yang sama tidak bocor ke train dan test.

Pembagian data yang disarankan:

```python
from sklearn.model_selection import StratifiedGroupKFold

splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
groups = pairs["profile_id"].astype(str)  # profile-holdout: test kandidat baru

train_idx, test_idx = next(
    splitter.split(pairs, y=pairs["fit_class"], groups=groups)
)
train_full_df = pairs.iloc[train_idx].reset_index(drop=True)
test_df = pairs.iloc[test_idx].reset_index(drop=True)

val_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
val_groups = train_full_df["profile_id"].astype(str)
train_idx, val_idx = next(
    val_splitter.split(train_full_df, y=train_full_df["fit_class"], groups=val_groups)
)
train_df = train_full_df.iloc[train_idx].reset_index(drop=True)
val_df = train_full_df.iloc[val_idx].reset_index(drop=True)
```

Tambahkan satu evaluasi tambahan dengan `groups = pairs["job_id"]` untuk mengukur generalisasi ke lowongan baru. Jangan pakai `profile_id + job_id` sebagai group karena hampir semua pair unik dan tidak mencegah leakage profile/job.

Jika `StratifiedGroupKFold` gagal karena class terlalu kecil:

- Kurangi jumlah class bucket sementara.
- Tambah pair untuk class langka.
- Pakai `GroupShuffleSplit`, lalu cek distribusi class manual.
- Jangan kembali ke row random split untuk metric final.

Validasi serialisasi wajib:

```python
loaded = keras.models.load_model("training/artifacts/bisakerja_matcher_v1.keras")
sample_batch = next(iter(test_ds.take(1)))[0]
pred_original = model.predict(sample_batch, verbose=0)
pred_loaded = loaded.predict(sample_batch, verbose=0)

for key in pred_original:
    np.testing.assert_allclose(pred_original[key], pred_loaded[key], rtol=1e-5, atol=1e-6)
```

## 16. TensorBoard Integration

Log path wajib berada di repo project:

```text
training/logs/
  bisakerja_matcher_v1/
    events.out.tfevents...
training/checkpoints/
training/reports/
training/artifacts/
```

Command monitoring:

```bash
tensorboard --logdir training/logs
```

Yang wajib dimonitor:

- Train vs validation loss.
- `fit_score_mae` dan `val_fit_score_mae`.
- `fit_class_accuracy` dan `val_fit_class_accuracy`.
- AUC recommendation.
- Learning rate.
- Histogram bobot hanya untuk debug; default `histogram_freq=0`.
- Overfitting gap.
- Custom scalar `target_gate_passed`.

Logging tambahan:

- Simpan `config.yaml`.
- Simpan `requirements.txt` atau lockfile.
- Simpan split manifest.
- Simpan git commit hash.
- Simpan model summary.
- Simpan confusion matrix.
- Simpan top-k recommendation report.

## 17. Validation Strategy

Split strategy:

- Train: 70%.
- Validation: 15%.
- Test: 15%.
- Stratify by `fit_class`.
- Group-aware split jika memungkinkan:
  - `GroupShuffleSplit` by `profile_id`.
  - Test kemampuan model pada kandidat baru.
  - Alternatif: group by `job_id` untuk lowongan baru.

Validation gates:

- Gerbang validasi Bahasa Inggris lulus.
- `val_fit_class_accuracy >= 0.85`.
- `val_fit_score_mae <= 0.02`.
- Macro F1 tidak jauh di bawah accuracy.
- Gap train-val kecil.
- Test metrics mirip validation metrics.
- Human-reviewed validation dipisahkan dari weak-label validation.
- Kesetaraan simpan-muat `.keras` lulus.

Manual QA:

- Ambil top 20 rekomendasi untuk 10 profil.
- Cek skill wajib, role, pengalaman, dan alasan gap.
- Tandai false positive berisiko: skor tinggi tetapi skill wajib hilang.

## 18. Evaluation Metrics

Classification:

- Accuracy: mudah dibaca, target >= 85%.
- Macro F1-score: wajib untuk class imbalance.
- Precision: penting agar rekomendasi high-fit tidak terlalu banyak false positive.
- Recall: penting agar job relevan tidak hilang.
- Confusion matrix: cek low/medium/high tertukar.

Regression:

- MAE: target <= 0.02 pada skala 0 sampai 1.
- RMSE: cek error besar.
- Calibration plot: cek skor 0.8 benar-benar high-fit.

Semantic:

- Cosine similarity: cek embedding profile-job.
- Pearson/Spearman correlation dengan human score.

Recommendation:

- precision@k: proporsi job relevan di top-k.
- recall@k: proporsi job relevan yang berhasil muncul di top-k.
- NDCG@k optional: ranking quality dengan gain bertingkat.
- MAP@k optional: evaluasi retrieval lebih ketat.

Contoh metric report:

```text
classification:
  accuracy: 0.87
  macro_f1: 0.84
regression:
  mae: 0.018
  rmse: 0.031
semantic:
  mean_cosine_positive: 0.72
  mean_cosine_negative: 0.38
recommendation:
  precision@5: 0.68
  recall@10: 0.74
```

## 19. Hyperparameter Tuning

Parameter utama:

- Embedding model.
- Embedding dim projection: 128, 256, 512.
- Dense units: 128, 256, 512.
- Dropout: 0.1, 0.2, 0.3.
- Learning rate: 1e-3, 3e-4, 1e-4.
- Weight decay: 0, 1e-5, 1e-4.
- Batch size: 256, 512, 1024.
- Loss weights.
- Negative sampling ratio.

Strategi:

- Mulai manual baseline 5 sampai 10 eksperimen.
- Lanjut KerasTuner atau Optuna jika baseline stabil.
- Tuning tidak boleh memakai test set.
- Pilih model berdasarkan validation + human-reviewed validation.

Experiment naming:

```text
bisakerja_matcher_v{major}_{embedding_model}_{date}_{seed}
```

Contoh:

```text
bisakerja_matcher_v1_e5base_20260515_seed42
```

## 20. Experiment Tracking

Minimal artifact per run:

```text
training/
  configs/
    bisakerja_matcher_v1.yaml
  logs/
    bisakerja_matcher_v1/
  checkpoints/
    bisakerja_matcher_best.keras
  artifacts/
    bisakerja_matcher_v1.keras
  reports/
    metrics_v1.json
    classification_report_v1.json
    recommendation_report_v1.json
    model_summary_v1.txt
  runs/
    run_manifest_v1.json
```

Isi `run_manifest_v1.json`:

```json
{
  "run_name": "bisakerja_matcher_v1_e5base_20260515_seed42",
  "seed": 42,
  "dataset_version": "profile_job_pairs_v1",
  "embedding_model": "intfloat/e5-base-v2",
  "model_format": ".keras",
  "git_commit": "fill-at-runtime",
  "target_accuracy": 0.85,
  "target_mae": 0.02
}
```

TensorBoard cukup untuk MVP. MLflow atau Weights & Biases boleh ditambahkan nanti jika butuh registry dan comparison dashboard.

## 21. Reproducibility Strategy

Wajib:

- Fixed seed Python, NumPy, TensorFlow.
- Simpan split manifest.
- Simpan `language_report.json`.
- Simpan config training.
- Simpan versi dataset.
- Simpan versi embedding model.
- Simpan embedding prompt strategy: `encode_query`/`encode_document` atau prefix manual.
- Simpan dependency lock.
- Simpan git commit.
- Simpan preprocessing code version.

Contoh kode seed:

```python
import random
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
```

Catatan:

- Determinism bisa memperlambat training.
- GPU tertentu tetap bisa punya variasi kecil.
- Target metric harus dilaporkan dengan run seed.

## 22. Training Optimization

Mixed precision:

- Aktifkan di GPU modern NVIDIA Tensor Core.
- Gunakan untuk mempercepat training dan mengurangi memory.
- Output regression terakhir sebaiknya float32 jika perlu stabil.

Contoh kode:

```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
```

Checkpointing:

- Save best model berdasarkan `val_fit_score_mae`.
- Simpan juga checkpoint berdasarkan `val_fit_class_accuracy` jika classification penting.
- Pakai `.keras`, bukan H5 legacy.

Early stopping:

- Monitor `val_fit_score_mae`.
- Patience 8 sampai 12.
- `restore_best_weights=True`.

Learning rate scheduling:

- Baseline: `ReduceLROnPlateau`.
- Advanced: cosine decay + warmup.
- AdamW disarankan untuk regularization.

Regularization:

- Dropout 0.1 sampai 0.3.
- Weight decay 1e-5 sampai 1e-4.
- LayerNorm untuk stabilitas dense tower.
- Label smoothing untuk classification jika label noisy.

Data optimization:

- Precompute embeddings.
- Cache embedding arrays.
- Store processed data as Parquet.
- Use `tf.data.AUTOTUNE`.
- Copy large datasets from slow mounted storage to local SSD before repeated reads.

## 23. Recommendation Architecture

Gunakan dua tahap:

1. Retrieval
   - Ambil kandidat job top 100 sampai 500 dari embedding similarity.
   - Bisa memakai Sentence Transformers + FAISS/ScaNN nanti.
   - TensorFlow Recommenders cocok untuk two-tower retrieval.

2. Ranking
   - Model Keras Bisakerja memberi final `recommendation_score`.
   - Input: profile-job embedding + structured feature.
   - Output top-k job untuk user.

Untuk MVP:

- Precompute job embeddings.
- Saat user punya CV/profile, hitung profile embedding.
- Ambil top 100 cosine similarity.
- Re-rank dengan `.keras` matching model.

Untuk production:

- Tambahkan ANN index.
- Tambahkan feedback loop interaction labels.
- Tambahkan daily/weekly retraining.

## 24. Explainability Hooks

Walau model deep learning, produk tetap butuh alasan:

- Skill matched: intersection profile skills dan job required skills.
- Skill gap: job required skills yang tidak ada di profile.
- Experience match: level kandidat vs job.
- Semantic reason: top phrases dari requirement yang paling dekat dengan profile.
- Preference match: lokasi, work type, salary.

Jangan mengklaim layer neural sebagai alasan langsung tanpa metode interpretability. Untuk MVP, explanation sebaiknya rule-based dari feature input yang dipakai model.

## 25. Folder Training Project Ideal

Struktur folder rekomendasi:

```text
bisakerja/
  bisakerja-training-optimization-flow.md
  notebooks/
    bisakerja_model_final.ipynb
  datasets/
    raw/
    interim/
    processed/
    labels/
    metadata/
  training/
    configs/
    data/
    logs/
    checkpoints/
    artifacts/
    reports/
    runs/
  src/
    bisakerja_ml/
      data/
        load.py
        validate.py
        preprocess.py
        pair_generation.py
      features/
        embeddings.py
        skill_features.py
        structured_features.py
      models/
        layers.py
        losses.py
        callbacks.py
        matcher.py
      training/
        train.py
        evaluate.py
        tune.py
      serving/
        predict.py
        rank.py
  tests/
    test_preprocess.py
    test_features.py
    test_model_serialization.py
```

Catatan untuk saat ini: user meminta dokumen dulu saja. Struktur ini adalah rekomendasi tahap berikutnya, bukan instruksi untuk langsung membuat semua folder.

## 26. Hardware Recommendation

Minimum eksperimen:

- CPU 4 core, RAM 16 GB.
- Cocok untuk preprocessing, weak label, model dense kecil dengan precomputed embeddings.

Rekomendasi lokal:

- RAM 32 GB.
- NVIDIA GPU 8 GB VRAM.
- Batch embedding dan dense model lebih cepat.

Rekomendasi pelatihan:

- NVIDIA GPU 16 GB VRAM atau lebih.
- Mixed precision aktif.
- SSD untuk cache embeddings.

Cloud option:

- Google Colab T4/A100.
- Kaggle GPU.
- GCP Vertex AI / AWS SageMaker jika pipeline sudah matang.

## 27. Training Bottlenecks

Potensi bottleneck:

- Pair generation terlalu besar karena profile x job bisa eksplosif.
- Transformer embedding generation lambat.
- Job description panjang menyebabkan truncation kehilangan requirement penting.
- Label weak supervision terlalu noisy.
- Class imbalance high-fit terlalu sedikit.
- Data leakage dari role/required skills.
- Saving TensorBoard histogram terlalu besar.
- Fine-tuning full transformer boros VRAM.

Mitigasi:

- Candidate generation dulu, jangan full cross join.
- Precompute embeddings dan cache.
- Extract requirement summary/skills sebelum embedding.
- Hard negative mining bertahap.
- Group-aware split.
- Batasi `histogram_freq=1` hanya untuk eksperimen penting; gunakan 0 untuk run cepat.
- Mulai dari frozen embeddings + Keras dense head sebelum full fine-tuning.

## 28. Production Readiness Checklist

Data:

- Schema valid.
- Missing values ditangani.
- Split manifest tersimpan.
- Label source jelas.
- Human-reviewed validation ada.

Model:

- `.keras` bisa save-load.
- Custom layer registered serializable.
- Test inference shape lulus.
- Target accuracy dan MAE lulus di validation dan test.

Training:

- TensorBoard logs tersimpan.
- Checkpoint best model tersimpan.
- Metrics report tersimpan.
- Config dan seed tersimpan.

Evaluation:

- Accuracy >= 85%.
- MAE <= 0.02.
- Macro F1 sehat.
- Precision@k dan recall@k dihitung.
- False positive high-fit direview manual.

Deployment handoff:

- Model artifact.
- Preprocessing artifact.
- Embedding model name/version.
- Feature schema.
- Inference contract.
- Model card singkat.

## 29. Fase Pengembangan yang Disarankan

Phase 1: Baseline scoring

- Rule-based fit score.
- Sentence Transformer embeddings.
- Cosine similarity.
- Manual evaluation sample.

Phase 2: Dense matching model

- Precomputed embeddings.
- Functional API.
- Custom `CosineInteractionLayer`.
- Multi-task output.
- TensorBoard + checkpoint.

Phase 3: Better labels

- Human-reviewed labels.
- Hard negative mining.
- Group-aware validation.

Phase 4: Recommendation

- Retrieval top-k.
- Ranking model.
- precision@k/recall@k.

Phase 5: Production training

- Config-driven script.
- Reproducible artifacts.
- Model registry.
- Scheduled retraining.

## 30. Referensi Resmi

Dokumentasi dicek ulang 2026-05-18.

TensorFlow and Keras:

- Keras 3 ModelCheckpoint: https://keras.io/api/callbacks/model_checkpoint/
- Keras 3 BackupAndRestore: https://keras.io/api/callbacks/backup_and_restore/
- Keras 3 mixed precision: https://keras.io/api/mixed_precision/
- Keras Functional API: https://www.tensorflow.org/guide/keras/functional_api
- Keras saving and `.keras` serialization: https://www.tensorflow.org/guide/keras/serialization_and_saving
- Keras training and evaluation: https://www.tensorflow.org/guide/keras/training_with_built_in_methods
- Keras custom layers: https://www.tensorflow.org/tutorials/customization/custom_layers
- TensorFlow mixed precision: https://www.tensorflow.org/guide/keras/mixed_precision
- `tf.keras.callbacks.TensorBoard`: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
- TensorBoard scalars with Keras: https://www.tensorflow.org/tensorboard/scalars_and_keras
- Keras callbacks API: https://keras.io/api/callbacks/
- Keras custom callbacks guide: https://keras.io/guides/writing_your_own_callbacks/

Hugging Face and Sentence Transformers:

- Sentence Transformers `encode`, `encode_query`, `encode_document`: https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
- Transformers padding and truncation: https://huggingface.co/docs/transformers/main/pad_truncation
- Transformers AutoTokenizer and model loading: https://huggingface.co/docs/transformers
- Transformers cache setup: https://huggingface.co/docs/transformers/installation#cache-setup
- Sentence Transformers semantic textual similarity: https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
- Sentence Transformers semantic search: https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html

Recommendation systems:

- TensorFlow Recommenders overview: https://www.tensorflow.org/recommenders
- TensorFlow Recommenders retrieval tutorial: https://www.tensorflow.org/recommenders/examples/basic_retrieval

Data, metrics, reproducibility:

- Scikit-learn `train_test_split`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- Pandas CSV IO: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
- NumPy random generators: https://numpy.org/doc/stable/reference/random/index.html

FastAPI dan Pydantic:

- FastAPI response model: https://fastapi.tiangolo.com/tutorial/response-model/
- FastAPI file upload: https://fastapi.tiangolo.com/tutorial/request-files/
- FastAPI OpenAPI docs: https://fastapi.tiangolo.com/how-to/extending-openapi/
- Pydantic model config: https://docs.pydantic.dev/latest/api/config/
- Pydantic fields: https://docs.pydantic.dev/latest/concepts/fields/

## 31. Final Recommendation

Untuk Bisakerja tahap sekarang, gunakan jalur paling stabil:

1. Export dataset Bahasa Inggris terbaru, lalu jalankan gerbang validasi Bahasa Inggris.
2. Hitung lebih dulu embedding Bahasa Inggris untuk profil dan job.
3. Buat weak labels berbasis skill, semantic similarity, role, experience, dan preference.
4. Train Keras Functional API two-tower interaction model dengan Custom `CosineInteractionLayer`.
5. Pakai multi-task outputs: `fit_score`, `fit_class`, `recommendation_score`.
6. Monitor dengan TensorBoard, checkpoint `.keras`, early stopping, ReduceLROnPlateau, dan target gate callback.
7. Evaluasi dengan group-aware split, accuracy, macro F1, MAE, cosine similarity, precision@k, recall@k.
8. Validasi save-load parity `.keras`.
9. Tambahkan human-reviewed validation sebelum mengklaim target accuracy >= 85% dan MAE <= 0.02.

Strategi ini modern, modular, scalable, dan paling aman untuk menuju production karena model utama tetap Keras-native dan embedding pipeline bisa ditingkatkan tanpa mengubah kontrak scoring.
