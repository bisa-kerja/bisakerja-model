# Alur Optimasi Pelatihan Model Bisakerja di Cloud

Dokumen ini adalah alur pelatihan model Bisakerja yang dioptimalkan untuk notebook cloud seperti Google Colab dan Kaggle Notebook. Fokusnya tetap pada pencocokan semantik CV/profil kandidat dengan deskripsi lowongan, skor kecocokan kerja, analisis CV, dan skor rekomendasi. Desain pipeline dibuat tahan runtime terputus, storage sementara, GPU terbatas, dan eksperimen cepat.

Sasaran:

- Platform utama: Google Colab, Kaggle Notebook.
- Teknologi: TensorFlow, Keras, Hugging Face Transformers, Sentence Transformers, Scikit-learn, Pandas, NumPy.
- Output model: `.keras`.
- Target validasi: akurasi >= 85%, MAE <= 0.02.
- Komponen custom lanjutan: layer custom `CosineInteractionLayer`, ditambah callback custom opsional `TargetGateCallback`.
- Artifact wajib: model, checkpoint, TensorBoard logs, metrics report, config, split manifest, preprocessing artifact.

Catatan: target akurasi dan MAE adalah gerbang rilis. Jika label masih hasil supervisi lemah, metrik harus dianggap baseline internal, bukan klaim final production.

Catatan bahasa dokumen:

- Penjelasan memakai Bahasa Indonesia.
- Nama endpoint, field JSON, enum, nama library, nama model, path file, tombol platform, dan kode tetap mengikuti kontrak teknis aslinya.

Pembaruan asumsi data:

- Dataset terbaru akan sepenuhnya berbahasa Inggris untuk job description, requirements, skills, profile, projects, education, experience, dan teks turunan.
- Dataset Bahasa Inggris terbaru belum diexport saat dokumen ini ditulis.
- Pelatihan penuh di cloud hanya boleh jalan setelah gerbang validasi Bahasa Inggris lulus.
- Karena target data hanya berbahasa Inggris, embedding default di cloud harus mengutamakan Bahasa Inggris; model multibahasa hanya fallback saat masih ada data campuran.

## 1. Prinsip Cloud Notebook

Cloud notebook punya batasan berbeda dari local training:

- Runtime bisa disconnect.
- GPU bisa berubah antar session.
- Storage lokal bisa hilang setelah session selesai.
- Install package bisa ulang setiap run.
- Internet bisa dibatasi.
- Unggah/unduh file manual rawan salah versi.

Strategi:

- Semua input disimpan di cloud storage: Google Drive untuk Colab, Kaggle Dataset untuk Kaggle.
- Semua output penting ditulis ke persistent path.
- Checkpoint disimpan per epoch terbaik.
- Embedding di-cache agar tidak generate ulang.
- Training config selalu versioned.
- Notebook hanya orchestrator; logic idealnya dipindah ke script modular.

## 2. Platform Decision

| Kebutuhan          | Colab                        | Kaggle                                     |
| ------------------ | ---------------------------- | ------------------------------------------ |
| Eksperimen cepat   | sangat cocok                 | cocok                                      |
| Dataset versioning | manual via Drive             | kuat via Kaggle Dataset                    |
| GPU gratis         | tersedia, tidak selalu sama  | tersedia, kuota terbatas                   |
| TPU                | tersedia di beberapa runtime | tersedia di beberapa runtime               |
| Output persisten   | Drive mount                  | `/kaggle/working` lalu Save Version        |
| Reproducibility    | perlu disiplin folder        | lebih mudah karena input dataset versioned |
| Sharing notebook   | mudah                        | sangat mudah                               |

Rekomendasi:

- Colab untuk iterasi awal dan eksplorasi.
- Kaggle untuk eksperimen yang butuh dataset versioning dan reproducible notebook output.
- Setelah stabil, pindah ke script training di repo atau managed training.

## 2A. Kesiapan Kontrak Backend dan Model API

`openapi.json` punya dua endpoint AI utama:

- `POST /api/v1/ai/job-fit`
- `POST /api/v1/ai/cv-analyzer`

Di production, backend API memanggil Model API FastAPI. Pengguna tidak perlu memanggil Model API langsung.

Alur yang disarankan:

```text
Backend API
  -> autentikasi user
  -> ambil profil, preferensi, job, bookmark, dan file CV
  -> bersihkan muatan data
  -> panggil Model API FastAPI
  -> validasi respons
  -> kembalikan format pembungkus sesuai OpenAPI backend
```

Output model untuk `job-fit` harus mendukung:

- `fitScore`: skor 0 sampai 100 dari `fit_score`.
- `readinessLevel`: kelas kesiapan dari skor dan skill gap.
- `recommendation.decision`: keputusan apply sekarang, perbaiki dulu, atau simpan.
- `recommendation.successProbability`: probabilitas terkalibrasi 0 sampai 1.
- `breakdown.skillMatch`: skor, skill cocok, dan skill hilang.
- `breakdown.experienceMatch`: skor dan alasan pengalaman.
- `breakdown.preferenceMatch`: skor dan preferensi match/tidak match.
- `skillGaps`: skill, prioritas, dan alasan.

Output model untuk `cv-analyzer` harus mendukung:

- `overallImpression`: skor dan ringkasan kualitas CV.
- `jobFitAlignment`: skor, ringkasan, sinyal match, dan sinyal hilang.
- `atsFriendliness`: skor dan masalah struktur CV.
- `keywordOptimization`: keyword yang disarankan dan alasan.
- `experienceQuantification`: skor dan saran kuantifikasi pengalaman.
- `actionableImprovements`: daftar perbaikan CV.

Catatan implementasi FastAPI:

- Gunakan model Pydantic ketat untuk permintaan internal dan respons.
- Set `extra="forbid"` agar field liar tidak lolos diam-diam.
- Pakai `UploadFile` untuk file PDF jika Model API menerima file langsung.
- Untuk fase awal, lebih aman backend mengekstrak PDF atau mengirim teks CV yang sudah bersih.
- Respons Model API harus stabil karena backend akan menganggap respons tidak valid sebagai `DOWNSTREAM_ERROR`.

Persiapan endpoint ketiga:

- Simpan job embedding dan mapping `job_id`.
- Simpan skema fitur ranking.
- Simpan pipeline ekstraksi profil/CV.
- Endpoint rekomendasi nanti tinggal memakai pencarian top-k lalu ranking dengan model yang sama.

## 2B. Ringkasan Langkah Cloud untuk Pemula

Urutan kerja paling aman:

1. Siapkan dataset Bahasa Inggris terbaru di Google Drive atau Kaggle Dataset.
2. Jalankan cell setup runtime, path, cache, dan dependency.
3. Muat CSV, validasi skema, lalu jalankan gerbang Bahasa Inggris.
4. Simpan laporan bahasa ke folder persistent.
5. Normalisasi teks, skill, pengalaman, preferensi, dan lowongan.
6. Buat pasangan profil-lowongan secara terbatas agar RAM aman.
7. Buat label lemah dan simpan ke Parquet.
8. Hitung embedding dan simpan cache di Drive atau `/kaggle/working`.
9. Buat `tf.data` dari file processed.
10. Jalankan uji cepat 2 epoch dengan data kecil.
11. Jika uji cepat lulus, jalankan pelatihan penuh dengan checkpoint dan TensorBoard.
12. Evaluasi, simpan `.keras`, laporan metrik, manifest, dan artifact preprocessing.
13. Uji model bisa dimuat ulang dan respons sesuai kontrak FastAPI.

Jangan menjalankan pelatihan penuh jika dataset belum diexport, gerbang bahasa gagal, atau cache embedding belum stabil.

## 3. Cloud Folder Layout

Struktur repo lokal:

```text
test-model-bisakerja/
  bisakerja-cloud-training-optimization-flow.md
  bisakerja-training-optimization-flow.md
  bisakerja_model_final.ipynb
  indotech_job_cleaned.csv
  techtalent_profile_cleaned.csv
  training/
    configs/
    logs/
    checkpoints/
    artifacts/
    reports/
    cache/
```

Colab layout:

```text
/content/
  bisakerja/
    repo/
    data/
    training/

/content/drive/MyDrive/bisakerja/
  datasets/
    raw/
    processed/
  training/
    logs/
    checkpoints/
    artifacts/
    reports/
    cache/
```

Kaggle layout:

```text
/kaggle/input/
  bisakerja-datasets/
    indotech_job_cleaned.csv
    techtalent_profile_cleaned.csv

/kaggle/working/
  bisakerja/
    training/
      logs/
      checkpoints/
      artifacts/
      reports/
      cache/
```

Rule:

- Colab: copy output penting ke Google Drive.
- Kaggle: tulis output ke `/kaggle/working`, lalu klik Save Version agar tersimpan.
- Jangan simpan artifact penting hanya di `/content` atau temp folder Kaggle.

## 4. Runtime Setup

Colab startup cell:

```python
from google.colab import drive
drive.mount("/content/drive")

PROJECT_DIR = "/content/bisakerja"
PERSIST_DIR = "/content/drive/MyDrive/bisakerja"
```

Kaggle startup cell:

```python
PROJECT_DIR = "/kaggle/working/bisakerja"
PERSIST_DIR = "/kaggle/working/bisakerja"
INPUT_DIR = "/kaggle/input/bisakerja-datasets"
```

GPU check:

```python
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
print(tf.config.list_physical_devices("TPU"))
```

Memory growth:

```python
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

## 5. Dependency Strategy

Install minimal:

```python
!pip install -q sentence-transformers transformers scikit-learn pandas numpy pyarrow
```

Jika TensorFlow bawaan environment sudah ada, jangan upgrade sembarang. Upgrade TensorFlow bisa memecahkan kompatibilitas CUDA di notebook.

Pin dependency untuk reproducibility:

```python
!pip freeze > {PERSIST_DIR}/training/reports/requirements_freeze.txt
```

Cache Hugging Face:

```python
import os

os.environ["HF_HOME"] = f"{PERSIST_DIR}/training/cache/huggingface"
os.environ["HF_HUB_CACHE"] = f"{PERSIST_DIR}/training/cache/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{PERSIST_DIR}/training/cache/huggingface/transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = f"{PERSIST_DIR}/training/cache/sentence_transformers"
```

Alasan:

- Model embedding tidak perlu download ulang.
- Runtime lebih cepat.
- Eksperimen lebih stabil jika internet lambat.
- `HF_HUB_CACHE`/`TRANSFORMERS_CACHE` mengikuti prioritas cache Hugging Face modern.

## 6. Ingesti Dataset

Input saat ini:

- `indotech_job_cleaned.csv`
- `techtalent_profile_cleaned.csv`

Opsi Colab:

```python
RAW_DIR = f"{PERSIST_DIR}/datasets/raw"
jobs_path = f"{RAW_DIR}/indotech_job_cleaned.csv"
profiles_path = f"{RAW_DIR}/techtalent_profile_cleaned.csv"
```

Opsi Kaggle:

```python
RAW_DIR = "/kaggle/input/bisakerja-datasets"
jobs_path = f"{RAW_DIR}/indotech_job_cleaned.csv"
profiles_path = f"{RAW_DIR}/techtalent_profile_cleaned.csv"
```

Muat data:

```python
import pandas as pd

jobs = pd.read_csv(jobs_path)
profiles = pd.read_csv(profiles_path)

assert {"job_id", "description", "requirements_concat", "skills_clean"}.issubset(jobs.columns)
assert {"ID", "Skills", "Projects", "Education", "Experience", "Job_Role"}.issubset(profiles.columns)
```

Praktik terbaik cloud:

- Simpan processed dataset sebagai Parquet.
- Simpan data split sebagai file, bukan generate ulang diam-diam.
- Jangan overwrite raw dataset.
- Jalankan gerbang validasi bahasa sebelum membuat pasangan data dan cache embedding.
- Simpan `language_report.json` di persistent path.

Gerbang validasi Bahasa Inggris:

```python
INDONESIAN_MARKERS = {
    "kami", "mencari", "pengalaman", "pendidikan", "minimal", "lulusan",
    "bersedia", "ditempatkan", "kemampuan", "tanggung jawab", "kualifikasi",
    "diutamakan", "memiliki", "menguasai", "bekerja sama",
}

def count_indonesian_markers(text):
    value = str(text).lower()
    return sum(marker in value for marker in INDONESIAN_MARKERS)

def validate_english_dataset(jobs, profiles):
    job_cols = [
        "description",
        "requirement_summary",
        "requirements_concat",
        "skills_clean",
        "skills_top_10_names",
    ]
    profile_cols = [
        "Skills",
        "Projects",
        "Education",
        "Experience",
        "Job_Role",
        "Required_Skills",
    ]
    job_markers = sum(
        jobs[col].fillna("").map(count_indonesian_markers).sum()
        for col in job_cols
        if col in jobs.columns
    )
    profile_markers = sum(
        profiles[col].fillna("").map(count_indonesian_markers).sum()
        for col in profile_cols
        if col in profiles.columns
    )
    report = {
        "passed": int(job_markers + profile_markers) == 0,
        "job_marker_count": int(job_markers),
        "profile_marker_count": int(profile_markers),
    }
    return report

language_report = validate_english_dataset(jobs, profiles)
assert language_report["passed"], language_report
```

## 7. Versioning Dataset Cloud

Manifest wajib:

```json
{
  "dataset_name": "bisakerja_semantic_pairs",
  "dataset_version": "v1",
  "jobs_file": "indotech_job_cleaned.csv",
  "profiles_file": "techtalent_profile_cleaned.csv",
  "created_at": "YYYY-MM-DD",
  "seed": 42,
  "label_strategy": "weak_supervision_v1",
  "split_strategy": "stratified_group_split"
}
```

Colab:

- Simpan manifest ke Google Drive.
- Tambahkan tanggal dan run id.
- Untuk banyak file kecil, simpan sebagai `.zip`/`.tar.gz`, copy ke VM, lalu extract lokal. Drive mount bisa lambat/error jika folder terlalu banyak item.
- Tulis checkpoint penting ke Drive, tetapi baca dataset besar berulang dari local VM path setelah copy.

Kaggle:

- Unggah data mentah/processed sebagai Kaggle Dataset.
- Pakai dataset version untuk eksperimen.
- Output notebook disimpan lewat Save Version.
- `/kaggle/input` read-only; semua artifact harus ke `/kaggle/working`.
- Jika processed data dipakai ulang, promote output menjadi Kaggle Dataset version baru.

## 8. Alur Data yang Dioptimalkan untuk Cloud

Alur:

```text
Raw CSV
  -> schema validation
  -> gerbang validasi Bahasa Inggris
  -> text normalization
  -> skill normalization
  -> candidate pair generation
  -> weak labels
  -> embedding cache
  -> structured features
  -> train/val/test parquet
  -> tf.data
  -> Keras model
  -> checkpoint + TensorBoard logs
  -> final .keras artifact
```

Kenapa embedding cache penting:

- Sentence Transformer embedding generation sering paling lambat.
- Notebook runtime bisa terputus.
- Precomputed embeddings memungkinkan resume cepat.

## 9. Strategi Label untuk Cloud

Labeling yang aman untuk cloud:

1. Buat weak label sekali.
2. Simpan pasangan berlabel ke Parquet.
3. Review sampel manual di CSV/Sheets.
4. Muat label hasil review sebagai override.

Label lemah:

```text
fit_score =
  0.40 * skill_overlap_weighted +
  0.25 * semantic_similarity +
  0.15 * role_similarity +
  0.10 * experience_match +
  0.05 * education_match +
  0.05 * preference_match
```

Kelas:

```text
high_fit: fit_score >= 0.75
medium_fit: 0.45 <= fit_score < 0.75
low_fit: fit_score < 0.45
```

Simpan:

```python
pairs.to_parquet(f"{PERSIST_DIR}/datasets/processed/profile_job_pairs_v1.parquet")
```

## 10. Strategi Model Embedding

Model yang disarankan:

Default training embedding model:

```text
intfloat/e5-base-v2
```

| Model                                     | Kegunaan                    | Biaya cloud | Catatan                              |
| ----------------------------------------- | --------------------------- | ----------- | ------------------------------------ |
| `intfloat/e5-base-v2`                     | pencarian + pencocokan      | sedang      | default training; query/passage wajib |
| `sentence-transformers/all-mpnet-base-v2` | pembanding semantik         | sedang      | STS kuat untuk baseline pembanding    |
| `sentence-transformers/all-MiniLM-L6-v2`  | baseline uji cepat          | rendah      | cepat; bukan default final            |
| `BAAI/bge-base-en-v1.5`                   | benchmark pencarian         | sedang      | kuat untuk pencarian Inggris          |
| `intfloat/e5-large-v2`                    | benchmark akhir             | tinggi      | lebih baik, lebih lambat              |
| `BAAI/bge-large-en-v1.5`                  | pencarian akhir             | tinggi      | VRAM/latensi lebih tinggi             |
| `BAAI/bge-m3`                             | fallback data campuran      | tinggi      | pakai hanya jika gerbang EN gagal     |

Rekomendasi cloud:

- Gunakan `intfloat/e5-base-v2` sebagai default Phase 17/18 training.
- Pakai `all-mpnet-base-v2` dan `all-MiniLM-L6-v2` hanya sebagai baseline pembanding.
- Benchmark model besar hanya setelah label, split, dan cache stabil.
- Pakai model multibahasa hanya untuk cek migrasi, bukan pelatihan final data Inggris.

Untuk model gaya E5:

```text
profile text prefix: query:
job text prefix: passage:
```

Untuk API pencarian Sentence Transformers:

```python
profile_embeddings = embedder.encode_query(
    profiles["profile_text"].tolist(),
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
)
job_embeddings = embedder.encode_document(
    jobs["job_text"].tolist(),
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
)
```

Jika model terpilih tidak mendukung prompt routing, gunakan `encode()` biasa dengan prefix eksplisit.

## 11. Pola Cache Embedding

Contoh kode:

```python
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

cache_dir = Path(f"{PERSIST_DIR}/training/cache/embeddings")
cache_dir.mkdir(parents=True, exist_ok=True)

profile_cache = cache_dir / "profile_embeddings_v1.npy"
job_cache = cache_dir / "job_embeddings_v1.npy"

embedder = SentenceTransformer("intfloat/e5-base-v2")

if profile_cache.exists():
    profile_embeddings = np.load(profile_cache)
else:
    profile_embeddings = embedder.encode_query(
        profiles["profile_text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    np.save(profile_cache, profile_embeddings)

if job_cache.exists():
    job_embeddings = np.load(job_cache)
else:
    job_embeddings = embedder.encode_document(
        jobs["job_text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    np.save(job_cache, job_embeddings)
```

Panduan batch:

- T4 GPU: 32 sampai 64 untuk embedding.
- P100/V100/A100: 64 sampai 256.
- CPU saja: 16 sampai 32.

## 12. Arsitektur Model untuk Cloud

Disarankan: model interaksi dua tower dengan Keras Functional API dan embedding yang sudah dihitung lebih dulu.

Alasan cocok untuk Colab/Kaggle:

- Lebih cepat daripada fine-tuning transformer penuh.
- Lower VRAM.
- Lebih mudah dilanjutkan ulang.
- Kompatibel dengan `.keras`.
- Tetap bisa berjalan di CPU jika perlu.

Arsitektur:

```text
profile_embedding -> Dense tower ----\
                                      CosineInteractionLayer -> fusion -> outputs
job_embedding     -> Dense tower ----/
structured_features ----------------/

outputs:
  fit_score
  fit_class
  recommendation_score
```

Advanced component:

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

## 13. Build Model Pseudo-Code

```python
def dense_tower(x, name, dropout=0.2):
    x = layers.Dense(512, activation="gelu", name=f"{name}_dense_1")(x)
    x = layers.LayerNormalization(name=f"{name}_ln_1")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout_1")(x)
    x = layers.Dense(256, activation="gelu", name=f"{name}_dense_2")(x)
    x = layers.LayerNormalization(name=f"{name}_ln_2")(x)
    return x

def build_model(embedding_dim, structured_dim):
    profile_in = keras.Input(shape=(embedding_dim,), name="profile_embedding")
    job_in = keras.Input(shape=(embedding_dim,), name="job_embedding")
    struct_in = keras.Input(shape=(structured_dim,), name="structured_features")

    profile_repr = dense_tower(profile_in, "profile")
    job_repr = dense_tower(job_in, "job")
    interaction = CosineInteractionLayer(name="cosine_interaction")([profile_repr, job_repr])

    x = layers.Concatenate(name="fusion")([profile_repr, job_repr, interaction, struct_in])
    x = layers.Dense(256, activation="gelu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation="gelu")(x)

    fit_score = layers.Dense(1, activation="sigmoid", dtype="float32", name="fit_score")(x)
    fit_class = layers.Dense(3, activation="softmax", dtype="float32", name="fit_class")(x)
    rec_score = layers.Dense(1, activation="sigmoid", dtype="float32", name="recommendation_score")(x)

    return keras.Model(
        inputs={
            "profile_embedding": profile_in,
            "job_embedding": job_in,
            "structured_features": struct_in,
        },
        outputs={
            "fit_score": fit_score,
            "fit_class": fit_class,
            "recommendation_score": rec_score,
        },
        name="bisakerja_cloud_matcher",
    )
```

## 14. Mixed Precision

Enable only when GPU supports it well:

```python
from tensorflow.keras import mixed_precision

if tf.config.list_physical_devices("GPU"):
    mixed_precision.set_global_policy("mixed_float16")
```

Rules:

- Keep final output layers `dtype="float32"`.
- If loss becomes NaN, disable mixed precision.
- For dense model on embeddings, mixed precision usually helps on Tensor Core GPU.

## 15. tf.data Pipeline

Cloud-optimized dataset:

```python
import tensorflow as tf

def make_dataset(df, batch_size, shuffle=False):
    x = {
        "profile_embedding": df[profile_embedding_cols].to_numpy("float32"),
        "job_embedding": df[job_embedding_cols].to_numpy("float32"),
        "structured_features": df[structured_cols].to_numpy("float32"),
    }
    y = {
        "fit_score": df["fit_score"].to_numpy("float32"),
        "fit_class": df["fit_class_id"].to_numpy("int32"),
        "recommendation_score": df["recommendation_label"].to_numpy("float32"),
    }
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(df), 10000), seed=42, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```

Batch size:

- CPU: 64 to 256.
- T4: 512 to 1024 for dense embeddings.
- P100/V100/A100: 1024 to 4096 if RAM/VRAM enough.

If OOM:

- Halve batch size.
- Disable mixed precision only if instability, not first OOM fix.
- Reduce dense units from 512/256 to 256/128.

## 16. Training Config

Use config dict:

```python
CONFIG = {
    "run_name": "bisakerja_cloud_matcher_v1",
    "seed": 42,
    "embedding_model": "intfloat/e5-base-v2",
    "embedding_prompt_strategy": "encode_query_encode_document",
    "dataset_language": "en",
    "embedding_dim": 768,
    "structured_dim": 12,
    "batch_size": 512,
    "epochs": 80,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "target_accuracy": 0.85,
    "target_mae": 0.02,
}
```

Save config:

```python
import json
from pathlib import Path

run_dir = Path(f"{PERSIST_DIR}/training/runs/{CONFIG['run_name']}")
run_dir.mkdir(parents=True, exist_ok=True)

with open(run_dir / "config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)
```

## 17. Callbacks for Cloud

Cloud callbacks:

```python
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=f"{PERSIST_DIR}/training/logs/{CONFIG['run_name']}",
        histogram_freq=0,
        write_graph=True,
        update_freq="epoch",
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=f"{PERSIST_DIR}/training/checkpoints/{CONFIG['run_name']}_best.keras",
        monitor="val_fit_score_mae",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.BackupAndRestore(
        backup_dir=f"{PERSIST_DIR}/training/checkpoints/{CONFIG['run_name']}_backup"
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
]
```

Alasan:

- `ModelCheckpoint`: best model survives disconnect.
- `BackupAndRestore`: resume interrupted training.
- `TensorBoard`: metric history stored in persistent path.
- `EarlyStopping`: avoid wasting GPU quota.
- `ReduceLROnPlateau`: stabilize convergence.

Catatan Keras:

- Whole-model checkpoints must end with `.keras` or `.h5`.
- Weight-only checkpoints must end with `.weights.h5`.
- `BackupAndRestore` directory must not be reused by another callback or run.
- Set `delete_checkpoint=False` if you want backup state preserved after successful training.

## 18. TensorBoard on Colab

```python
%load_ext tensorboard
%tensorboard --logdir {PERSIST_DIR}/training/logs
```

Monitor:

- `loss`, `val_loss`.
- `fit_score_mae`, `val_fit_score_mae`.
- `fit_class_accuracy`, `val_fit_class_accuracy`.
- `recommendation_score_auc`, `val_recommendation_score_auc`.
- Learning rate.

Use `histogram_freq=0` for normal cloud runs to reduce log size. Enable histogram only for debugging.

## 19. TensorBoard on Kaggle

Kaggle can write TensorBoard logs to `/kaggle/working`. If inline TensorBoard is not convenient, save logs as output artifact and inspect later.

Disarankan:

```python
log_dir = f"/kaggle/working/bisakerja/training/logs/{CONFIG['run_name']}"
```

After run:

- Klik Save Version.
- Download output if needed.
- Re-open logs locally with:

```bash
tensorboard --logdir training/logs
```

## 20. Compile and Train

```python
model = build_model(
    embedding_dim=CONFIG["embedding_dim"],
    structured_dim=CONFIG["structured_dim"],
)

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    ),
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

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CONFIG["epochs"],
    callbacks=callbacks,
)

model.save(f"{PERSIST_DIR}/training/artifacts/{CONFIG['run_name']}.keras")
```

## 21. Resume Strategy

If runtime disconnects:

1. Reconnect runtime.
2. Mount Drive or load Kaggle output/dataset.
3. Re-run setup cells.
4. Load processed data/cache.
5. Re-run model compile.
6. Continue with same `BackupAndRestore` path.

Manual fallback:

```python
best_path = f"{PERSIST_DIR}/training/checkpoints/{CONFIG['run_name']}_best.keras"
model = keras.models.load_model(best_path)
```

If custom layer fails to load:

- Ensure `CosineInteractionLayer` class cell ran before `load_model`.
- Ensure `@keras.saving.register_keras_serializable` exists.

## 22. Evaluation

After training:

```python
test_metrics = model.evaluate(test_ds, return_dict=True)
print(test_metrics)
```

Scikit-learn metrics:

```python
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, classification_report

pred = model.predict(test_ds)
y_score_pred = pred["fit_score"].reshape(-1)
y_class_pred = pred["fit_class"].argmax(axis=1)

mae = mean_absolute_error(y_score_true, y_score_pred)
acc = accuracy_score(y_class_true, y_class_pred)
f1 = f1_score(y_class_true, y_class_pred, average="macro")

print({"mae": mae, "accuracy": acc, "macro_f1": f1})
print(classification_report(y_class_true, y_class_pred))
```

Release gate:

```python
passed = acc >= 0.85 and mae <= 0.02
```

Jangan tandai siap production jika:

- Gerbang validasi Bahasa Inggris gagal.
- Accuracy high but macro F1 low.
- MAE low only on weak-label validation.
- Test set leaks profiles/jobs from train.
- Human-reviewed sample fails.
- `.keras` save-load parity fails.

Aturan split cloud:

- Fast smoke test may use stratified random split.
- Final cloud benchmark must use group-aware split by `profile_id`.
- Add second benchmark grouped by `job_id` to test new-job generalization.
- Save exact split IDs to `split_manifest.json`.
- Never use `profile_id + job_id` as the only group key because pair IDs are usually unique.

Kesetaraan simpan-muat:

```python
model.save(f"{PERSIST_DIR}/training/artifacts/{CONFIG['run_name']}.keras")
loaded = keras.models.load_model(f"{PERSIST_DIR}/training/artifacts/{CONFIG['run_name']}.keras")
sample_batch = next(iter(test_ds.take(1)))[0]
pred_original = model.predict(sample_batch, verbose=0)
pred_loaded = loaded.predict(sample_batch, verbose=0)

for key in pred_original:
    np.testing.assert_allclose(pred_original[key], pred_loaded[key], rtol=1e-5, atol=1e-6)
```

## 23. Recommendation Metrics

For each profile:

1. Score candidate jobs.
2. Sort descending by `recommendation_score`.
3. Compare top-k with relevant jobs.

Metrics:

```text
precision@k = relevant jobs in top-k / k
recall@k = relevant jobs in top-k / all relevant jobs
```

Cloud pseudo-code:

```python
def precision_at_k(relevance, k):
    top_k = relevance[:k]
    return sum(top_k) / k

def recall_at_k(relevance, total_relevant, k):
    if total_relevant == 0:
        return 0.0
    return sum(relevance[:k]) / total_relevant
```

Report:

```json
{
  "precision@5": 0.68,
  "precision@10": 0.61,
  "recall@5": 0.42,
  "recall@10": 0.74
}
```

## 24. Artifact Export

Export files:

```text
training/artifacts/
  bisakerja_cloud_matcher_v1.keras
training/checkpoints/
  bisakerja_cloud_matcher_v1_best.keras
training/reports/
  language_report.json
  metrics.json
  classification_report.json
  recommendation_report.json
  requirements_freeze.txt
training/runs/
  bisakerja_cloud_matcher_v1/
    config.json
    split_manifest.json
training/cache/
  embeddings/
```

Save metrics:

```python
with open(f"{PERSIST_DIR}/training/reports/{CONFIG['run_name']}_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2)
```

For Kaggle:

- Make sure artifacts are under `/kaggle/working`.
- Klik Save Version.
- Download output zip if needed.

For Colab:

- Artifacts already in Drive if `PERSIST_DIR` points to Drive.
- Optional zip:

```python
!zip -r /content/bisakerja_training_artifacts.zip {PERSIST_DIR}/training/artifacts {PERSIST_DIR}/training/reports
```

## 25. TPU Strategy

Use TPU only if:

- Model uses TensorFlow/Keras native ops.
- Batch size is large.
- Data pipeline is stable.
- You train dense model or TensorFlow-native transformer.

Avoid TPU for first iteration if:

- Sentence Transformers embedding generation uses PyTorch.
- Pipeline mixes many CPU/Python ops.
- You need quick debugging.

TPU setup pattern:

```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    model = build_model(...)
    model.compile(...)
```

Recommendation:

- Start with GPU.
- Use TPU later only for TensorFlow-native large training.

## 26. GPU Memory Optimization

If OOM:

1. Lower batch size.
2. Reduce `max_length` for transformer embedding.
3. Use precomputed embeddings.
4. Reduce dense units.
5. Use mixed precision.
6. Clear session between experiments.

```python
import gc
import tensorflow as tf

tf.keras.backend.clear_session()
gc.collect()
```

Avoid:

- Keeping huge DataFrames duplicated.
- Full profile x job cross join.
- Running multiple models in same runtime without clearing session.
- Histogram TensorBoard every epoch for large models.

## 27. Experiment Flow in Notebook

Bagian notebook yang disarankan:

```text
00 Runtime check
01 Mount storage / set paths
02 Install + import dependencies
03 Config + seed
04 Load raw data
05 Validate schema
06 Preprocess text and skills
07 Build candidate pairs
08 Generate/load embeddings cache
09 Build features and labels
10 Split train/val/test
11 Build tf.data
12 Build model
13 Train with callbacks
14 Evaluate
15 Save artifacts
16 Error analysis
17 Recommendation demo
```

Rule:

- Every expensive section must be cache-aware.
- Every output section must write to persistent path.
- Every run must have unique `run_name`.

## 28. Minimal Smoke Test

Before full training:

- Use 500 to 2000 pair rows.
- Use MiniLM embedding.
- Train 2 epochs.
- Check model save/load.
- Check TensorBoard logs.
- Check metrics JSON.

Smoke test target:

```text
model.fit runs
model.evaluate runs
model.save creates .keras
load_model works
TensorBoard log exists
metrics report exists
```

Only after smoke test pass:

- Run full pair dataset.
- Use stronger embedding.
- Increase epochs.
- Enable tuning.

## 29. Hyperparameter Tuning in Cloud

Cloud-friendly tuning:

- Tune small search space.
- Use early stopping.
- Save each trial result.
- Avoid huge grid search.

Suggested search:

```text
embedding_model: e5-base-v2 default; all-mpnet-base-v2/MiniLM comparator only
dense_units: 256/128, 512/256
dropout: 0.1, 0.2, 0.3
learning_rate: 1e-4, 3e-4, 1e-3
batch_size: 256, 512, 1024
loss_weight_fit_score: 0.4, 0.5, 0.6
```

Kaggle:

- Better for comparing versions.
- Save notebook versions per experiment.

Colab:

- Better for fast manual experiments.
- Save each run folder in Drive.

## 30. Cloud Cost and Quota Strategy

Reduce wasted GPU:

- Do preprocessing on CPU.
- Cache embeddings.
- Run smoke test first.
- Use early stopping.
- Use lower-cost embedding for debugging.
- Keep model dense first.
- Fine-tune transformer only after label quality improves.

GPU should be used for:

- Embedding generation.
- Dense model training.
- Final benchmark runs.

CPU is enough for:

- CSV load.
- Cleaning.
- Pembuatan label lemah.
- Metrics report.
- Small smoke test.

## 31. Error Analysis

Save false positives:

```text
high predicted fit, low true fit
```

Save false negatives:

```text
low predicted fit, high true fit
```

Fields to inspect:

- `profile_text`
- `job_text`
- `profile_skills`
- `job_skills`
- `skill_overlap`
- `semantic_similarity`
- `fit_score_true`
- `fit_score_pred`
- `fit_class_true`
- `fit_class_pred`

Cloud output:

```python
errors.to_csv(f"{PERSIST_DIR}/training/reports/{CONFIG['run_name']}_error_analysis.csv", index=False)
```

Use findings to update:

- Skill alias map.
- Label weights.
- Hard negative mining.
- Text construction.

## 32. Best Cloud Architecture

Rekomendasi untuk Bisakerja saat ini:

```text
Colab/Kaggle Notebook
  -> mount/load cloud data
  -> preprocess + cache embeddings
  -> train Keras Functional API dense interaction model
  -> TensorBoard persistent logs
  -> save best checkpoint .keras
  -> save final .keras + reports
  -> manual review + iteration
```

Jangan mulai dari:

- Full transformer fine-tuning inside `.keras`.
- Huge cross join all profiles x all jobs.
- No checkpoint.
- No artifact manifest.
- No human-reviewed validation.

## 33. Colab-Specific Checklist

Before run:

- Runtime type set to GPU.
- Drive mounted.
- `PERSIST_DIR` points to Drive.
- Raw data exists in Drive.
- Hugging Face cache points to Drive.

During run:

- Watch GPU allocation.
- Save checkpoint to Drive.
- TensorBoard logs to Drive.
- Avoid idle timeout.

After run:

- Confirm `.keras` exists.
- Confirm metrics JSON exists.
- Confirm TensorBoard event files exist.
- Save notebook copy.

## 34. Kaggle-Specific Checklist

Before run:

- Dataset attached under `/kaggle/input`.
- GPU enabled in Notebook settings.
- Internet enabled if downloading models.
- Output paths under `/kaggle/working`.

During run:

- Avoid writing to `/kaggle/input`; read-only.
- Cache outputs under `/kaggle/working`.
- Keep logs/artifacts compact.

After run:

- Klik Save Version.
- Check output files in notebook artifacts.
- Promote processed dataset to Kaggle Dataset if reused.

## 35. Security and Privacy

CV data can contain personal information.

Rules:

- Jangan upload CV asli ke Kaggle Dataset publik.
- Use private Kaggle dataset for sensitive data.
- Avoid logging raw CV text to TensorBoard.
- Error analysis with personal data must stay private.
- Mask name, phone, email, address before training if using real CVs.

For current dataset, profile appears synthetic/structured, but production CVs must be treated as sensitive.

## 36. Referensi Resmi

Dokumentasi dicek ulang 2026-05-18.

Colab:

- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- Colab forms, Drive, and data loading examples: https://colab.research.google.com/notebooks/io.ipynb
- Colab TensorBoard guide: https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/docs/r2/tensorboard_in_notebooks.ipynb

Kaggle:

- Kaggle Notebooks docs: https://www.kaggle.com/docs/notebooks
- Kaggle Datasets docs: https://www.kaggle.com/docs/datasets
- Kaggle GPU docs: https://www.kaggle.com/docs/efficient-gpu-usage

TensorFlow and Keras:

- Keras 3 ModelCheckpoint: https://keras.io/api/callbacks/model_checkpoint/
- Keras 3 BackupAndRestore: https://keras.io/api/callbacks/backup_and_restore/
- Keras 3 mixed precision: https://keras.io/api/mixed_precision/
- Keras Functional API: https://www.tensorflow.org/guide/keras/functional_api
- Keras save and serialize: https://www.tensorflow.org/guide/keras/serialization_and_saving
- Keras callbacks: https://keras.io/api/callbacks/
- BackupAndRestore callback: https://keras.io/api/callbacks/backup_and_restore/
- TensorBoard callback: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
- TensorBoard with Keras: https://www.tensorflow.org/tensorboard/scalars_and_keras
- Mixed precision: https://www.tensorflow.org/guide/keras/mixed_precision
- tf.data performance: https://www.tensorflow.org/guide/data_performance

Hugging Face and Sentence Transformers:

- Sentence Transformers `encode`, `encode_query`, `encode_document`: https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
- Transformers cache setup: https://huggingface.co/docs/transformers/installation#cache-setup
- Transformers padding/truncation: https://huggingface.co/docs/transformers/main/pad_truncation
- Sentence Transformers semantic similarity: https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
- Sentence Transformers semantic search: https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html

Evaluation:

- Scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Scikit-learn train/test split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

FastAPI dan Pydantic:

- FastAPI response model: https://fastapi.tiangolo.com/tutorial/response-model/
- FastAPI file upload: https://fastapi.tiangolo.com/tutorial/request-files/
- FastAPI OpenAPI docs: https://fastapi.tiangolo.com/how-to/extending-openapi/
- Pydantic model config: https://docs.pydantic.dev/latest/api/config/
- Pydantic fields: https://docs.pydantic.dev/latest/concepts/fields/

## 37. Rekomendasi Akhir Cloud

Jalur terbaik untuk Bisakerja:

1. Export dataset Bahasa Inggris terbaru.
2. Jalankan gerbang validasi Bahasa Inggris sebelum pembuatan pasangan data.
3. Use Colab for fast experiments with Drive persistence.
4. Use Kaggle for versioned dataset and reproducible notebook runs.
5. Hitung lebih dulu embedding Bahasa Inggris dan simpan cache.
6. Train Keras Functional API model on embeddings + structured features.
7. Use Custom `CosineInteractionLayer`.
8. Save TensorBoard logs, checkpoint, final `.keras`, config, metrics, language report, and split manifest.
9. Validate save-load parity, accuracy, macro F1, MAE, cosine similarity, precision@k, recall@k.
10. Treat accuracy >= 85% and MAE <= 0.02 as release gate only after leakage checks and human-reviewed validation.

This approach gives fast iteration in cloud notebook while preserving production-ready artifacts for later API deployment.
