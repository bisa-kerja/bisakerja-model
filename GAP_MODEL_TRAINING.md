# GAP Model Training Production Ready

Dokumen ini menilai gap production-ready dari sisi **training model saja** untuk Bisakerja Model API, dengan referensi kontrak API di `references/docs/generated/openapi.json`.

## 1. Scope Training Model

Training model tidak harus menghasilkan semua field response API secara langsung.

### Output yang menjadi tanggung jawab model/training

Untuk flow CV Analyzer sesuai kebutuhan API, output utama dari model/training difokuskan ke:

1. `jobFitAlignment`
   - skor kecocokan CV/profil terhadap target role atau job context.
   - ringkasan alignment berbasis sinyal skill, role, experience, dan requirement.

2. `atsFriendliness`
   - skor keterbacaan/kelayakan ATS.
   - ringkasan masalah format/struktur CV yang terdeteksi.

3. `overallImpression`
   - impresi global CV terhadap target role.
   - summary pendek, product-ready, bukan raw model log.

### Output yang tidak wajib dari training model langsung

Field berikut boleh dihasilkan oleh API wrapper, bukan core training model:

1. `topActionables`
   - dibuat oleh AI wrapper di API memakai OpenAI SDK.
   - input wrapper: output model (`jobFitAlignment`, `atsFriendliness`, `overallImpression`) + metadata role/job.

2. `sectionReviews`
   - dibuat oleh AI wrapper di API memakai OpenAI SDK.
   - wrapper menyusun review section CV, action points, dan alasan pentingnya.

3. `jobRecommendations`
   - kandidat job berasal dari database backend.
   - backend mengambil kandidat job sesuai user/role/filter.
   - model/training memberi ranking atau match score terhadap kandidat tersebut.
   - API mengembalikan job detail dari database backend, bukan dari artifact model statis.

## 2. Expected Production Flow

```text
User upload / refer CV
        ↓
Backend validasi auth, file, ownership, request payload
        ↓
Backend load CV/job/user context dari DB/storage
        ↓
Model API inference core
        ├─ jobFitAlignment
        ├─ atsFriendliness
        └─ overallImpression
        ↓
API AI wrapper dengan OpenAI SDK
        ├─ topActionables
        └─ sectionReviews
        ↓
Backend DB candidate retrieval
        ↓
Model ranking/scoring candidate jobs
        ↓
Backend map jobRecommendations dari DB + model score
        ↓
Response sesuai OpenAPI
```

## 3. Current Training State

Berdasarkan artifact repo saat ini:

- Notebook: `bisakerja_model_training_FINAL.ipynb`
- Model artifact: `models/model_jobfit_v1.keras`
- Legacy embedding model: `all-MiniLM-L6-v2`
- Target v2 training embedding model for English-only focus: `intfloat/e5-base-v2`
- Job embeddings: `cache/all_job_embeddings.npy`
- Job index: `artifacts/job_index.json`
- Training pairs: `artifacts/pairs.parquet`
- Model card: `reports/model_card.json`

Model card saat ini:

```json
{
  "training_pairs": 25500,
  "val_pairs": 4500,
  "metrics": {
    "mae": 0.0844,
    "accuracy": 0.802,
    "r2": -0.0038
  },
  "gate_passed": true
}
```

Dataset ringkas:

- jobs: `2073`
- profiles: `69929`
- generated pairs: `30000`
- fit score range: `0.03` sampai `0.5545`
- tidak ada training sample dengan label high-fit `> 0.6`

## 4. Major Production Gaps

### GAP-01 — Label training masih weak-label rule based

Current label dibuat dari:

```text
fit_score = 0.70 * skill_jaccard + 0.30 * experience_score
```

Masalah:

- model belajar meniru heuristic sederhana, bukan real hiring/job-fit outcome.
- tidak ada ground truth dari recruiter, application success, interview, atau user feedback.
- model output sulit diklaim valid secara business.

Impact:

- `jobFitAlignment.score` rawan tidak akurat.
- `overallImpression` bisa terlalu optimistis/terlalu generik jika hanya dari weak signal.

Production requirement:

- definisikan label schema v2.
- tambahkan human-labeled sample minimal untuk validation set.
- pakai weak label hanya sebagai bootstrap, bukan satu-satunya truth.

### GAP-02 — Distribusi label tidak sehat

Current pair distribution sempit:

- mean sekitar `0.206`
- max `0.5545`
- sample high fit tidak tersedia.

Masalah:

- model tidak belajar membedakan kandidat kuat.
- output skor cenderung rendah/flat.
- threshold readiness sulit dikalibrasi.

Impact:

- `jobFitAlignment.score` tidak reliable untuk range 70–100.
- ranking `jobRecommendations` dapat buruk karena model belum pernah belajar positive pair kuat.

Production requirement:

- generate balanced pairs:
  - high fit positive pairs.
  - medium fit pairs.
  - hard negatives.
  - random negatives.
- pastikan setiap split punya distribusi score lengkap `0–1`.

### GAP-03 — Baseline belum dikalahkan

Metric current:

```text
MAE      : 0.0844
Accuracy : 0.802 @ tolerance ±0.15
R²       : -0.0038
```

Masalah:

- R² negatif berarti model tidak lebih baik dari prediksi rata-rata/median.
- baseline median bisa mencapai MAE dan accuracy mirip.
- gate `mae <= 0.10` dan `accuracy >= 0.75` terlalu lemah.

Impact:

- model belum punya bukti predictive power.
- `gate_passed: true` misleading untuk production.

Production requirement:

- report baseline wajib:
  - constant mean.
  - constant median.
  - Jaccard-only.
  - cosine-only.
- gate baru:
  - R² > 0.15 untuk weak-label validation awal.
  - MAE improve minimal 15–25% dari baseline.
  - ranking metric NDCG/MAP untuk recommendation.

### GAP-04 — Experience feature mapping rusak

Dataset profile memakai value:

- `Fresher`
- `1-2 years`
- `3-5 years`
- `5+ years`

Training mapping saat ini fokus ke:

- `Fresher`
- `Entry Level`
- `Junior`
- `Mid Level`
- `Senior`
- `Lead`
- `Manager`

Masalah:

- `1-2 years`, `3-5 years`, `5+ years` kemungkinan fallback ke default.
- signal experience tidak akurat.

Impact:

- label weak-fit salah.
- `jobFitAlignment` dan `jobRecommendations` bias.

Production requirement:

- normalisasi experience sebelum pair generation.
- mapping eksplisit:
  - `Fresher -> 0`
  - `1-2 years -> 1/2`
  - `3-5 years -> 3/4`
  - `5+ years -> 5`
- simpan mapping di versioned config artifact.

### GAP-05 — Training objective belum sesuai target API CV Analyzer

Model saat ini fokus ke job fit score berbasis profile/job embeddings.

OpenAPI CV Analyzer butuh:

- `jobFitAlignment`
- `atsFriendliness`
- `overallImpression`

Masalah:

- `atsFriendliness` current masih rule-based runtime, bukan hasil training/evaluation.
- `overallImpression` current berupa template/rule, belum trained/evaluated.
- CV text extraction dan section quality belum masuk training dataset.

Impact:

- produk terlihat AI-ready, tapi core training belum mendukung semua output utama.

Production requirement:

- pisahkan model tasks:
  - job-fit alignment scorer.
  - ATS quality scorer/classifier.
  - summary/impression generator or wrapper-grounded summarizer.
- buat eval set CV nyata/sintetis terkontrol dengan label ATS dan impression.

### GAP-06 — Recommendation training belum mengikuti backend candidate flow

OpenAPI/backend flow ideal:

- backend ambil candidate jobs dari DB.
- model hanya rank/scoring candidate ids.
- backend return job detail dari DB.

Current artifact:

- `job_index.json` statis dari training dataset.
- `all_job_embeddings.npy` statis.
- recommendation runtime bisa rank semua job di artifact.

Masalah:

- tidak menjamin kandidat sama dengan DB backend terbaru.
- risk stale job, inactive job, wrong visibility, duplicate source.
- model bisa return job yang tidak boleh muncul.

Production requirement:

- training tetap bisa produce scoring/ranking model.
- inference input harus menerima `jobCandidates` dari backend.
- output model hanya:
  - `jobId`
  - `matchScore`
  - `matchLevel`
  - `reasons`
  - `matchedSkills`
  - `missingSkills`
  - `nextSteps`
- backend hydrate detail dari DB.

### GAP-07 — No calibration report

Masalah:

- skor 80 belum terbukti berarti kandidat strong.
- `successProbability` belum terkalibrasi.
- threshold readiness belum berbasis data.

Impact:

- UX bisa misleading.
- product decision berbasis score rawan salah.

Production requirement:

- calibration curve.
- score bucket evaluation:
  - 0–20
  - 21–40
  - 41–60
  - 61–80
  - 81–100
- define score semantics per bucket.

### GAP-08 — Data language quality belum cukup jelas

Language report:

```json
{
  "EN": 1426,
  "UNKNOWN": 602,
  "MIXED": 23,
  "ID": 22
}
```

Masalah:

- EN ratio lolos gate, tapi UNKNOWN tinggi.
- Produk Indonesia (`language: id/en`) butuh output ID/EN stabil.
- training text mostly English; Indonesian CV/job text belum representatif.

Impact:

- `overallImpression` dan wrapper prompt bisa bagus, tapi model alignment bisa bias ke English skills.

Production requirement:

- language-specific eval.
- normalize skill aliases ID/EN.
- add Indonesian CV/job samples.

### GAP-09 — Training reproducibility belum production-grade

Masalah:

- training utama masih notebook.
- custom layer belum dipaketkan sebagai module deployable.
- config, dataset version, code version belum lengkap di model card.

Impact:

- sulit retrain deterministic.
- model artifact bisa gagal load di API.
- audit model version sulit.

Production requirement:

- pindahkan training ke script/package:
  - `training/data.py`
  - `training/features.py`
  - `training/model_def.py`
  - `training/train.py`
  - `training/evaluate.py`
- model card wajib berisi:
  - git commit.
  - dataset hash.
  - label version.
  - split seed.
  - feature config.
  - baseline metrics.

### GAP-10 — CV parser/evaluator belum jadi bagian training validation

Current CV Analyzer runtime parse PDF/DOCX dan rule ATS.

Masalah:

- parse failure tidak dievaluasi pada dataset CV.
- ATS score tidak diuji terhadap variasi format CV.
- model tidak tahu apakah text extraction valid.

Impact:

- `atsFriendliness` bisa salah jika parser gagal.
- `overallImpression` bisa buruk karena input kosong/partial.

Production requirement:

- buat CV parsing benchmark:
  - normal PDF.
  - scanned PDF.
  - multi-column PDF.
  - DOCX.
  - table-heavy CV.
- metric:
  - extraction coverage.
  - section detection accuracy.
  - ATS issue precision/recall.

### GAP-11 — Current notebooks belum memenuhi `REQUIREMENT.md` TensorFlow delivery

Konteks terbaru:

- Phase 12-24 sudah memperbaiki banyak gap produksi: snapshot, feature builder, pair generation v2, E5 runtime, human validation, ATS benchmark, reranking fixtures, calibration, contract validation, dan final gate evidence.
- Namun selected model terbaru masih diekspor sebagai `joblib`/scikit-learn scorer, bukan TensorFlow `.keras` atau `SavedModel`.
- `REQUIREMENT.md` mewajibkan TensorFlow Functional API atau Model Subclassing, custom component, custom training/evaluation loop dengan `tf.GradientTape`, TensorBoard logs, export `.keras`/`SavedModel`, inference code, REST API, dan GenAI secondary feature.

Legacy context:

- `legacy/bisakerja_model_training_FINAL.ipynb` lebih dekat dengan requirement karena memakai TensorFlow, custom component, TensorBoard, `.keras` export, inference helper, FastAPI/GenAI context.
- Legacy tetap belum cukup production-ready karena masih memakai `model.fit()` sebagai training utama, belum ada `tf.GradientTape`, memakai weak-label lama, embedding `all-MiniLM-L6-v2`, distribusi high-fit buruk, MAE lama tidak memenuhi target `0.02` pada skala `0-1`, dan API/runtime masih prototype.

Masalah:

- Current production-track notebooks kuat secara audit, tetapi tidak memenuhi format dan training-loop requirement.
- Legacy notebook memenuhi sebagian format, tetapi gagal pada kualitas label, metric, dan custom training loop.
- Jika langsung dipakai untuk Model API, artifact `joblib` current tidak sesuai requirement TensorFlow deployment dan legacy `.keras` tidak sesuai evidence produksi terbaru.

Impact:

- Repository bisa terlihat `production-ready` dari sisi reports, tetapi gagal penilaian requirement akademik/teknis.
- Model API berisiko memakai artifact yang tidak sesuai kontrak deployment target.
- Training evidence terpecah di banyak notebook sehingga sulit dinilai sebagai satu workflow final.

Production requirement:

- Buat satu notebook training final, misalnya `training/notebooks/phase_25_tensorflow_training_delivery.ipynb`, yang mengonsolidasikan bukti Phase 12-24 dan mengganti selected scorer menjadi TensorFlow model.
- Notebook final wajib memakai TensorFlow Functional API atau Model Subclassing, minimal satu custom layer/loss/callback, custom training/evaluation loop dengan `tf.GradientTape`, TensorBoard logs, dan export `.keras` atau `SavedModel`.
- Notebook final wajib tetap memakai kontrak produksi terbaru: `pairs_v2`, E5 `intfloat/e5-base-v2`, normalized embeddings, anti-leakage splits, human validation, ATS benchmark, calibration, model card, artifact manifest, dan OpenAPI handoff fixtures.
- Notebook final hanya untuk training/export. REST API, direct DB access, auth, persistence, backend hydration, dan GenAI wrapper runtime tetap dipisah ke service/API code.
- Production-ready status harus gagal jika export dilakukan dari dirty worktree, notebook menyimpan error output, TensorFlow artifact tidak bisa reload, TensorBoard logs hilang, atau strict baseline/score-band gate masih membutuhkan prototype-only trade-off.

## 5. Target Training Outputs for API Contract

### 5.1 CV Analyzer core model output

Minimal model response before wrapper:

```json
{
  "schemaVersion": "cv-analysis-core-v1",
  "jobFitAlignment": {
    "score": 78,
    "summarySignals": ["REST API", "PostgreSQL", "Backend project"],
    "missingSignals": ["Deployment", "Testing"]
  },
  "atsFriendliness": {
    "score": 84,
    "detectedIssues": ["Skill section kurang terstruktur"]
  },
  "overallImpression": {
    "score": 80,
    "summary": "CV menunjukkan fondasi backend kuat dengan gap pada deployment dan metrik impact."
  },
  "model": {
    "name": "bisakerja-cv-core",
    "version": "v1"
  }
}
```

API wrapper lalu transform ke OpenAPI shape:

```text
core model output
  + OpenAI SDK wrapper
  + backend candidate jobs
  + backend DB hydration
  → CvAnalysis.analysisResult
```

### 5.2 Job recommendations model output

Model tidak perlu menyimpan detail job final.

Expected model ranking output:

```json
{
  "recommendations": [
    {
      "jobId": "backend-db-job-id",
      "matchScore": 82,
      "matchLevel": "strong",
      "matchedSkills": ["TypeScript", "PostgreSQL"],
      "missingSkills": ["Docker"],
      "rankingSignals": ["role_match", "skill_overlap", "experience_match"]
    }
  ],
  "model": {
    "name": "bisakerja-job-ranker",
    "version": "v1"
  }
}
```

Backend/API wrapper dapat membuat `reason` dan `nextStep` memakai OpenAI SDK atau deterministic template.

## 6. Production Readiness Checklist — Training Only

### Data

- [ ] Dataset versioned dengan hash.
- [ ] Skill alias dictionary tersedia.
- [ ] Experience mapping fix.
- [ ] CV evaluation dataset tersedia.
- [ ] Indonesian + English eval split tersedia.
- [ ] Candidate job freshness tidak bergantung ke `job_index.json` statis.

### Label

- [ ] Weak label v2 documented.
- [ ] Human-labeled validation set tersedia.
- [ ] High/medium/low fit balanced.
- [ ] ATS labels tersedia.
- [ ] Section quality labels tersedia atau didelegasikan penuh ke wrapper.

### Model

- [ ] Model loadable tanpa notebook.
- [ ] Custom TensorFlow component serializable via `get_config()` and registered custom objects.
- [ ] Single final training notebook reproducible from clean kernel; optional package extraction can follow later.
- [ ] Model version jelas.
- [ ] Inference input schema match backend candidate flow.

### Evaluation

- [ ] Baseline comparison wajib.
- [ ] R² positif dan meaningful.
- [ ] Ranking metric untuk recommendation.
- [ ] Calibration report tersedia.
- [ ] CV parser benchmark tersedia.
- [ ] Error analysis per role, skill group, language, experience level.

### Export

- [ ] Model card lengkap.
- [ ] Artifact manifest lengkap.
- [ ] Dataset hash masuk model card.
- [ ] Label version masuk model card.
- [ ] Evaluation report masuk `reports/`.

## 7. Suggested Gates Before Production

Minimal gate untuk model v2:

```text
Job-fit / jobFitAlignment:
- MAE improve >= 20% vs best baseline
- R² >= 0.15 on weak-label validation
- Spearman >= 0.35 for pair ranking
- human validation agreement >= 0.65 weighted kappa or equivalent

ATS Friendliness:
- issue detection precision >= 0.75
- issue detection recall >= 0.65
- score bucket agreement >= 0.70

Job Recommendations:
- NDCG@10 >= baseline + 15%
- no unknown jobId returned
- no duplicate jobId returned
- all returned jobId must be from backend candidate set

Robustness:
- CV parse empty-text rate < 5% on supported PDF/DOCX benchmark
- ID and EN eval reported separately
```

## 8. Priority Roadmap

### P0 — Must fix before integration confidence

1. Fix experience mapping.
2. Add baseline metrics to evaluation.
3. Build balanced pair generation.
4. Implement serializable TensorFlow custom component (`CosineInteractionLayer`, custom loss, or callback) in the final training notebook.
5. Create one clean-kernel reproducible training notebook that exports `.keras` or `SavedModel`.
6. Align recommendation training with backend candidate ranking flow.

### P1 — Required for production-readiness claim

1. Human-labeled validation set.
2. CV ATS evaluation set.
3. Calibration report.
4. Language-specific evaluation.
5. Model card v2 with dataset hash and label version.

### P2 — Quality improvement

1. Learning-to-rank objective for recommendations.
2. Multi-task CV model for alignment + ATS.
3. Feedback loop from user applications/outcomes.
4. Periodic retraining pipeline.

## 9. Final Assessment

Current model training is acceptable for **prototype/demo**.

Current model training is not yet production-ready because:

- model has not beaten simple baseline.
- target label is weak and narrow.
- high-fit samples are missing.
- training objective does not fully match API output boundary.
- recommendation still depends on static job artifact instead of backend candidate flow.
- ATS and overall impression are not trained/evaluated as production tasks.

Recommended direction:

- keep core model focused on `jobFitAlignment`, `atsFriendliness`, and `overallImpression`.
- use OpenAI SDK wrapper in API for `topActionables` and `sectionReviews`.
- use backend DB candidate jobs for `jobRecommendations`, with model only ranking/scoring candidate IDs.
