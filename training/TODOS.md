# TODO Stabilisasi Training

Tujuan: menstabilkan workflow training yang sudah ada agar bisa dirun ulang, diverifikasi, dan diserahterimakan dengan bukti yang jelas.

Ruang lingkup: TODO ini fokus pada `training/`, artifact training di `artifacts/`, dan bukti training di `reports/`. Backend, auth, persistence, format response publik, dan hidrasi detail job tetap di luar scope training.

Kondisi awal saat ini:

- Notebook utama production-track: `training/notebooks/phase_25_tensorflow_training_delivery.ipynb`
- Runtime target: Python `3.13.x`, TensorFlow `2.21.0`, Keras `3.14.1`
- Folder artifact utama: `artifacts/phase_25_tensorflow_training_delivery/`
- Report utama: `reports/phase_25_tensorflow_training_delivery.json`
- Status Phase 25 saat ini dari report: `prototype-only`
- Blocker production yang terlihat dari report Phase 27:
  - git state masih dirty sehingga klaim clean release belum boleh dibuat
  - final report Phase 25 belum `production-ready`
  - kandidat TensorFlow Phase 25 masih gagal preservation check terhadap Phase 18 (`validation score_band_agreement` dan `test mae_0_100`)
  - bukti human validation sudah frozen dan evaluation-only, tetapi coverage produksi belum cukup: 120 unique item < 600, score band 40/item < 150, serta slice `language` dan `experience_band` belum tersedia
  - release script masih `blocked` sampai dirty state dibersihkan atau dirilis dengan dokumentasi sengaja yang diterima gate

## Definisi Selesai

Training dianggap stabil jika semua poin ini terpenuhi:

- [x] Notebook Phase 25 berhasil dirun ulang dari clean kernel memakai runtime TensorFlow Python `3.13.x`.
- [x] Phase 25 mengekspor model, manifest, model card, bukti kalibrasi, handoff fixtures, dan TensorBoard release evidence.
- [ ] Final report Phase 25 berstatus `production-ready`, bukan hanya `staging-ready`/`prototype-only`.
- [x] Notebook hygiene gate lulus tanpa error output atau production cell yang belum dieksekusi.
- [x] Bukti human validation dan label governance sudah frozen dan dipisahkan dari weak-label training support.
- [x] Hash artifact manifest cocok dengan file export.
- [ ] Script verifikasi release lulus dari git state yang clean atau terdokumentasi dengan sengaja.
- [x] Fixture handoff Model API tersedia dan cocok dengan contract, tanpa training mengambil tanggung jawab Backend.

Catatan verifikasi saat ini:

- Notebook hygiene PASS: 27 active notebook, 0 error output, 0 unexecuted production code cell.
- Label governance PASS untuk freeze/separation: frozen human labels ada di `artifacts/manual_validation/phase_16_human_labels_frozen.csv`, manual labels bukan model input, weak labels hanya bootstrap/training support.
- Label governance masih FAIL untuk klaim production score: minimum release validation belum terpenuhi.
- Manifest/export PASS: Phase 25 final gate mencatat `artifact_manifest_hashes_match_current_files` PASS untuk 25 artifact.
- Model API handoff PASS: fixture positif hanya model-core output, negative fixture menolak wrapper/backend-owned fields, candidate membership, language, dan score bounds.

## Step 1. Kunci Scope dan Baseline

- [x] Pastikan output yang dimiliki training hanya:
  - `jobFitAlignment`
  - `atsFriendliness`
  - `overallImpression`
  - skor candidate reranking untuk job ID yang disediakan Backend
- [x] Pastikan output yang dimiliki wrapper tetap di luar training:
  - `topActionables`
  - `sectionReviews`
  - hidrasi detail job publik
  - auth, persistence, validasi request, dan orkestrasi GenAI
- [x] Baca bukti saat ini sebelum mengubah apa pun:
  - `training/README.md`
  - `training/notebooks/README.md`
  - `reports/phase_25_tensorflow_training_delivery.json`
  - `reports/phase_27_1_27_2_release_gate.json`
  - `reports/phase_27_3_27_4_notebook_label_gate.json`
  - `reports/phase_27_5_27_6_validation_expansion.json`
  - `reports/phase_27_7_clean_kernel_production_export.json`
  - `reports/phase_27_8_model_card_manifest_refresh.json`
  - `reports/phase_27_10_requirement_matrix.json`
- [x] Catat git state saat ini dan daftar dirty paths sebelum rerun notebook.

Kriteria penerimaan:

- [x] Scope training sudah eksplisit.
- [x] Report yang sudah ada sudah dipahami.
- [x] File dirty atau untracked sudah jelas apakah memang disengaja atau perlu dibersihkan.

Bukti Step 1:

- `reports/training_step_1_scope_baseline.json`
- `reports/training_step_1_scope_baseline.md`
- `scripts/verify_training_step_1_scope_baseline.py`
- `tests/test_training_step_1_scope_baseline.py`

Catatan: Step 1 selesai sebagai audit scope dan baseline. Release production-ready tetap belum diklaim karena report Phase 25/27 masih mencatat blocker/warning yang harus diselesaikan di step lanjutan.

## Step 2. Stabilkan Runtime

- Bukti audit: `scripts/verify_training_step_2_runtime.py`, `tests/test_training_step_2_runtime.py`, `reports/training_step_2_runtime.json`, dan `reports/training_step_2_runtime.md`.
- Status audit terakhir: `pass`. `training/.tf-venv-3.13/Scripts/python.exe` berjalan dengan Python `3.13.13`, TensorFlow `2.21.0`, Keras `3.14.1`, pytest tersedia di venv, dan kernel `Bisakerja Model TF 3.13` terdaftar ke venv.

- [x] Gunakan hanya Python `3.13.x` untuk training.
- [x] Buat ulang atau validasi `training/.tf-venv-3.13`.
- [x] Install dependency dari `training/requirements.txt`.
- [x] Jalankan verifikasi dari venv, bukan `python` default terminal:
  - Windows: `training\.tf-venv-3.13\Scripts\python.exe -m pytest ...`
  - Notebook: kernel `Bisakerja Model TF 3.13`
- [x] Tambahkan verifier Step 2 yang menulis bukti JSON/Markdown.
- [x] Install dan daftarkan Jupyter kernel `Bisakerja Model TF 3.13`.
- [x] Verifikasi import:
  - TensorFlow
  - Keras
  - pandas
  - sentence-transformers
- [x] Pastikan versi TensorFlow adalah `2.21.0`.
- [x] Pastikan versi Keras adalah `3.14.1`.

Kriteria penerimaan:

- [x] Kernel notebook mengarah ke `training/.tf-venv-3.13`.
- [x] `pytest` tersedia di `training/.tf-venv-3.13`, bukan bergantung ke Python `3.14` default.
- [x] TensorFlow bisa diimport tanpa error.
- [x] Python `3.14` tidak dipakai untuk Phase 25.

## Step 3. Bersihkan Notebook Hygiene

- [x] Cek semua notebook aktif di `training/notebooks/`.
- [x] Pertahankan urutan phase berdasarkan angka.
- [x] Pastikan setiap step notebook produksi diawali Markdown bahasa Inggris yang berisi:
  - Purpose
  - Required input
  - Action
  - Expected output
  - Verification
- [x] Hapus atau perbaiki error output dari notebook aktif.
- [x] Pastikan Phase 13 tetap retired dengan sengaja, dengan bukti tahan lama di `reports/phase_13_*.json`.
- [x] Rerun notebook hygiene release evidence gate.

Kriteria penerimaan:

- [x] Tidak ada notebook aktif yang memiliki error output.
- [x] Tidak ada production notebook yang memiliki cell belum dieksekusi tanpa alasan.
- [x] Notebook hygiene gate lulus.

## Step 4. Validasi Ulang Data dan Contract Input

- [x] Pastikan data snapshot dan bukti contract masih tersedia:
  - `reports/phase_13_data_snapshot_contract_freezing.json`
  - `reports/phase_13_snapshot_manifests.json`
  - `reports/phase_13_model_core_schema_contracts.json`
- [x] Pastikan Backend contract snapshot atau fixture adalah input hasil generate, bukan source Backend.
- [x] Pastikan tidak ada secret, DB URL, raw CV text, token, atau PII yang tidak relevan di output notebook.
- [x] Pastikan snapshot yang stale atau digenerate ulang memiliki hash dan manifest terbaru.

Kriteria penerimaan:

- [x] Input training sudah frozen atau direfresh dengan sengaja.
- [x] Contract input bisa ditelusuri.
- [x] Batas privasi tetap dijaga.

## Step 5. Cek Ulang Embedding dan Feature Contract

- [x] Pastikan model embedding produksi adalah `intfloat/e5-base-v2`.
- [x] Pastikan prefix CV/profile adalah `query:`.
- [x] Pastikan prefix job text adalah `passage:`.
- [x] Pastikan embedding dinormalisasi.
- [x] Pastikan model pembanding tidak diperlakukan sebagai default produksi.
- [x] Cek ulang feature quality report:
  - `reports/phase_14_feature_quality_report.json`
  - `reports/phase_14_embedding_manifest.json`
  - `reports/phase_17_embedding_manifest.json`
  - `reports/phase_18_embedding_manifest.json`
  - `reports/phase_25_e5_embedding_contract.json`

Kriteria penerimaan:

- [x] Embedding contract konsisten dari feature building sampai export.
- [x] Perubahan manifest disengaja dan tercatat.

## Step 6. Kunci Label Governance

- [x] Pastikan weak label hanya dipakai sebagai bootstrap atau training support.
- [x] Pastikan klaim skor produksi memakai bukti validasi human/recruiter-reviewed yang frozen.
- [x] Review:
  - `reports/phase_16_human_validation_label_governance.json`
  - `reports/phase_16_label_manifest.json`
  - `reports/phase_16_reviewer_guidelines.md`
  - `reports/phase_18_human_label_evaluation.json`
  - `reports/phase_27_5_27_6_validation_expansion.json`
- [x] Pastikan validation slice mencakup score band, role, bahasa, seniority, dan required ATS cases.
- [x] Pastikan reviewer guidance dan label schema sudah berversi.

Kriteria penerimaan:

- [x] Label policy aman untuk produksi.
- [x] Bukti human validation dipisahkan dari data training.
- [x] Validation expansion gate lulus.

## Step 7. Cek Ulang Pair Generation dan Split

- [x] Validasi `artifacts/pairs_v2.parquet`.
- [x] Pastikan pair type mencakup:
  - high-fit positives
  - medium-fit pairs
  - hard negatives
  - random negatives
  - same-role different-seniority pairs
  - cross-role confusing pairs
- [x] Pastikan split isolation mencegah profile leakage.
- [x] Review:
  - `reports/phase_15_balanced_pair_generation_splits.json`
  - `reports/phase_15_leakage_report.json`
  - `reports/phase_15_pair_distribution_diagnostics.json`
- [x] Regenerate pair diagnostics hanya jika source data atau label policy berubah.

Kriteria penerimaan:

- [x] Pair distribution mencakup range skor rendah, sedang, dan tinggi.
- [x] Split leakage report lulus.
- [x] Diagnostics sudah terbaru.

## Step 8. Cek Baseline Sebelum Model Selection

- [x] Review baseline report:
  - `reports/phase_17_baseline_metrics.json`
  - `reports/phase_17_ranking_metrics.json`
  - `reports/phase_17_slice_metrics.json`
  - `reports/phase_17_model_improvement_floor.json`
  - `reports/phase_25_baseline_selection_gate.json`
- [x] Pastikan model terpilih mengalahkan simple baseline.
- [x] Pastikan metrics dievaluasi per slice penting, bukan hanya global.
- [x] Pastikan weak slice punya risk note atau remediation yang eksplisit.

Kriteria penerimaan:

- [x] Best baseline sudah diketahui.
- [x] Model selection punya justifikasi berbasis bukti.
- [x] Risiko per slice terlihat.

Catatan audit Step 3-8:

- Bukti gabungan: `reports/training_steps_3_8_audit.json` dan `reports/training_steps_3_8_audit.md`.
- Status audit akhir: `implemented-with-warnings`.
- Warning tersisa:
  - Step 5: Phase 14 masih menyimpan embedding historis `local-hash-embedding-v1`; kontrak produksi Phase 25 tetap memakai E5 dan melarang fallback.
  - Step 6: frozen human validation baru 120 item dan belum memenuhi threshold klaim skor produksi; required slice sekarang hadir tetapi beberapa bucket masih under-covered.
  - Step 8: kandidat TensorFlow Phase 25 memenuhi target MAE tetapi belum mempertahankan seluruh metrik Phase 18, sehingga final readiness tetap dibatasi.

## Step 9. Rerun Phase 25 Dari Clean Kernel

- [x] Jalankan Jupyter Lab dari environment TensorFlow Python `3.13.x`.
- [x] Buka `training/notebooks/phase_25_tensorflow_training_delivery.ipynb`.
- [x] Pilih kernel `Bisakerja Model TF 3.13`.
- [x] Restart kernel.
- [x] Run all cells.
- [x] Save notebook setelah selesai tanpa error.
- [x] Pastikan report berikut sudah refresh:
  - `reports/phase_25_tensorflow_training_delivery.json`
  - `reports/phase_25_training_evaluation_loop.json`
  - `reports/phase_25_tensorflow_artifact_export.json`
  - `reports/phase_25_calibration_model_card_export.json`
  - `reports/phase_25_human_readable_final_report.json`
- [x] Pastikan artifact berikut sudah refresh:
  - `artifacts/phase_25_tensorflow_training_delivery/`
  - `artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json`
  - `artifacts/phase_25_tensorflow_training_delivery/model_card.json`

Kriteria penerimaan:

- [x] Phase 25 selesai tanpa error notebook.
- [x] Final report minimal `staging-ready`.
- [x] Jika git clean dan release evidence terpenuhi, final report menjadi `production-ready`.

Catatan hasil rerun 2026-06-04:

- Notebook Phase 25 tersimpan dengan semua code cell dieksekusi dan 0 error output.
- Report Phase 25 refresh sampai `reports/phase_25_tensorflow_training_delivery.json` generated at `2026-06-04T04:00:19.747646+00:00`.
- Final report menjadi `production-ready` setelah release evidence committed dan final gate dijalankan dari git clean state.
- Bukti final: `reports/phase_25_tensorflow_training_delivery.json` generated at `2026-06-04T04:39:58.810410+00:00`, `git_dirty_at_export=false`, `production_selection_passed=true`, dan TensorBoard release evidence PASS.

## Step 10. Verifikasi Calibration, Model Card, dan Artifact Manifest

- [ ] Pastikan model card merujuk model terpilih dan bukti kalibrasi.
- [ ] Pastikan artifact manifest berisi hash dan byte size untuk artifact export.
- [ ] Pastikan TensorBoard release evidence disalin ke path artifact yang release-visible, bukan hanya local ignored log directory.
- [ ] Rerun verifikasi model card dan manifest refresh.

Kriteria penerimaan:

- [ ] Model card dan manifest konsisten.
- [ ] TensorBoard release evidence tercatat.
- [ ] Tidak ada artifact export yang dimutasi tanpa manifest refresh.

## Step 11. Refresh Training Release Gates

- [ ] Jalankan release gate untuk clean baseline dan TensorBoard evidence.
- [ ] Jalankan notebook hygiene dan label evidence gate.
- [ ] Jalankan validation expansion gate.
- [ ] Jalankan clean-kernel production export gate.
- [ ] Jalankan model card dan manifest refresh gate.
- [ ] Jalankan requirement matrix gate.

Kriteria penerimaan:

- [ ] Phase 27.1 dan 27.2 lulus.
- [ ] Phase 27.3 dan 27.4 lulus.
- [ ] Phase 27.5 dan 27.6 lulus.
- [ ] Phase 27.7 tidak blocked.
- [ ] Phase 27.8 complete.
- [ ] Phase 27.10 tidak blocked untuk requirement yang dimiliki training.

## Step 12. Siapkan Bukti Handoff Model API

- [ ] Pastikan training hanya mengekspor handoff fixtures.
- [ ] Pastikan candidate reranking memakai candidate ID yang diberikan Backend.
- [ ] Pastikan training tidak mengarang job atau mengisi detail job.
- [ ] Review:
  - `reports/phase_21_backend_candidate_reranking.json`
  - `reports/phase_23_model_api_contract_validation.json`
  - `reports/phase_25_model_api_handoff_fixtures.json`
  - `artifacts/phase_23_model_api_contract_validation/`
- [ ] Pastikan source Backend tetap di luar repository ini kecuali fixture yang memang sengaja diversi.

Kriteria penerimaan:

- [ ] Handoff fixtures cocok dengan model-core output contract.
- [ ] Behavior milik Backend tetap di luar training.

## Step 13. Review Stabilitas Final

- [ ] Pastikan working tree hanya berisi perubahan yang disengaja.
- [ ] Pastikan generated artifacts dan reports sengaja tracked atau sengaja ignored.
- [ ] Pastikan tidak ada cache besar, virtual environment, secret, atau raw private data yang ikut distage.
- [ ] Pastikan `training/README.md` dan `RUNNING_STEPS.md` sesuai runtime dan workflow aktual.
- [ ] Tulis catatan stabilitas final yang mencakup:
  - runtime yang dipakai
  - tanggal rerun notebook
  - folder artifact export
  - status akhir Phase 25
  - status release gate
  - limitasi yang masih tersisa

Kriteria penerimaan:

- [ ] Training bisa dirun ulang oleh orang lain dari dokumentasi yang ada.
- [ ] Klaim production readiness didukung report dan manifest.
- [ ] Risiko tersisa disebutkan dengan jelas, bukan disembunyikan.

## Urutan Eksekusi yang Disarankan

1. Step 1: Kunci scope dan baseline.
2. Step 2: Stabilkan runtime.
3. Step 3: Bersihkan notebook hygiene.
4. Step 4: Validasi ulang data dan contract input.
5. Step 5: Cek ulang embedding dan feature contract.
6. Step 6: Kunci label governance.
7. Step 7: Cek ulang pair generation dan split.
8. Step 8: Cek baseline sebelum model selection.
9. Step 9: Rerun Phase 25 dari clean kernel.
10. Step 10: Verifikasi calibration, model card, dan artifact manifest.
11. Step 11: Refresh training release gates.
12. Step 12: Siapkan bukti handoff Model API.
13. Step 13: Review stabilitas final.

## Aksi Berikutnya yang Paling Dekat

- [ ] Selesaikan atau dokumentasikan dirty git state sebelum membuat klaim `production-ready`.
- [ ] Rerun Phase 25 dari clean kernel Python `3.13.x`.
- [ ] Refresh Phase 27 release gates setelah Phase 25 direrun.
- [ ] Naikkan Phase 25 dari `staging-ready` ke `production-ready` hanya setelah clean export dan release gates lulus.
