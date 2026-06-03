# Running Steps — Bisakerja Model Training

Panduan ini menjelaskan cara menjalankan project training model Bisakerja pelan-pelan dari terminal sampai notebook bisa dipakai.

Target runtime sekarang:

- Python: `3.13.11`
- Virtual environment: `training/.tf-venv-3.13`
- Jupyter kernel: `Bisakerja Model TF 3.13`
- Notebook utama Phase 25: `training/notebooks/phase_25_tensorflow_training_delivery.ipynb`

> Jangan pakai Python `3.14` untuk Phase 25. TensorFlow `2.21.0` belum tersedia untuk runtime itu di environment ini.

---

## 1. Masuk ke root project

Buka terminal baru, lalu masuk ke folder project:

```bash
cd /Users/macbookpro/Development/bisakerja-model
```

Cek posisi:

```bash
pwd
```

Harus keluar:

```txt
/Users/macbookpro/Development/bisakerja-model
```

Cek file penting:

```bash
ls TODOS.md REQUIREMENT.md GAP_MODEL_TRAINING.md training/requirements.txt
```

Kalau ada yang `No such file`, berarti belum di root project.

---

## 2. Matikan virtual environment lama dulu

Kalau prompt terminal menunjukkan `notebooks Py`, `.venv`, `venv`, atau env lain, matikan dulu:

```bash
deactivate 2>/dev/null || true
```

Cek Python global yang sedang terlihat:

```bash
which python
python -V
```

Tidak masalah kalau masih Python `3.14` di sini, karena langkah berikutnya akan membuat venv khusus Python `3.13`.

---

## 3. Buat virtual environment TensorFlow Python 3.13

Jalankan:

```bash
PYENV_VERSION=3.13.11 pyenv exec python -m venv training/.tf-venv-3.13
```

Aktifkan venv:

```bash
source training/.tf-venv-3.13/bin/activate
```

Cek lagi:

```bash
which python
python -V
python -m pip -V
```

Expected:

```txt
.../bisakerja-model/training/.tf-venv-3.13/bin/python
Python 3.13.11
.../bisakerja-model/training/.tf-venv-3.13/lib/python3.13/...
```

Kalau masih mengarah ke `training/notebooks/venv/lib/python3.14`, ulangi dari langkah 2.

---

## 4. Install dependency project

Dengan venv `training/.tf-venv-3.13` aktif, jalankan:

```bash
python -m pip install --upgrade pip
python -m pip install -r training/requirements.txt
python -m pip install ipykernel jupyterlab
```

Kenapa harus begini:

- `training/requirements.txt` berisi dependency training, termasuk TensorFlow.
- `ipykernel` membuat venv muncul sebagai pilihan kernel notebook.
- `jupyterlab` menjalankan UI notebook dari venv yang sama.

---

## 5. Validasi TensorFlow dan dependency utama

Jalankan:

```bash
python - <<'PY'
import sys
import pandas as pd
import tensorflow as tf
import keras
from sentence_transformers import SentenceTransformer

print('python:', sys.executable)
print('python_version:', sys.version.split()[0])
print('pandas:', pd.__version__)
print('tensorflow:', tf.__version__)
print('keras:', keras.__version__)
print('sentence_transformers: import-ok')
PY
```

Expected penting:

```txt
python: .../training/.tf-venv-3.13/bin/python
python_version: 3.13.11
tensorflow: 2.21.0
```

Kalau TensorFlow gagal dengan `No matching distribution`, hampir pasti pip masih memakai Python `3.14`. Ulangi langkah 2 dan 3.

---

## 6. Register Jupyter kernel yang benar

Jalankan:

```bash
python -m ipykernel install --user --name bisakerja-model-tf313 --display-name "Bisakerja Model TF 3.13"
```

Cek kernel:

```bash
jupyter kernelspec list
```

Harus ada:

```txt
bisakerja-model-tf313
```

Kernel lama yang membingungkan sebaiknya dihapus:

```bash
jupyter kernelspec uninstall -f bisakerja-model-venv 2>/dev/null || true
```

Jangan pilih kernel `python3` default untuk Phase 25, karena biasanya mengarah ke Python global `3.14`.

---

## 7. Pastikan tidak ada Jupyter server lama berjalan

Cek server aktif:

```bash
jupyter server list
jupyter notebook list
```

Kalau output hanya:

```txt
Currently running servers:
```

berarti tidak ada server aktif.

Kalau ada server lama, lihat port dari output, misalnya `http://localhost:8888/...`, lalu stop:

```bash
jupyter server stop 8888
```

Ulangi untuk port lain yang muncul. Setelah itu cek lagi:

```bash
jupyter server list
```

---

## 8. Jalankan Jupyter Lab dari venv yang benar

Pastikan venv masih aktif:

```bash
which python
python -V
```

Lalu jalankan Jupyter Lab dari root project:

```bash
python -m jupyter lab --notebook-dir .
```

Browser akan terbuka. Kalau tidak terbuka otomatis, copy URL dari terminal.

Biarkan terminal ini tetap hidup selama memakai notebook.

---

## 9. Pilih kernel benar di notebook

Di Jupyter Lab:

1. Buka notebook.
2. Klik menu **Kernel**.
3. Pilih **Change Kernel**.
4. Pilih **Bisakerja Model TF 3.13**.
5. Klik **Restart Kernel**.

Cek dari cell notebook:

```python
import sys
import tensorflow as tf
print(sys.executable)
print(tf.__version__)
```

Expected:

```txt
.../training/.tf-venv-3.13/bin/python
2.21.0
```

Kalau bukan path itu, kernel salah. Ganti lagi ke **Bisakerja Model TF 3.13**.

---

## 10. Jalankan notebook Phase 25

Notebook utama untuk memenuhi requirement TensorFlow:

```txt
training/notebooks/phase_25_tensorflow_training_delivery.ipynb
```

Cara run aman:

1. Buka notebook Phase 25.
2. Pilih kernel **Bisakerja Model TF 3.13**.
3. Klik **Kernel → Restart Kernel**.
4. Klik **Run → Run All Cells**.
5. Tunggu sampai selesai.
6. Save notebook.

Setelah selesai, cek report Phase 25:

```bash
ls reports/phase_25_*.json
```

Cek ringkas status report:

```bash
python - <<'PY'
import json
from pathlib import Path

for p in sorted(Path('reports').glob('phase_25_*.json')):
    d = json.loads(p.read_text())
    print(p.name, 'passed=', d.get('passed'), 'status=', d.get('status'))
PY
```

Cek log TensorBoard Phase 25:

```bash
ls artifacts/tensorboard/phase_25_tensorflow_training_delivery
```

Buka TensorBoard dari venv yang punya package TensorBoard:

```bash
tensorboard --logdir artifacts/tensorboard/phase_25_tensorflow_training_delivery
```

Kalau command `tensorboard` belum tersedia, install ulang dependency project dari venv aktif:

```bash
python -m pip install -r training/requirements.txt
```

---

## 11. Urutan notebook kalau mulai dari awal

Kalau ingin menjalankan seluruh track notebook-first, jalankan berurutan:

```txt
phase_00_reproducibility_snapshot.ipynb
phase_01_data_audit_contracts.ipynb
phase_02_label_schema_baselines.ipynb
phase_03_normalization_feature_design.ipynb
phase_04_pair_generation_splits.ipynb
phase_05_baseline_evaluation.ipynb
phase_06_jobfit_training_experiments.ipynb
phase_07_ats_friendliness_scoring.ipynb
phase_08_overall_impression_signals.ipynb
phase_09_candidate_reranking.ipynb
phase_10_calibration_model_card.ipynb
phase_11_final_gate_review.ipynb
phase_12_repository_hygiene_runtime_bootstrap.ipynb
phase_13_data_snapshot_contract_freezing.ipynb
phase_14_normalization_feature_builder.ipynb
phase_15_balanced_pair_generation_splits.ipynb
phase_16_human_validation_label_governance.ipynb
phase_17_baseline_evaluation_v2.ipynb
phase_18_jobfit_training_v2.ipynb
phase_19_ats_friendliness_benchmark_scorer.ipynb
phase_19_5_jobfit_blocker_remediation.ipynb
phase_20_overall_impression_signals.ipynb
phase_21_backend_candidate_reranking.ipynb
phase_22_calibration_model_card_export.ipynb
phase_23_model_api_contract_validation.ipynb
phase_24_reproducibility_final_gate.ipynb
phase_25_tensorflow_training_delivery.ipynb
```

Untuk tugas saat ini, fokus ke Phase 25 dulu karena Phase 25 adalah notebook final untuk TensorFlow training delivery.

---

## 12. Validasi E5 embedding model

Phase 17+ dan Phase 25 memakai `intfloat/e5-base-v2`.

Jalankan sekali dari venv aktif:

```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('intfloat/e5-base-v2')
emb = model.encode(
    ['query: backend developer with python', 'passage: python backend job'],
    normalize_embeddings=True,
)
print('shape:', emb.shape)
print('norms:', np.linalg.norm(emb, axis=1).round(4).tolist())
PY
```

Expected:

```txt
shape: (2, 768)
norms: [1.0, 1.0]
```

Kalau gagal karena download/cache, jalankan ulang saat internet stabil.

---

## 13. Troubleshooting pelan-pelan

### Error: `No matching distribution found for tensorflow==2.21.0`

Penyebab umum: pip memakai Python `3.14`.

Cek:

```bash
python -V
python -m pip -V
```

Kalau ada `python3.14` atau `training/notebooks/venv/lib/python3.14`, fix:

```bash
deactivate 2>/dev/null || true
source training/.tf-venv-3.13/bin/activate
python -V
python -m pip -V
python -m pip install -r training/requirements.txt
```

### Notebook package tidak ketemu

Cek cell:

```python
import sys
print(sys.executable)
```

Kalau bukan `training/.tf-venv-3.13/bin/python`, kernel salah.

Fix: **Kernel → Change Kernel → Bisakerja Model TF 3.13**.

### Terlalu banyak kernel membingungkan

Lihat daftar kernel:

```bash
jupyter kernelspec list
```

Untuk Phase 25, yang dipakai hanya:

```txt
bisakerja-model-tf313
```

Hapus kernel lama project kalau muncul:

```bash
jupyter kernelspec uninstall -f bisakerja-model-venv
```

### Terlalu banyak server Jupyter membingungkan

Lihat server:

```bash
jupyter server list
```

Stop server by port:

```bash
jupyter server stop 8888
jupyter server stop 8889
```

Jalankan lagi satu server saja:

```bash
python -m jupyter lab --notebook-dir .
```

### Report tidak berubah setelah Run All

Cek:

- notebook sudah pakai kernel benar
- tidak ada error cell
- notebook sudah disimpan
- file `reports/phase_25_*.json` timestamp berubah

Command:

```bash
python - <<'PY'
from pathlib import Path
from datetime import datetime
for p in sorted(Path('reports').glob('phase_25_*.json')):
    print(datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds'), p)
PY
```

---

## 14. Project rules

- Training execution tetap di `.ipynb` notebooks.
- Jangan tambah `training/*.py` entrypoint untuk training execution.
- Phase 25 memakai TensorFlow Functional API / custom training flow.
- Embedding default Phase 17+: `intfloat/e5-base-v2`.
- Profile/CV prefix: `query:`.
- Job prefix: `passage:`.
- Embeddings wajib normalized.
- TF-IDF/local-hash fallback hanya boleh untuk local smoke/plumbing, bukan staging/production evidence.
- Backend/wrapper-owned outputs tetap di luar model core: `topActionables`, `sectionReviews`, hydration, auth, persistence.
