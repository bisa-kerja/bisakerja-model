# Training Notebooks

Versioned notebook-first workflow for Bisakerja model training and release evidence.

## Documentation-First Rule

Every step must start with English Markdown before code. Required sections:

- Purpose
- Required input
- Action
- Expected output
- Verification

Code cells may follow only after the step intent and acceptance criteria are clear.

## Notebook Hygiene

Active production notebooks must be saved without error outputs. Old planning code must either be executed successfully or intentionally retired with durable report evidence.

Phase 13 executable cells are retired. Frozen evidence remains in:

```text
../../reports/phase_13_*.json
```

## Current Production Notebook

```text
phase_25_tensorflow_training_delivery.ipynb
```

Use kernel:

```text
Bisakerja Model TF 3.13
```

Create that kernel from Python `3.13.11`, not Python `3.14`. Use the same Python version for live Model API smoke tests.

## Verification Gates

Run hygiene and label evidence gates before any production-ready claim:

```bash
python scripts/verify_phase_27_3_27_4_release_evidence.py --write
python scripts/verify_phase_27_5_27_6_validation_expansion.py --write
```

## Rules

- Keep notebooks in numeric order.
- Keep generated evidence under `../../artifacts/` and `../../reports/`.
- Do not store secrets, tokens, DB URLs, raw CV text, or unrelated PII in notebook outputs.
- Do not use Python `3.14` for TensorFlow production-track notebooks unless compatible TensorFlow wheels are confirmed.
