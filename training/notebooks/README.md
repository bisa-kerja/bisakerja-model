# Notebook Phase Index

These notebooks define the model-training phases. They are documentation-first. Later production-track notebooks may include executable cells after each step documents intent, inputs, outputs, and verification.

Each notebook uses Markdown cells to describe:

- objective
- required inputs
- step-by-step actions
- expected outputs
- verification checks
- acceptance criteria

Add executable code only after the Markdown intent for that step is clear.

## Production notebook hygiene

Active production notebooks must be saved without error outputs. Unexecuted code cells in older planning notebooks must either be executed or intentionally retired with durable report evidence. Phase 13 executable cells are retired; the frozen evidence remains in `../../reports/phase_13_*.json`.

Run the hygiene and label evidence gate before any production-ready claim:

```bash
python scripts/verify_phase_27_3_27_4_release_evidence.py --write
python scripts/verify_phase_27_5_27_6_validation_expansion.py --write
```
