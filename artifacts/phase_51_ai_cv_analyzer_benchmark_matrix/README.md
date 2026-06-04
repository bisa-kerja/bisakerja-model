# Phase 51 AI CV Analyzer Benchmark Matrix

Fixed benchmark fixture for real CV uploads used by AI CV Analyzer output-quality checks.

Contents:

- `benchmark_matrix.json` — required CV files, SHA-256 hashes, parser evidence snapshots, target-role probes, role-specific evidence expectations, score ranges, and English-only output policy.

The fixture stores parser evidence and regression expectations only. It must not store raw CV text or backend-owned response fields.
