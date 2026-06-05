# Shared CV Evidence Cache Report

Generated: `2026-06-04T21:34:42.006428+00:00`

Passed: `True`

## Decision

Backend-owned hybrid fallback cache with Backend parser for Generate MVP and Model API evidence adapter for Analyzer wrapper.

## Parser Ownership Options

### Model API parse-only

Owner: Model API primary, Backend cache adapter
Quality: Highest available parser and ATS signal quality after analyzer inference.
Latency risk: Adds service hop when used only for Generate; acceptable for Analyzer because Model API call already happens.
Privacy risk: Raw CV stays internal between Backend and Model API; sanitized evidence only may reach GenAI.
Implementation risk: Medium because Generate would need extra internal endpoint or analyzer dependency.
Decision: Use for Analyzer wrapper evidence only in MVP.

### Backend parser

Owner: Backend API
Quality: Good enough for bounded section/skill evidence from text-like stored bytes; weak for scanned PDFs.
Latency risk: Lowest latency for Generate because no extra service hop.
Privacy risk: Low when raw text is not cached and contact data is redacted before provider calls.
Implementation risk: Low because Generate already reads owned CV storage.
Decision: Use as MVP Generate parser and shared cache owner.

### Latest analysis reuse

Owner: Backend API persisted sanitized snapshots
Quality: Useful fallback for ATS/actionables and section reviews when current stored bytes are weak.
Latency risk: Low read-path latency.
Privacy risk: Low if scoped to same user and same CV file.
Implementation risk: Low but stale without source/model invalidation.
Decision: Use as fallback only when same-CV latest analysis exists.

### Hybrid fallback

Owner: Backend API orchestration
Quality: Best practical MVP coverage by combining model_api, backend_parser, latest_analysis_cache, metadata_only.
Latency risk: Bounded by using source already available in each flow.
Privacy risk: Low when all provider inputs pass shared no-leak policy.
Implementation risk: Medium due to schema/invalidation tests.
Decision: Chosen MVP architecture.

## Invalidation

- `fresh_cache` fresh=`True`
- `cv_file_changed` fresh=`False`
- `parser_version_changed` fresh=`False`
- `analysis_model_changed` fresh=`False`
- `template_policy_changed` fresh=`False`
- `retention_expired` fresh=`False`

## Schema And Privacy

```json
{
  "forbiddenFields": [
    "raw CV text",
    "raw PDF content",
    "contact data",
    "prompts",
    "provider payloads",
    "storage identifiers",
    "tokens and secrets"
  ],
  "observability": [
    "parser confidence",
    "evidence source",
    "cache hit/miss/bypass",
    "wrapper fallback reason",
    "template validation failure",
    "no-leak checks"
  ],
  "retainedFields": [
    "bounded section evidence",
    "skill and requirement coverage",
    "ATS/actionable evidence",
    "parser confidence",
    "source hash",
    "timestamps and retention policy"
  ],
  "schemaVersion": "shared-cv-evidence-v1",
  "sources": [
    "model_api",
    "backend_parser",
    "latest_analysis_cache",
    "metadata_only"
  ]
}
```

## Quality, Latency, Risk

```json
{
  "latency": "No extra Model API parse-only hop for Generate in MVP.",
  "quality": "Analyzer uses Model API parser/scoring evidence; Generate uses Backend parser with same-CV latest analysis fallback.",
  "risk": "Weak scanned-PDF Generate evidence remains low confidence and must not fabricate content."
}
```
