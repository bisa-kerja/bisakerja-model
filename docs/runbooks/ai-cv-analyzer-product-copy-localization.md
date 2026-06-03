# AI CV Analyzer Product Copy and Localization

## Purpose

Define safe staging/demo behavior for AI CV Analyzer public copy. Model API returns model-core evidence only. Backend renders user-visible text from sanitized model evidence, backend job metadata, and optional validated GenAI output.

## Language Policy

- `language=en`: return English public copy.
- `language=id`: return English public copy during staging/demo.
- Indonesian localization stays deferred until product-approved Indonesian templates or validated GenAI output are available.
- Public copy must not mix Indonesian and English unless a product-approved bilingual policy replaces this rule.

## Approved Deterministic Fallback

Deterministic fallback is accepted for staging/demo when GenAI is disabled or rejected by validation. It covers:

- job-fit summary
- ATS summary
- overall impression
- top actionables
- section reviews
- recommendation reasons and next steps

Template source: `artifacts/phase_39_product_copy_localization/approved_fallback_templates.json`.

## Safety Rules

Public responses must never expose:

- raw CV text
- prompt text
- tokens or service credentials
- storage keys or DB URLs
- email, phone, address, or unrelated PII
- unsupported hiring guarantees, seniority claims, salary claims, or protected-class claims

GenAI output, when supplied, must pass schema validation, score/order preservation, and safety filters before persistence or frontend response.

## Verification

```bash
python scripts/verify_phase_39_product_copy_localization.py --write
python -m unittest tests.test_phase_39_product_copy_localization
cd references && bun test --preload ./tests/preload-env.ts tests/unit/ai-cv-analyzer/ai-cv-analyzer.service.test.ts
```
