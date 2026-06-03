# AI CV Analyzer Staging Runtime

Use this runbook before demo/staging traffic reaches Backend `/api/v1/ai/cv-analyzer`.

## Runtime policy

- Model API must expose `/ready.ready=true` before traffic.
- Staging and production default `MODEL_API_WARMUP_REQUIRED=true`.
- Set `MODEL_API_WARMUP_ON_STARTUP=true` only when startup can spend the E5/TensorFlow cold-start cost.
- If startup warmup is disabled, run the warmup command manually before routing traffic.
- `MODEL_API_TIMEOUT_MS` must be greater than measured cold-start latency, or warmup is mandatory before traffic.

Suggested budgets:

| Path                                         |   Budget | Evidence                             |
| -------------------------------------------- | -------: | ------------------------------------ |
| Cold-start `/internal/model/cv-analysis`     | 30000 ms | first warmup run after process start |
| Warm inference `/internal/model/cv-analysis` |  5000 ms | second run after E5/TensorFlow load  |

Model responses include safe observability keys for latency evidence: `parseLatencyMs`, `embeddingLatencyMs`, `tensorflowLatencyMs`, and `totalLatencyMs`.

## E5 cache and Hugging Face settings

Set `SENTENCE_TRANSFORMERS_HOME` to persistent storage so `intfloat/e5-base-v2` weights are not fetched on every deploy.

Examples:

```bash
# macOS/local
export SENTENCE_TRANSFORMERS_HOME="$PWD/.cache/sentence-transformers"

# Linux/container staging
export SENTENCE_TRANSFORMERS_HOME="/var/cache/bisakerja/sentence-transformers"
```

Optional `HF_TOKEN` can be set in staging/container environments to reduce Hugging Face rate-limit failures during image build or first model download. Do not log token values. Mount cache persistence read/write during build or init, then read-only for serving when possible.

## Warmup command

Start Model API, then run:

```bash
python scripts/warmup_ai_cv_analyzer_runtime.py \
  --model-api-url "${MODEL_API_URL:-http://127.0.0.1:8000}" \
  --token "${MODEL_API_SERVICE_TOKEN}" \
  --latency-budget-ms 30000 \
  --output reports/ai_cv_analyzer_warmup_staging.json
```

The command uses a sanitized PDF fixture. It checks `/ready`, posts `POST /internal/model/cv-analysis`, checks `/ready` again, records latency breakdown, and asserts no private field leakage such as `storageKey` or `rawCv`.

Repeat command once after cold-start. First run records cold-start latency. Second run records warm inference latency.

## Backend public staging smoke

After Model API warmup passes, run a Backend-to-Model fixture through public Backend route:

```bash
curl -X POST "${BACKEND_API_URL}/api/v1/ai/cv-analyzer" \
  -H "authorization: Bearer ${USER_ACCESS_TOKEN}" \
  -F "cvFile=@artifacts/smoke/sanitized-cv.pdf;type=application/pdf" \
  -F "compareSource=JOB_SEARCH" \
  -F "language=en"
```

Assert:

- response shape matches public `CvAnalysis` contract.
- latency stays within staging budget.
- persistence behavior matches request policy.
- no private field leakage: no raw CV text, uploaded bytes, storage keys, service tokens, DB URLs, prompts, or auth headers.
- `generatedCv.available=false` is valid when deterministic prose fallback is used.

## Rollback checklist

1. Disable broader traffic to Backend `/api/v1/ai/cv-analyzer`.
2. Keep deterministic prose fallback enabled; disable any provider-generated prose if needed.
3. Rotate `MODEL_API_SERVICE_TOKEN` if token exposure or routing confusion is suspected.
4. Run artifact path rollback: set `MODEL_API_ARTIFACT_ROOT` / `MODEL_API_MODEL_PATH` to previous verified artifact path.
5. Restart Model API and rerun warmup.
6. Confirm `/ready.ready=true` before traffic resumes.
