FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/.cache/sentence-transformers \
    && chown -R user:user /home/user/.cache

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MODEL_API_ENV=staging \
    MODEL_API_SERVICE_NAME=bisakerja-model-api \
    MODEL_API_ALLOW_UNAUTHENTICATED_LOCAL=false \
    MODEL_API_MAX_RECOMMENDATIONS=5 \
    MODEL_API_TIMEOUT_MS=30000 \
    MODEL_API_WARMUP_ON_STARTUP=true \
    MODEL_API_WARMUP_REQUIRED=true \
    MODEL_API_ARTIFACT_ROOT=artifacts/phase_46_calibration_model_card_manifest_handoff_refresh \
    MODEL_API_EXPECTED_EMBEDDING_MODEL=intfloat/multilingual-e5-small \
    MODEL_API_MAX_PDF_BYTES=5000000 \
    MODEL_API_MAX_PDF_PAGES=10 \
    MODEL_API_ENABLE_GENAI_WRAPPER=false \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence-transformers

WORKDIR $HOME/app

COPY --chown=user requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.9.1" \
    && python -m pip install -r requirements.txt \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')" \
    && python -c "from pathlib import Path; import shutil, site; roots=site.getsitepackages()+[site.getusersitepackages()]; targets=('torch/include','torch/share','torch/test'); [shutil.rmtree(Path(root)/target, ignore_errors=True) for root in roots for target in targets if Path(root).exists()]; [path.unlink() for root in roots if Path(root).exists() for pattern in ('**/*.a','**/*.pyc') for path in Path(root).glob(pattern) if path.is_file()]"

COPY --chown=user model_api ./model_api
COPY --chown=user artifacts/phase_25_tensorflow_training_delivery ./artifacts/phase_25_tensorflow_training_delivery
COPY --chown=user artifacts/phase_46_calibration_model_card_manifest_handoff_refresh ./artifacts/phase_46_calibration_model_card_manifest_handoff_refresh

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=10 \
    CMD python -c "import json, urllib.request; data=json.loads(urllib.request.urlopen('http://127.0.0.1:7860/ready', timeout=5).read()); raise SystemExit(0 if data.get('ready') is True else 1)" || exit 1

CMD ["uvicorn", "model_api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
