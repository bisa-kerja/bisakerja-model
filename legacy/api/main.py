"""
Bisakerja — FastAPI Model Service

Endpoints:
  POST /ai/job-fit              → JobFitAnalysis
  POST /ai/cv-analyzer          → CvAnalysis
  POST /ai/job-recommendation   → JobRecommendation
  GET  /health                  → liveness
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn, logging

from .predictor import ModelPredictor
from .schemas import JobFitRequest, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bisakerja-model-api")

predictor: ModelPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading models...")
    predictor = ModelPredictor()
    predictor.load()
    logger.info("Models loaded OK")
    yield
    logger.info("Shutdown")


app = FastAPI(
    title="Bisakerja Model API",
    version="1.0.0",
    description="Internal FastAPI — dipanggil Express backend, bukan public",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    loaded = predictor is not None and predictor.is_loaded
    return {"status": "ok", "model_loaded": loaded}


@app.post("/ai/job-fit")
def analyze_job_fit(payload: JobFitRequest):
    """
    Express memanggil endpoint ini setelah:
    1. Autentikasi user (Bearer token)
    2. Fetch profil user + preferensi dari DB
    3. Fetch job data dari DB berdasarkan jobId
    4. Attach semua ke payload dan kirim ke sini
    """
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(503, "Model belum siap")
    try:
        return predictor.job_fit(payload)
    except Exception as e:
        logger.error(f"job-fit error: {e}", exc_info=True)
        raise HTTPException(500, detail=str(e))


@app.post("/ai/cv-analyzer")
async def analyze_cv(
    job_id: str = Form(...),
    language: str = Form("id"),
    profile_skills: str = Form(...),
    profile_experience: str = Form("Fresher"),
    profile_job_role: str = Form(""),
    cv_file: UploadFile = File(...),
):
    """
    Express kirim CV file + data profil user.
    FastAPI parse CV → extract text → analisis vs job.
    """
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(503, "Model belum siap")
    try:
        cv_bytes = await cv_file.read()
        return predictor.cv_analyzer(
            job_id=job_id,
            language=language,
            profile_skills=profile_skills,
            profile_experience=profile_experience,
            profile_job_role=profile_job_role,
            cv_bytes=cv_bytes,
            filename=cv_file.filename or "cv.pdf",
        )
    except Exception as e:
        logger.error(f"cv-analyzer error: {e}", exc_info=True)
        raise HTTPException(500, detail=str(e))


@app.post("/ai/job-recommendation")
def job_recommendation(payload: JobFitRequest):
    """Top-K job recommendation berdasarkan profil user."""
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(503, "Model belum siap")
    try:
        return predictor.job_recommendation(payload)
    except Exception as e:
        logger.error(f"job-recommendation error: {e}", exc_info=True)
        raise HTTPException(500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8001, reload=True)
