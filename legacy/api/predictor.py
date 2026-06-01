"""
ModelPredictor — inti logika inferensi.
Load model .keras + embeddings, lalu jalankan:
  - job_fit()
  - cv_analyzer()
  - job_recommendation()
"""

import re, json, io, logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List

import numpy as np

logger = logging.getLogger("bisakerja-predictor")

# Artifact paths stay explicit so prototype inference can run from different working directories.
BASE_DIR        = Path(__file__).parent.parent
ROOT_DIR        = BASE_DIR.parent
MODEL_PATH      = BASE_DIR / "models" / "model_jobfit_v1.keras"
JOB_EMB_PATH    = BASE_DIR / "cache"  / "all_job_embeddings.npy"
JOB_INDEX_PATH  = BASE_DIR / "artifacts" / "job_index.json"

MODEL_NAME    = "bisakerja_jobfit_v1"
MODEL_VERSION = "v1"

EXP_LEVEL_MAP = {
    "Fresher": 0, "Entry Level": 1, "Junior": 2,
    "Mid Level": 3, "Senior": 4, "Lead": 5,
}
JOB_EXP_MAP = {
    "ENTRY_LEVEL": 1, "JUNIOR": 2, "MID_LEVEL": 3,
    "SENIOR": 4, "LEAD": 5, "MANAGER": 6,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_skills(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _jaccard(a: str, b: str) -> float:
    sa = {x.lower() for x in _parse_skills(a)}
    sb = {x.lower() for x in _parse_skills(b)}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _skill_match_lists(profile_skills: str, job_skills: str):
    ps = {x.lower(): x for x in _parse_skills(profile_skills)}
    js = {x.lower(): x for x in _parse_skills(job_skills)}
    matched = [js[k] for k in ps if k in js]
    missing = [js[k] for k in js if k not in ps]
    return matched, missing


def _exp_score(profile_exp: float, job_exp: float) -> float:
    diff = job_exp - profile_exp
    if diff <= 0:   return 1.0
    elif diff == 1: return 0.70
    elif diff == 2: return 0.40
    else:           return 0.10


def _readiness(score: int) -> str:
    if score >= 75:   return "READY"
    elif score >= 55: return "READY_WITH_MINOR_GAPS"
    elif score >= 35: return "NEEDS_PREPARATION"
    else:             return "NOT_RECOMMENDED_YET"


def _decision(score: int) -> str:
    if score >= 75:   return "APPLY_NOW"
    elif score >= 45: return "IMPROVE_FIRST"
    else:             return "SAVE_FOR_LATER"


def _gap_priority(rank: int) -> str:
    if rank == 0:   return "HIGH"
    elif rank <= 2: return "MEDIUM"
    else:           return "LOW"


def _extract_text_from_cv(cv_bytes: bytes, filename: str) -> str:
    """Parse CV dari PDF atau DOCX, kembalikan plain text."""
    fname = filename.lower()
    try:
        if fname.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(io.BytesIO(cv_bytes)) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        elif fname.endswith(".docx"):
            from docx import Document
            doc = Document(io.BytesIO(cv_bytes))
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            # Fallback: decode as utf-8
            return cv_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"CV parse warning: {e}")
        return ""


def _extract_skills_from_cv(cv_text: str, job_skills: str) -> List[str]:
    """
    Cek apakah skill dari job requirements muncul di teks CV.
    Sederhana: keyword match. Bisa diganti NLP lebih lanjut.
    """
    cv_lower = cv_text.lower()
    job_skill_list = _parse_skills(job_skills)
    found = [s for s in job_skill_list if s.lower() in cv_lower]
    return found


def _ats_issues(cv_text: str) -> List[str]:
    """Rule-based ATS check."""
    issues = []
    if len(cv_text) < 300:
        issues.append("CV terlalu pendek atau tidak terbaca dengan baik oleh parser")
    if not re.search(r'\b(experience|work|project|skill)\b', cv_text, re.I):
        issues.append("Tidak ditemukan section Experience, Work, Project, atau Skill")
    if not re.search(r'\d{4}', cv_text):
        issues.append("Tidak ada tahun pengalaman kerja yang terdeteksi")
    if len(cv_text) > 8000:
        issues.append("CV terlalu panjang — ATS biasanya hanya memproses 2 halaman pertama")
    if not re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', cv_text):
        issues.append("Email tidak ditemukan di CV")
    return issues if issues else ["Tidak ada masalah ATS yang terdeteksi"]


def _quantification_suggestions(cv_text: str) -> List[str]:
    """Cek apakah ada angka/metrik dalam achievement."""
    suggestions = []
    if not re.search(r'\d+\s*(%|users?|customers?|projects?|services?|apps?)', cv_text, re.I):
        suggestions.append("Tambahkan angka/metrik konkret pada pencapaian, contoh: 'Meningkatkan performa API sebesar 40%'")
    if not re.search(r'(led|managed|built|developed|reduced|increased|improved)', cv_text, re.I):
        suggestions.append("Gunakan kata kerja aksi yang kuat: Led, Built, Reduced, Improved")
    if not re.search(r'\d+\s*(years?|months?|tahun|bulan)', cv_text, re.I):
        suggestions.append("Cantumkan durasi pengalaman kerja secara eksplisit")
    return suggestions if suggestions else ["Pencapaian sudah cukup terkuantifikasi"]


class ModelPredictor:
    def __init__(self):
        self.model          = None
        self.sentence_model = None
        self.job_embeddings = None
        self.job_index      = None
        self.is_loaded      = False

    def load(self):
        """Load inference artifacts once during FastAPI startup."""
        try:
            # Load the Keras artifact with custom objects from the packaged training module.
            if MODEL_PATH.exists():
                from tensorflow import keras

                import sys
                for import_root in (ROOT_DIR, BASE_DIR):
                    root_text = str(import_root)
                    if root_text not in sys.path:
                        sys.path.insert(0, root_text)
                try:
                    from training.models.model_def import custom_objects
                    self.model = keras.models.load_model(
                        str(MODEL_PATH),
                        custom_objects=custom_objects(),
                        compile=False,
                    )
                    logger.info(f"Keras model loaded: {MODEL_PATH}")
                except Exception as e:
                    logger.warning(f"Could not load Keras model: {e}. Will use rule-based only.")
                    self.model = None
            else:
                logger.warning(f"Model not found at {MODEL_PATH} — using rule-based fallback")

            # Load Sentence Transformer
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer loaded")

            # Load job embeddings
            if JOB_EMB_PATH.exists():
                self.job_embeddings = np.load(str(JOB_EMB_PATH))
                logger.info(f"Job embeddings loaded: {self.job_embeddings.shape}")
            else:
                logger.warning(f"Job embeddings not found at {JOB_EMB_PATH}")

            # Load job index
            if JOB_INDEX_PATH.exists():
                with open(JOB_INDEX_PATH) as f:
                    self.job_index = json.load(f)
                logger.info(f"Job index loaded: {len(self.job_index)} jobs")

            self.is_loaded = True

        except Exception as e:
            logger.error(f"Load failed: {e}", exc_info=True)
            self.is_loaded = False

    def _predict_fit_score(self, profile_text: str, job_text: str) -> float:
        """
        Gunakan model Keras jika tersedia, fallback ke rule-based jika tidak.
        Return nilai float 0–1.
        """
        if self.model is not None and self.sentence_model is not None:
            try:
                p_emb = self.sentence_model.encode([profile_text], convert_to_numpy=True)
                j_emb = self.sentence_model.encode([job_text], convert_to_numpy=True)
                score = self.model.predict(
                    {"profile_emb": p_emb, "job_emb": j_emb}, verbose=0
                )
                return float(score[0][0])
            except Exception as e:
                logger.warning(f"Model predict failed, fallback: {e}")
        return None  # signal fallback to caller

    def job_fit(self, payload) -> dict:
        """
        Hitung job fit score dan kembalikan JobFitAnalysis sesuai openapi.json.
        """
        from .schemas import (
            JobFitAnalysis, Breakdown, SkillMatch, ExperienceMatch,
            PreferenceMatch, Recommendation, SkillGap, ModelInfo
        )

        # ── 1. Skill analysis ─────────────────────────────────────────────────
        matched_skills, missing_skills = _skill_match_lists(
            payload.profileSkills, payload.jobSkills
        )
        jaccard = _jaccard(payload.profileSkills, payload.jobSkills)
        skill_score = int(round(jaccard * 100))

        # ── 2. Experience analysis ────────────────────────────────────────────
        prof_exp_num = EXP_LEVEL_MAP.get(payload.profileExperience, 1)
        job_exp_num  = JOB_EXP_MAP.get(payload.jobExperienceLevel, 2)
        exp_raw      = _exp_score(prof_exp_num, job_exp_num)
        exp_score    = int(round(exp_raw * 100))

        exp_reasons = {
            1.0: f"Pengalaman kamu ({payload.profileExperience}) memenuhi requirement {payload.jobExperienceLevel}",
            0.7: f"Pengalaman kamu ({payload.profileExperience}) sedikit di bawah requirement {payload.jobExperienceLevel}",
            0.4: f"Pengalaman kamu ({payload.profileExperience}) cukup di bawah requirement {payload.jobExperienceLevel}",
            0.1: f"Gap pengalaman terlalu besar untuk posisi {payload.jobExperienceLevel}",
        }
        exp_reason = exp_reasons.get(round(exp_raw * 10) / 10, f"Gap: {payload.profileExperience} → {payload.jobExperienceLevel}")

        # ── 3. Preference analysis ────────────────────────────────────────────
        matched_prefs = []
        unmatched_prefs = []
        if payload.preferences:
            pref = payload.preferences
            if pref.workType and payload.jobWorkType:
                if pref.workType.upper() == payload.jobWorkType.upper():
                    matched_prefs.append(f"Tipe kerja: {pref.workType}")
                else:
                    unmatched_prefs.append(f"Tipe kerja: ingin {pref.workType}, job {payload.jobWorkType}")
            if pref.province and payload.jobProvince:
                if pref.province.lower() in payload.jobProvince.lower():
                    matched_prefs.append(f"Lokasi: {pref.province}")
                else:
                    unmatched_prefs.append(f"Lokasi: ingin {pref.province}, job di {payload.jobProvince}")
            if pref.salaryMin and payload.jobSalaryMin:
                if payload.jobSalaryMax and payload.jobSalaryMax >= pref.salaryMin:
                    matched_prefs.append(f"Gaji dalam range ekspektasi")
                else:
                    unmatched_prefs.append(f"Gaji di bawah ekspektasi minimum")

        pref_score = 100 if not matched_prefs and not unmatched_prefs else (
            int(round(len(matched_prefs) / max(len(matched_prefs) + len(unmatched_prefs), 1) * 100))
        )

        # ── 4. Model-based score OR weighted rule ─────────────────────────────
        profile_text = (
            f"Role: {payload.profileJobRole}. Skills: {payload.profileSkills}. "
            f"Experience: {payload.profileExperience}. Projects: {payload.profileProjects}."
        ).lower()
        job_text = (
            f"{payload.jobTitle}. Skills required: {payload.jobSkills}."
        ).lower()

        model_score_raw = self._predict_fit_score(profile_text, job_text)

        if model_score_raw is not None:
            # Blend: 60% model + 20% exp + 20% pref
            fit_score_float = (
                0.60 * model_score_raw +
                0.20 * (exp_score / 100) +
                0.20 * (pref_score / 100)
            )
        else:
            # Pure rule-based fallback
            fit_score_float = (
                0.60 * jaccard +
                0.30 * exp_raw +
                0.10 * (pref_score / 100)
            )

        fit_score = int(round(fit_score_float * 100))
        fit_score = max(0, min(100, fit_score))

        # ── 5. Skill gaps ─────────────────────────────────────────────────────
        skill_gaps = []
        for i, sk in enumerate(missing_skills[:6]):
            skill_gaps.append(SkillGap(
                skill=sk,
                priority=_gap_priority(i),
                reason=f"Skill '{sk}' dibutuhkan untuk posisi {payload.jobTitle or 'ini'} tapi tidak ada di profilmu"
            ))

        # ── 6. Recommendation texts ───────────────────────────────────────────
        readiness = _readiness(fit_score)
        decision  = _decision(fit_score)

        summary_map = {
            "APPLY_NOW": f"Profil kamu sangat cocok untuk posisi ini ({fit_score}/100). Segera lamar!",
            "IMPROVE_FIRST": f"Profil kamu cukup cocok ({fit_score}/100) tapi ada beberapa skill yang perlu diperkuat sebelum melamar.",
            "SAVE_FOR_LATER": f"Masih ada gap yang signifikan ({fit_score}/100). Simpan lowongan ini dan persiapkan diri lebih dulu.",
        }

        next_steps = []
        if missing_skills:
            next_steps.append(f"Pelajari skill yang kurang: {', '.join(missing_skills[:3])}")
        if exp_raw < 0.7:
            next_steps.append(f"Tambah pengalaman kerja — posisi ini butuh level {payload.jobExperienceLevel}")
        if unmatched_prefs:
            next_steps.append("Pertimbangkan fleksibilitas preferensi lokasi atau tipe kerja")
        if not next_steps:
            next_steps = ["Update CV dengan pencapaian terbaru", "Kirim aplikasi sekarang"]

        result = JobFitAnalysis(
            jobId=payload.jobId,
            fitScore=fit_score,
            readinessLevel=readiness,
            recommendation=Recommendation(
                decision=decision,
                summary=summary_map[decision],
                nextSteps=next_steps[:4],
                successProbability=round(fit_score_float, 2),
            ),
            breakdown=Breakdown(
                skillMatch=SkillMatch(
                    score=skill_score,
                    matchedSkills=matched_skills[:10],
                    missingSkills=missing_skills[:10],
                ),
                experienceMatch=ExperienceMatch(score=exp_score, reason=exp_reason),
                preferenceMatch=PreferenceMatch(
                    score=pref_score,
                    matchedPreferences=matched_prefs,
                    unmatchedPreferences=unmatched_prefs,
                ),
            ),
            skillGaps=skill_gaps,
            model=ModelInfo(name=MODEL_NAME, version=MODEL_VERSION),
            analyzedAt=_now_iso(),
        )
        return result.model_dump()

    def cv_analyzer(
        self,
        job_id: str,
        language: str,
        profile_skills: str,
        profile_experience: str,
        profile_job_role: str,
        cv_bytes: bytes,
        filename: str,
    ) -> dict:
        """
        Parse CV → analisis vs job → kembalikan CvAnalysis sesuai openapi.json.
        Untuk job data: kita pakai job_id untuk lookup dari job_index.
        """
        from .schemas import (
            CvAnalysis, OverallImpression, JobFitAlignment,
            AtsFriendliness, KeywordOptimization, ExperienceQuantification,
            GeneratedCv, ModelInfo
        )

        # ── Parse CV ──────────────────────────────────────────────────────────
        cv_text = _extract_text_from_cv(cv_bytes, filename)
        cv_len  = len(cv_text)

        # ── Get job data dari index ────────────────────────────────────────────
        job_data = None
        if self.job_index:
            for idx, jd in self.job_index.items():
                if jd.get("job_id") == job_id:
                    job_data = jd
                    break

        job_skills = job_data["skills"] if job_data else ""
        job_title  = job_data["title"]  if job_data else "posisi ini"

        # ── Skill analysis ────────────────────────────────────────────────────
        cv_skills_found = _extract_skills_from_cv(cv_text, job_skills)
        job_skill_list  = _parse_skills(job_skills)
        missing_in_cv   = [s for s in job_skill_list if s.lower() not in cv_text.lower()]

        # ── Scores ────────────────────────────────────────────────────────────
        skill_ratio = len(cv_skills_found) / max(len(job_skill_list), 1)
        fit_score   = int(round(skill_ratio * 100))

        overall_score = min(100, int(
            0.4 * fit_score +
            0.3 * (100 if cv_len > 500 else 50) +
            0.3 * (80 if not missing_in_cv else 50)
        ))

        ats_issues   = _ats_issues(cv_text)
        ats_score    = max(0, 100 - len([i for i in ats_issues if "tidak" in i.lower() or "terlalu" in i.lower()]) * 20)

        quant_issues = _quantification_suggestions(cv_text)
        quant_score  = max(40, 100 - len(quant_issues) * 20)

        # ── Keyword rekomendasi ────────────────────────────────────────────────
        recommended_kw = missing_in_cv[:8]

        # ── Summary tergantung language ───────────────────────────────────────
        if language == "en":
            overall_summary  = f"Your CV shows {fit_score}% skill alignment with {job_title}. {len(cv_skills_found)} out of {len(job_skill_list)} required skills were found."
            fit_summary      = f"Skills found in CV: {', '.join(cv_skills_found[:5]) or 'none detected'}. Missing: {', '.join(missing_in_cv[:3]) or 'none'}."
            kw_reason        = f"These keywords appear in the job requirements but are missing from your CV."
            actionable_items = [
                f"Add the following skills to your CV: {', '.join(missing_in_cv[:3])}",
                "Quantify your achievements with numbers and metrics",
                "Ensure your CV is in a clean, ATS-friendly format (no tables or columns)",
            ]
        else:
            overall_summary  = f"CV kamu menunjukkan {fit_score}% kecocokan skill dengan {job_title}. Ditemukan {len(cv_skills_found)} dari {len(job_skill_list)} skill yang dibutuhkan."
            fit_summary      = f"Skill yang ditemukan di CV: {', '.join(cv_skills_found[:5]) or 'tidak ada'}. Yang kurang: {', '.join(missing_in_cv[:3]) or 'tidak ada'}."
            kw_reason        = f"Keyword berikut ada di requirement job tapi tidak ditemukan di CV kamu."
            actionable_items = [
                f"Tambahkan skill berikut ke CV: {', '.join(missing_in_cv[:3])}",
                "Kuantifikasi pencapaian dengan angka dan metrik konkret",
                "Pastikan format CV bersih tanpa tabel kompleks agar lolos ATS",
            ]

        result = CvAnalysis(
            jobId=job_id,
            language=language,
            overallImpression=OverallImpression(score=overall_score, summary=overall_summary),
            jobFitAlignment=JobFitAlignment(
                score=fit_score,
                summary=fit_summary,
                matchedSignals=cv_skills_found[:8],
                missingSignals=missing_in_cv[:8],
            ),
            atsFriendliness=AtsFriendliness(score=ats_score, issues=ats_issues),
            keywordOptimization=KeywordOptimization(
                recommendedKeywords=recommended_kw,
                reason=kw_reason,
            ),
            experienceQuantification=ExperienceQuantification(
                score=quant_score,
                suggestions=quant_issues,
            ),
            actionableImprovements=actionable_items,
            generatedCv=GeneratedCv(
                available=False,
                note="Fitur generate CV otomatis belum tersedia di versi ini."
            ),
            model=ModelInfo(name=MODEL_NAME, version=MODEL_VERSION),
            analyzedAt=_now_iso(),
        )
        return result.model_dump()

    def job_recommendation(self, payload) -> dict:
        """
        Top-K job recommendation pakai cosine similarity embedding.
        Jika embeddings tidak tersedia, fallback ke rule-based skill match.
        """
        from .schemas import JobRecommendationResult, RecommendedJob, ModelInfo

        profile_text = (
            f"Role: {payload.profileJobRole}. Skills: {payload.profileSkills}. "
            f"Experience: {payload.profileExperience}."
        ).lower()

        top_k = min(payload.topK, 20)
        recommendations = []

        if (self.sentence_model is not None and
                self.job_embeddings is not None and
                self.job_index is not None):
            # Semantic search
            profile_emb = self.sentence_model.encode([profile_text], convert_to_numpy=True)
            # Cosine similarity
            norms = np.linalg.norm(self.job_embeddings, axis=1, keepdims=True)
            normed = self.job_embeddings / (norms + 1e-8)
            p_norm = profile_emb / (np.linalg.norm(profile_emb) + 1e-8)
            sims   = (normed @ p_norm.T).squeeze()

            top_indices = np.argsort(sims)[::-1][:top_k * 3]  # ambil lebih untuk filter
            count = 0
            for idx in top_indices:
                if count >= top_k:
                    break
                jd = self.job_index.get(str(idx))
                if not jd:
                    continue

                matched, missing = _skill_match_lists(payload.profileSkills, jd.get("skills", ""))
                fit = int(round(float(sims[idx]) * 100))
                fit = max(0, min(100, fit))

                recommendations.append(RecommendedJob(
                    jobId=jd["job_id"],
                    title=jd["title"],
                    company=jd["company"],
                    fitScore=fit,
                    matchReason=f"Kecocokan semantik tinggi ({fit}%) berdasarkan profil dan skill kamu",
                    skillMatch=matched[:5],
                    skillGap=missing[:5],
                ))
                count += 1
        else:
            # Fallback: rule-based dari job_index
            if self.job_index:
                scored = []
                for _, jd in self.job_index.items():
                    j_score = _jaccard(payload.profileSkills, jd.get("skills", ""))
                    scored.append((j_score, jd))
                scored.sort(key=lambda x: x[0], reverse=True)

                for j_score, jd in scored[:top_k]:
                    matched, missing = _skill_match_lists(payload.profileSkills, jd.get("skills", ""))
                    fit = int(round(j_score * 100))
                    recommendations.append(RecommendedJob(
                        jobId=jd["job_id"],
                        title=jd["title"],
                        company=jd["company"],
                        fitScore=fit,
                        matchReason=f"Kecocokan skill {fit}% dengan posisi {jd['title']}",
                        skillMatch=matched[:5],
                        skillGap=missing[:5],
                    ))

        result = JobRecommendationResult(
            recommendations=recommendations,
            totalFound=len(recommendations),
            model=ModelInfo(name=MODEL_NAME, version=MODEL_VERSION),
            generatedAt=_now_iso(),
        )
        return result.model_dump()
