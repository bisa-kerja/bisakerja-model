"""
Pydantic schemas — HARUS sesuai openapi.json contract.
JobFitAnalysis  → openapi.json #/components/schemas/JobFitAnalysis
CvAnalysis      → openapi.json #/components/schemas/CvAnalysis
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class UserPreferences(BaseModel):
    workType: Optional[str] = None
    employmentType: Optional[str] = None
    province: Optional[str] = None
    city: Optional[str] = None
    salaryMin: Optional[int] = None
    salaryMax: Optional[int] = None


class JobFitRequest(BaseModel):
    jobId: str
    # Profil user (Express isi dari DB)
    profileSkills: str = ""
    profileExperience: str = "Fresher"
    profileJobRole: str = ""
    profileProjects: str = ""
    # Job data (Express isi dari DB)
    jobTitle: str = ""
    jobSkills: str = ""
    jobExperienceLevel: str = "ENTRY_LEVEL"
    jobWorkType: str = ""
    jobEmploymentType: str = ""
    jobProvince: str = ""
    jobCity: str = ""
    jobSalaryMin: Optional[int] = None
    jobSalaryMax: Optional[int] = None
    # Preferences
    preferences: Optional[UserPreferences] = None
    topK: int = 10
    language: str = "id"


# ── JobFitAnalysis output ─────────────────────────────────────────────────────
class SkillMatch(BaseModel):
    score: int = Field(ge=0, le=100)
    matchedSkills: List[str]
    missingSkills: List[str]

class ExperienceMatch(BaseModel):
    score: int = Field(ge=0, le=100)
    reason: str

class PreferenceMatch(BaseModel):
    score: int = Field(ge=0, le=100)
    matchedPreferences: List[str]
    unmatchedPreferences: List[str]

class Breakdown(BaseModel):
    skillMatch: SkillMatch
    experienceMatch: ExperienceMatch
    preferenceMatch: PreferenceMatch

class Recommendation(BaseModel):
    decision: Literal["APPLY_NOW", "IMPROVE_FIRST", "SAVE_FOR_LATER"]
    summary: str
    nextSteps: List[str]
    successProbability: float = Field(ge=0.0, le=1.0)

class SkillGap(BaseModel):
    skill: str
    priority: Literal["HIGH", "MEDIUM", "LOW"]
    reason: str

class ModelInfo(BaseModel):
    name: str
    version: str

class JobFitAnalysis(BaseModel):
    jobId: str
    fitScore: int = Field(ge=0, le=100)
    readinessLevel: Literal["READY","READY_WITH_MINOR_GAPS","NEEDS_PREPARATION","NOT_RECOMMENDED_YET"]
    recommendation: Recommendation
    breakdown: Breakdown
    skillGaps: List[SkillGap]
    model: ModelInfo
    analyzedAt: str


# ── CvAnalysis output ─────────────────────────────────────────────────────────
class OverallImpression(BaseModel):
    score: int = Field(ge=0, le=100)
    summary: str

class JobFitAlignment(BaseModel):
    score: int = Field(ge=0, le=100)
    summary: str
    matchedSignals: List[str]
    missingSignals: List[str]

class AtsFriendliness(BaseModel):
    score: int = Field(ge=0, le=100)
    issues: List[str]

class KeywordOptimization(BaseModel):
    recommendedKeywords: List[str]
    reason: str

class ExperienceQuantification(BaseModel):
    score: int = Field(ge=0, le=100)
    suggestions: List[str]

class GeneratedCv(BaseModel):
    available: bool
    note: str

class CvAnalysis(BaseModel):
    jobId: str
    language: str
    overallImpression: OverallImpression
    jobFitAlignment: JobFitAlignment
    atsFriendliness: AtsFriendliness
    keywordOptimization: KeywordOptimization
    experienceQuantification: ExperienceQuantification
    actionableImprovements: List[str]
    generatedCv: GeneratedCv
    model: ModelInfo
    analyzedAt: str


# ── Job Recommendation output ─────────────────────────────────────────────────
class RecommendedJob(BaseModel):
    jobId: str
    title: str
    company: str
    fitScore: int = Field(ge=0, le=100)
    matchReason: str
    skillMatch: List[str]
    skillGap: List[str]

class JobRecommendationResult(BaseModel):
    recommendations: List[RecommendedJob]
    totalFound: int
    model: ModelInfo
    generatedAt: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
