# Panduan Integrasi Express → FastAPI

## Kenapa endpoint bisa 500?

500 error di Express biasanya karena:
1. Express forward request ke FastAPI tapi **tidak attach data profil user**
2. FastAPI terima `profileSkills = ""` dan `jobSkills = ""` → semua score 0
3. Atau FastAPI belum jalan / model belum load

## Checklist fix untuk Express developer

### 1. Pastikan FastAPI jalan
```bash
curl http://localhost:8001/health
# Expected: {"status":"ok","model_loaded":true}
```

### 2. Pola request Express ke FastAPI untuk /api/v1/ai/job-fit

```typescript
// Di Express route handler POST /api/v1/ai/job-fit
router.post('/job-fit', authenticate, async (req, res) => {
  const { jobId } = req.body;
  const userId = req.user.id;

  // 1. Fetch user profile dari DB
  const user = await prisma.user.findUnique({
    where: { id: userId },
    include: { preferences: true, skills: true }
  });

  // 2. Fetch job dari DB
  const job = await prisma.job.findUnique({ where: { id: jobId } });
  if (!job) return res.status(404).json({ ... });

  // 3. Build payload untuk FastAPI
  const payload = {
    jobId: job.id,
    profileSkills: user.skills.map(s => s.name).join(', '),
    profileExperience: user.experienceLevel ?? 'Fresher',
    profileJobRole: user.currentRole ?? '',
    jobTitle: job.title,
    jobSkills: job.skillsClean ?? '',
    jobExperienceLevel: job.experienceLevel ?? 'ENTRY_LEVEL',
    jobWorkType: job.workType ?? '',
    jobEmploymentType: job.employmentType ?? '',
    jobProvince: job.province ?? '',
    jobCity: job.city ?? '',
    jobSalaryMin: job.salaryMin ?? null,
    jobSalaryMax: job.salaryMax ?? null,
    preferences: {
      workType: user.preferences?.workType ?? null,
      province: user.preferences?.province ?? null,
      salaryMin: user.preferences?.salaryMin ?? null,
    },
    language: 'id',
  };

  // 4. Kirim ke FastAPI
  try {
    const modelRes = await fetch('http://localhost:8001/ai/job-fit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!modelRes.ok) throw new Error(`Model API error: ${modelRes.status}`);
    const analysis = await modelRes.json();

    // 5. Return ke frontend sesuai openapi.json envelope
    return res.json({
      success: true,
      message: 'Analisis job fit berhasil',
      data: analysis,
      meta: null,
    });
  } catch (err) {
    return res.status(502).json({
      success: false,
      message: 'Model service tidak tersedia',
      data: null,
      error: { code: 'MODEL_SERVICE_ERROR', details: null, requestId: req.id }
    });
  }
});
```

### 3. Untuk /api/v1/ai/cv-analyzer (multipart)

```typescript
// Express menerima multipart dari user, lalu forward ke FastAPI juga multipart
router.post('/cv-analyzer', authenticate, upload.single('cvFile'), async (req, res) => {
  const { jobId, language, inputMode } = req.body;
  const userId = req.user.id;
  const cvFile = req.file; // buffer

  const user = await prisma.user.findUnique({ where: { id: userId }, include: { skills: true } });
  const job  = await prisma.job.findUnique({ where: { id: jobId } });
  if (!job) return res.status(404).json({ ... });

  // Build FormData untuk FastAPI
  const form = new FormData();
  form.append('job_id', jobId);
  form.append('language', language ?? 'id');
  form.append('profile_skills', user.skills.map(s => s.name).join(', '));
  form.append('profile_experience', user.experienceLevel ?? 'Fresher');
  form.append('profile_job_role', user.currentRole ?? '');
  form.append('cv_file', new Blob([cvFile.buffer], { type: cvFile.mimetype }), cvFile.originalname);

  const modelRes = await fetch('http://localhost:8001/ai/cv-analyzer', {
    method: 'POST',
    body: form,
  });
  const analysis = await modelRes.json();
  return res.json({ success: true, message: 'Analisis CV berhasil', data: analysis, meta: null });
});
```

### 4. Environment variable yang dibutuhkan

Tambahkan ke `.env` Express:
```
MODEL_API_URL=http://localhost:8001
```
