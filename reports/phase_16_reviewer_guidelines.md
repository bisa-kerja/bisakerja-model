# Phase 16 Reviewer Guidelines — Human Validation Labels

## Purpose
Create trusted evaluation-only labels for job-fit, ATS quality, recommendation relevance, and unsupported-claim rejection. These labels measure model quality. They must never become model input features.

## Required evidence
Reviewers must inspect the sampled pair fields, matched/missing skills, role family, experience band, and any available CV/job text evidence. Every label needs an evidence note that cites observed facts.

## Job-fit score bands
- `low` (`0-39`): weak role/skill/requirement match, critical missing requirements, or clear seniority mismatch.
- `medium` (`40-69`): partial match with useful overlap but important gaps remain.
- `high` (`70-100`): strong role, skill, requirement, and seniority alignment with only minor gaps.

Reviewers may assign a numeric `reviewer_job_fit_score` from `0` to `100`; the score band is derived from that value.

## ATS issue labels
Mark issue flags only when evidence is present:
- `parseability_issue`: CV text is empty, garbled, scanned-only, or materially incomplete.
- `section_completeness_issue`: core sections such as experience, education, skills, or contact are missing.
- `contact_detection_issue`: contact details are absent or unreadable.
- `date_detection_issue`: experience or education dates are absent, contradictory, or unreadable.
- `metric_evidence_issue`: achievements lack measurable impact where role expectations require evidence.
- `formatting_risk_issue`: tables, columns, graphics, or unusual ordering create parser risk.

## Recommendation relevance
Use `low`, `medium`, or `high` relevance for whether the job should appear in a candidate recommendation set for this profile/CV.

## Unsupported-claim rejection
Set `unsupported_claim_flag=true` when generated or proposed output claims a skill, seniority, hiring likelihood, language ability, credential, or experience not supported by evidence.

## Disagreement flags
Set `disagreement_flag=true` when reviewer confidence is low, another reviewer should adjudicate, evidence is insufficient, or labels conflict with the score band.

## Governance rules
- Keep human labels evaluation-only.
- Do not edit frozen label files; create a new label version instead.
- Do not expose raw reviewer notes to user-facing product copy.
- Do not use manual labels to construct profile/job features, prompts, embeddings, or training targets unless a future explicitly approved label-training phase creates a separate training dataset.
